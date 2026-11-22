//! Opening a clicked link in the user's default browser.
//!
//! The app resolves a click to a destination ([`crate::platform::link`] finds the
//! ones the text carries no markup for) and hands it here. We shell out to
//! `xdg-open`, the freedesktop way to reach whatever the user set as their default
//! handler, rather than hardcoding a browser.

use std::fmt::Write as _;
use std::io::Read as _;
use std::os::unix::fs::PermissionsExt;
use std::process::{Command, Stdio};

use crate::platform::error::{Error, Result};
use crate::platform::host::{self, Host};
use crate::platform::link;

/// The schemes we are willing to open, and so also exactly the schemes
/// [`crate::platform::link`] detects: a link the app decorates and then refuses to
/// follow is a worse bug than one it never decorated. Kept deliberately small: the
/// web and mail schemes text links to, plus `file:` for local references. Nothing
/// executable (`javascript:`, `data:`) is reachable.
pub const OPENABLE_SCHEMES: [&str; 4] = ["https://", "http://", "file://", "mailto:"];

/// Why a target is refused. A markdown link shows its text, not its target, so
/// every rule is about the target being what the user would take it for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Refusal {
    /// It starts with a dash xdg-open could read as an option.
    Dash,
    /// Its scheme is not one of OPENABLE_SCHEMES.
    Scheme,
    /// It carries a control character, or a bidi control that reorders how the text
    /// around it is shown, so the target that opens is not the text the user reads.
    /// Any other invisible character can only sit in the path, since the host must be
    /// ASCII, and there it changes nothing the user relies on.
    Hidden,
    /// Its host is one a browser would rewrite before resolving it (host::read), so
    /// the host reached is not the host written; or a file: target names another
    /// machine, where the opener would open the same path on this one.
    Host,
    /// A file: target's path could be read more than one way (decoded_file_path):
    /// a percent sign that is not a plain escape, an escape for / or a control
    /// character, a query, a backslash, or bytes that are not UTF-8.
    FilePath,
    /// A mailto: target sets a field beyond to, cc, bcc, subject and body. Mail
    /// clients have honored attach=, which turns a click into a draft with a local
    /// file already attached.
    MailField,
    /// A file: target resolves to a desktop entry, which an opener launches: a
    /// .desktop or .kdelnk name, or a file whose content begins like one, which is
    /// how some openers type it. Judged at the click, since it needs the filesystem.
    Launcher,
    /// A file: target resolves to a program an opener may run rather than show: an
    /// ELF binary or a #! script with an exec bit, or a Windows program (by name or
    /// by its MZ header), which wine runs with no exec bit at all. Judged at the
    /// click.
    Executable,
    /// A file: target names nothing on disk. Judged at the click.
    Missing,
}

impl Refusal {
    fn reason(self) -> &'static str {
        match self {
            Refusal::Dash => "it starts with a dash",
            Refusal::Scheme => "its scheme is not one the editor opens",
            Refusal::Hidden => "it carries a character that does not show",
            Refusal::Host => "its host is not the one it reads as",
            Refusal::FilePath => "its path could be read more than one way",
            Refusal::MailField => "it sets a mail field beyond to, cc, bcc, subject and body",
            Refusal::Launcher => "it is a desktop launcher",
            Refusal::Executable => "it is an executable file",
            Refusal::Missing => "it names no file",
        }
    }
}

/// The rules a target must pass before the app offers it to the pointer or hands
/// it to xdg-open.
pub fn judge(url: &str) -> core::result::Result<(), Refusal> {
    if url.starts_with('-') {
        return Err(Refusal::Dash);
    }
    if !has_allowed_scheme(url) {
        return Err(Refusal::Scheme);
    }
    if hides_a_character(url) {
        return Err(Refusal::Hidden);
    }
    let file = url.starts_with("file://");
    match host::read(url) {
        Host::Refused => return Err(Refusal::Host),
        // xdg-open strips only a lowercase localhost; any other host would leave a
        // relative path behind.
        Host::Name(name) if file && !(name.is_empty() || name == "localhost") => {
            return Err(Refusal::Host);
        }
        Host::Name(_) | Host::Absent => {}
    }
    if file && decoded_file_path(url).is_none() {
        return Err(Refusal::FilePath);
    }
    if url.starts_with("mailto:") && !mail_fields_are_allowed(url) {
        return Err(Refusal::MailField);
    }
    Ok(())
}

/// Whether a link's visible text agrees with its target: every URL the text shows
/// must begin with the target's host. The text is read the way a reader takes it:
/// past characters that draw nothing and a code span's backticks, in any case, with
/// a URL anywhere in it (an openable scheme, or www. at the start of a word), so
/// markup and quotes around it change nothing. A leading www. on either side is the
/// same site. A host shown with a non-ASCII letter never matches, since a target's
/// host is ASCII. Text that shows no URL agrees with any target.
pub fn text_agrees(text: &str, url: &str) -> bool {
    let target = match host::read(url) {
        Host::Name(name) => name.to_ascii_lowercase(),
        Host::Absent | Host::Refused => String::new(),
    };
    let (shown, breaks) = shown_text(text);
    shown_hosts(&shown, &breaks).all(|host| same_site(host, &target))
}

/// The text as a reader sees it, ASCII-lowercased: characters that draw nothing and
/// backticks dropped. The offsets returned are where something was dropped, which a
/// reader sees as a break between words.
fn shown_text(text: &str) -> (String, Vec<usize>) {
    let mut shown = String::with_capacity(text.len());
    let mut breaks = Vec::new();
    for c in text.chars() {
        if c == '`' || link::is_hidden(c) {
            if breaks.last() != Some(&shown.len()) {
                breaks.push(shown.len());
            }
        } else {
            shown.push(c.to_ascii_lowercase());
        }
    }
    (shown, breaks)
}

/// What starts a URL in shown text.
const URL_STARTS: [&str; 4] = ["https://", "http://", "file://", "www."];

/// The host each URL in shown text begins with: the bracketed address or the run of
/// letters, digits, dots, hyphens and underscores after its scheme (trailing dots
/// left to the sentence), or from a www. that starts a word. The rest of the URL, up
/// to the next whitespace, is its path, so a URL quoted in another one's path or
/// query names no host of its own.
fn shown_hosts<'a>(shown: &'a str, breaks: &'a [usize]) -> impl Iterator<Item = &'a str> {
    let mut at = 0;
    std::iter::from_fn(move || {
        while let Some(rest) = shown.get(at..).filter(|rest| !rest.is_empty()) {
            let start = URL_STARTS.iter().find(|start| {
                rest.starts_with(**start) && (**start != "www." || starts_word(shown, breaks, at))
            });
            let Some(start) = start else {
                at += rest.chars().next().map_or(1, char::len_utf8);
                continue;
            };
            let from = if *start == "www." {
                at
            } else {
                at + start.len()
            };
            let url = &shown[from..];
            let host = match url.strip_prefix('[') {
                Some(v6) => v6.find(']').map_or(url, |end| &url[..end + 2]),
                None => {
                    let end = url
                        .find(|c: char| !(c.is_alphanumeric() || matches!(c, '.' | '-' | '_')))
                        .unwrap_or(url.len());
                    url[..end].trim_end_matches('.')
                }
            };
            at = url
                .find(char::is_whitespace)
                .map_or(shown.len(), |space| from + space);
            return Some(host);
        }
        None
    })
}

/// Whether offset at in shown text starts a word: nothing before it, something that
/// is not a letter or digit, or a dropped character.
fn starts_word(shown: &str, breaks: &[usize], at: usize) -> bool {
    breaks.binary_search(&at).is_ok()
        || shown[..at]
            .chars()
            .next_back()
            .is_none_or(|before| !before.is_alphanumeric())
}

/// Whether two lowercased hosts are the same site, a leading www. aside.
fn same_site(shown: &str, target: &str) -> bool {
    let bare = |host: &'_ str| host.strip_prefix("www.").unwrap_or(host).to_string();
    bare(shown) == bare(target)
}

/// What a file: target reaches on disk, judged at the click: the path resolved
/// through every symlink, refused when it is a launcher, an executable or nothing,
/// else a URL encoded from that resolved path, so the file checked is the file
/// opened.
fn judge_file(url: &str) -> core::result::Result<String, Refusal> {
    let path = decoded_file_path(url).ok_or(Refusal::FilePath)?;
    let resolved = std::fs::canonicalize(path).map_err(|_| Refusal::Missing)?;
    let resolved = resolved
        .to_str()
        .filter(|path| plain_path(path))
        .ok_or(Refusal::FilePath)?;
    let name = resolved.rsplit('/').next().unwrap_or_default();
    let has_suffix = |suffixes: &[&str]| {
        name.rsplit_once('.')
            .is_some_and(|(_, ext)| suffixes.iter().any(|s| ext.eq_ignore_ascii_case(s)))
    };
    if has_suffix(&LAUNCHER_SUFFIXES) {
        return Err(Refusal::Launcher);
    }
    if has_suffix(&WINDOWS_SUFFIXES) {
        return Err(Refusal::Executable);
    }
    let meta = std::fs::metadata(resolved).map_err(|_| Refusal::Missing)?;
    // Only a regular file is read: opening a FIFO would wait for a writer.
    if meta.is_file() {
        let mut head = [0u8; 64];
        let len = std::fs::File::open(resolved)
            .and_then(|mut file| file.read(&mut head))
            .map_err(|_| Refusal::Missing)?;
        let head = &head[..len];
        if is_desktop_entry(head) {
            return Err(Refusal::Launcher);
        }
        let runnable = head.starts_with(b"\x7fELF") || head.starts_with(b"#!");
        let executable = meta.permissions().mode() & 0o111 != 0;
        if head.starts_with(b"MZ") || (executable && runnable) {
            return Err(Refusal::Executable);
        }
    }
    let fragment = url.split_once('#').map(|(_, fragment)| fragment);
    let mut target = encode_file_url(resolved);
    if let Some(fragment) = fragment {
        target.push('#');
        target.push_str(fragment);
    }
    Ok(target)
}

/// Names an opener launches as a desktop entry.
const LAUNCHER_SUFFIXES: [&str; 2] = ["desktop", "kdelnk"];

/// Names wine runs as a Windows program, with or without an exec bit.
const WINDOWS_SUFFIXES: [&str; 7] = ["exe", "dll", "cpl", "drv", "scr", "msi", "lnk"];

/// Whether head, a file's first bytes, begins the way a desktop entry does, which
/// is how an opener that types files by content recognizes one whatever its name.
fn is_desktop_entry(head: &[u8]) -> bool {
    let first = head.get(..32).unwrap_or(head);
    first.windows(15).any(|w| w == b"[Desktop Entry]")
        || [
            &b"[Desktop Action"[..],
            b"[KDE Desktop Entry]",
            b"# Config File",
            b"# KDE Config File",
        ]
        .iter()
        .any(|magic| head.starts_with(magic))
}

/// The path a file: URL names, decoded the one way every opener agrees on: no
/// query, every percent sign a two-digit escape for something other than / or a
/// control character, and a plain_path result. A fragment is not part of the path,
/// and judge_file carries it over to the URL it opens.
fn decoded_file_path(url: &str) -> Option<String> {
    let url = url.split_once('#').map_or(url, |(url, _)| url);
    if url.contains('?') {
        return None;
    }
    let rest = url.strip_prefix("file://")?;
    let path = rest.get(rest.bytes().position(|b| b == b'/')?..)?;
    if escapes_a_reserved_byte(path) {
        return None;
    }
    let bytes = percent_decoded(path).collect::<Option<Vec<u8>>>()?;
    let text = String::from_utf8(bytes).ok()?;
    plain_path(&text).then_some(text)
}

/// Whether path holds an escape that decodes to / or a control character. An
/// escaped slash reads as two different paths, and openers drop or reinterpret a
/// control character.
fn escapes_a_reserved_byte(path: &str) -> bool {
    let bytes = path.as_bytes();
    bytes.iter().enumerate().any(|(i, &b)| {
        b == b'%'
            && percent_decoded(path.get(i..i + 3).unwrap_or_default())
                .next()
                .flatten()
                .is_some_and(|d| d == b'/' || d.is_ascii_control())
    })
}

/// Whether path is absolute and holds no control character and no backslash,
/// which openers drop or reinterpret.
fn plain_path(path: &str) -> bool {
    path.starts_with('/') && !path.bytes().any(|b| b < 0x20 || b == 0x7F || b == b'\\')
}

/// A file:// URL for path with every byte outside the unreserved set and / percent
/// encoded, so an opener decodes it back to exactly path.
fn encode_file_url(path: &str) -> String {
    let mut url = String::from("file://");
    for byte in path.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~' | b'/') {
            url.push(char::from(byte));
        } else {
            let _ = write!(url, "%{byte:02X}");
        }
    }
    url
}

/// The header fields a mailto: link may set. In-Reply-To and References are what a
/// mailing-list archive's reply link sets.
const MAIL_FIELDS: [&str; 7] = [
    "to",
    "cc",
    "bcc",
    "subject",
    "body",
    "in-reply-to",
    "references",
];

/// Whether every field url sets is one of MAIL_FIELDS. A name is compared after
/// percent decoding and ASCII case folding, so ATTACH and %61ttach are both attach.
/// A second ?, which RFC 6068 does not allow and clients split differently,
/// refuses the link too. A link setting anything else is refused rather than
/// stripped, since a stripped link opens something other than what was written.
fn mail_fields_are_allowed(url: &str) -> bool {
    let Some((_, query)) = url.split_once('?') else {
        return true;
    };
    !query.contains('?')
        && query
            .split('&')
            .filter(|field| !field.is_empty())
            .all(|field| {
                let name = field.split_once('=').map_or(field, |(name, _)| name);
                MAIL_FIELDS.iter().any(|allowed| decodes_to(name, allowed))
            })
}

/// Whether the percent-encoded name decodes to allowed, without regard to ASCII case.
fn decodes_to(name: &str, allowed: &str) -> bool {
    let mut decoded = percent_decoded(name);
    let mut wanted = allowed.bytes();
    loop {
        match (decoded.next(), wanted.next()) {
            (None, None) => return true,
            (Some(Some(got)), Some(want)) if got.eq_ignore_ascii_case(&want) => {}
            _ => return false,
        }
    }
}

/// The bytes text percent-decodes to, strictly: a percent sign not followed by two
/// hex digits yields None, and the caller decides what that means.
fn percent_decoded(text: &str) -> impl Iterator<Item = Option<u8>> + '_ {
    let mut bytes = text.bytes();
    std::iter::from_fn(move || {
        let byte = bytes.next()?;
        if byte != b'%' {
            return Some(Some(byte));
        }
        let hex = |digit: u8| char::from(digit).to_digit(16).map(|d| d as u8);
        let high = bytes.next().and_then(hex);
        let low = bytes.next().and_then(hex);
        Some(high.zip(low).map(|(high, low)| high << 4 | low))
    })
}

/// The target to hand `xdg-open` for a clicked `url`, or why it is refused.
///
/// `xdg-open` is a thin shell wrapper, so the URL is judged first: a clicked link
/// can only ever open a web, file or mail target that is what it reads as, never a
/// `javascript:`/`data:`-style payload.
pub fn vet(url: &str) -> Result<String> {
    let refused =
        |refusal: Refusal| Error::msg(format!("refusing to open {url:?}: {}", refusal.reason()));
    judge(url).map_err(refused)?;
    if url.starts_with("file://") {
        judge_file(url).map_err(refused)
    } else {
        Ok(url.to_string())
    }
}

/// Open a vetted target in the default browser via `xdg-open`, detached from the
/// app. `xdg-open` launches the handler and exits almost immediately; it is reaped
/// on a short-lived detached thread so it never lingers as a zombie and the launch
/// never blocks the event loop.
pub fn launch(target: &str, token: Option<&str>) -> Result<()> {
    let mut child = opener(target, token).spawn()?;
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}

/// The xdg-open command for target. The activation token, when the compositor
/// gave one for the click, lets the browser's window come to the front; with none
/// the variable is cleared, so no stale token reaches it.
fn opener(target: &str, token: Option<&str>) -> Command {
    let mut command = Command::new("xdg-open");
    command
        .arg(target)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .env_remove("DESKTOP_STARTUP_ID");
    match token {
        Some(token) => command.env("XDG_ACTIVATION_TOKEN", token),
        None => command.env_remove("XDG_ACTIVATION_TOKEN"),
    };
    command
}

/// Whether [`vet`] would pass `url`. The app checks this before offering a
/// link to the pointer, so a target it would refuse never presents itself as
/// clickable in the first place.
pub fn can_open(url: &str) -> bool {
    judge(url).is_ok()
}

/// Whether url carries a control character or a bidi control.
fn hides_a_character(url: &str) -> bool {
    if url.is_ascii() {
        url.bytes().any(|b| b.is_ascii_control())
    } else {
        url.chars().any(|c| c.is_control() || is_bidi_control(c))
    }
}

/// The Bidi_Control characters (PropList.txt): the marks, embeddings, overrides and
/// isolates that reorder how the text around them is shown.
fn is_bidi_control(c: char) -> bool {
    matches!(
        c,
        '\u{061C}' | '\u{200E}' | '\u{200F}' | '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}'
    )
}

/// Whether `url` carries a scheme we are willing to open.
fn has_allowed_scheme(url: &str) -> bool {
    OPENABLE_SCHEMES
        .iter()
        .any(|scheme| url.starts_with(scheme))
}

#[cfg(test)]
mod tests {
    use crate::platform::browser::*;

    #[test]
    fn a_target_that_is_not_what_it_reads_as_is_refused() {
        for (url, refusal) in [
            ("https://example.com@evil.example/", Refusal::Host),
            ("http://2130706433/", Refusal::Host),
            ("https://pаypal.com/", Refusal::Host),
            ("file://other-machine/etc/hosts", Refusal::Host),
            ("file://LOCALHOST/etc/passwd", Refusal::Host),
            ("https://exa\u{200B}mple.com/", Refusal::Host),
            ("https://example.com/\u{202E}fdp.exe", Refusal::Hidden),
            ("https://example.com/a\u{2066}b", Refusal::Hidden),
            ("file:///home/user/notes/report%d.pdf", Refusal::FilePath),
            ("file:///home/user/a%2Fb.md", Refusal::FilePath),
            ("file:///home/user/a%00.md", Refusal::FilePath),
            ("file:///home/user/a%", Refusal::FilePath),
            ("https://example.com/a\tb", Refusal::Hidden),
            ("https://example.com/\u{1B}[31m", Refusal::Hidden),
            ("https://example.com/\u{85}", Refusal::Hidden),
            ("-https://example.com", Refusal::Dash),
            ("javascript:alert(1)", Refusal::Scheme),
        ] {
            assert_eq!(judge(url), Err(refusal), "{url:?}");
            assert!(!can_open(url), "{url:?} must not be offered to the pointer");
        }
    }

    #[test]
    fn link_text_that_shows_another_host_disagrees() {
        for (text, url) in [
            ("https://paypal.com/login", "https://evil.example/"),
            ("HTTPS://PAYPAL.COM", "https://evil.example/"),
            ("**https://paypal.com**", "https://evil.example/"),
            ("“https://paypal.com”", "https://evil.example/"),
            (
                "see https://paypal.com for details",
                "https://evil.example/",
            ),
            ("ht\u{200B}tps://paypal.com", "https://evil.example/"),
            ("https://pаypal.com", "https://xn--pypal-4ve.com/"),
            ("https://paypal.com.evil.example", "https://evil.example/"),
            ("www.paypal.com", "https://evil.example/"),
            ("https://example.com", "mailto:someone@example.com"),
            ("[`https:`//paypal.com", "https://evil.example/"),
            ("https:/`/`paypal.com", "https://evil.example/"),
            ("Login\u{3164}www.paypal.com", "https://evil.example/"),
            (
                "https://example.com and https://paypal.com",
                "https://example.com/",
            ),
        ] {
            assert!(!text_agrees(text, url), "{text:?} -> {url}");
        }
    }

    #[test]
    fn link_text_that_shows_no_other_host_agrees() {
        for (text, url) in [
            ("the docs", "https://evil.example/"),
            ("https://example.com/a", "https://example.com/b"),
            ("**https://Example.com**", "https://example.com/"),
            ("see https://example.com.", "https://example.com/"),
            ("www.example.com", "https://www.example.com/x"),
            ("awww.example.com", "https://evil.example/"),
            ("file:///home/user/a.md", "file:///home/user/b.md"),
            ("https://example.com", "https://example.com"),
            ("www.example.com", "https://example.com/"),
            ("https://example.com", "https://www.example.com/"),
            ("http://localhost:3000/", "http://localhost:3000/"),
        ] {
            assert!(text_agrees(text, url), "{text:?} -> {url}");
        }
        // A bare URL is its own text, whatever its path quotes.
        for url in [
            "https://web.archive.org/web/2020/https://example.com/",
            "https://www.google.com/search?q=www.example.com",
            "file:///srv/www.example.com/index.html",
            "http://[::1]:5173/",
            "https://example.com/a?next=https://example.com/b",
        ] {
            assert!(text_agrees(url, url), "{url}");
        }
    }

    #[test]
    fn a_mail_field_beyond_the_allowlist_is_refused() {
        for url in [
            "mailto:a@example.com?attach=/home/user/.ssh/id_ed25519",
            "mailto:a@example.com?ATTACH=/home/user/.ssh/id_ed25519",
            "mailto:a@example.com?%61ttach=/home/user/.ssh/id_ed25519",
            "mailto:a@example.com?subject=hi&x-mailer=x",
            "mailto:a@example.com?subject=hi?attach=x",
            "mailto:a@example.com?subject%zz=hi",
            "mailto:a@example.com?subject%26attach=x",
        ] {
            assert_eq!(judge(url), Err(Refusal::MailField), "{url}");
            assert!(!can_open(url), "{url}");
        }
        for url in [
            "mailto:a@example.com",
            "mailto:a@example.com?subject=Hi%20there&body=x",
            "mailto:a@example.com?CC=c@example.com&Bcc=e@example.com",
            "mailto:?to=a@example.com&%73ubject=x",
            "mailto:a@example.com?body",
            "mailto:a@example.com?",
            "mailto:a@example.com?subject=x&",
            "mailto:list@example.com?In-Reply-To=%3Cx%40y%3E&subject=Re%3A%20x",
            "mailto:a@example.com?body=x%26attach%3D%2Fetc%2Fpasswd",
        ] {
            assert_eq!(judge(url), Ok(()), "{url}");
        }
    }

    #[test]
    fn a_file_path_read_more_than_one_way_is_refused() {
        for url in [
            "file:///tmp/payload.desktop%0A",
            "file:///tmp/a%zz",
            "file:///tmp/a%2",
            "file:///tmp/a%5Cnb",
            "file:///tmp/a?b",
            "file:///tmp/a%7Fb",
            "file:///tmp/%FF",
        ] {
            assert_eq!(judge(url), Err(Refusal::FilePath), "{url}");
        }
        assert_eq!(
            decoded_file_path("file:///home/user/%D1%84%D0%B0%D0%B9%D0%BB.txt").as_deref(),
            Some("/home/user/файл.txt"),
        );
    }

    /// A directory of the test's own under the temp dir, emptied first.
    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("bnknotes-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir.canonicalize().expect("canonical scratch dir")
    }

    #[test]
    fn a_file_target_that_would_run_something_is_refused_at_the_click() {
        let dir = scratch("browser-files");
        let write = |name: &str, mode: u32| {
            let path = dir.join(name);
            std::fs::write(&path, "x").expect("write");
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).expect("chmod");
            path
        };
        let note = write("note.txt", 0o644);
        write("evil.desktop", 0o644);
        write("EVIL.DESKTOP", 0o644);
        write("setup.EXE", 0o644);
        write("shortcut.lnk", 0o644);
        let with = |name: &str, content: &[u8], mode: u32| {
            let path = write(name, mode);
            std::fs::write(&path, content).expect("write content");
            path
        };
        with("plain", b"MZ\x90\x00", 0o644);
        with("evil.dеsktop", b"[Desktop Entry]\nExec=x\n", 0o644);
        with("tool", b"\x7fELF\x02\x01", 0o755);
        with("run", b"#!/bin/sh\nid\n", 0o755);
        // On exFAT, NTFS or a CIFS share every file has the exec bit: a document is
        // still a document.
        let pdf = with("report.pdf", b"%PDF-1.7\n", 0o755);
        with("notes.md", b"# Notes\n", 0o755);
        // A harmless name that leads to the launcher: the link is judged by what it
        // reaches, not by what it is called.
        std::os::unix::fs::symlink(dir.join("evil.desktop"), dir.join("diagram.png"))
            .expect("symlink");
        let url = |name: &str| format!("file://{}/{name}", dir.display());

        assert_eq!(judge_file(&url("evil.desktop")), Err(Refusal::Launcher));
        assert_eq!(judge_file(&url("EVIL.DESKTOP")), Err(Refusal::Launcher));
        assert_eq!(judge_file(&url("diagram.png")), Err(Refusal::Launcher));
        assert_eq!(judge_file(&url("absent.txt")), Err(Refusal::Missing));
        assert_eq!(judge_file(&url("setup.EXE")), Err(Refusal::Executable));
        assert_eq!(judge_file(&url("shortcut.lnk")), Err(Refusal::Executable));
        assert_eq!(
            judge_file(&url("plain")),
            Err(Refusal::Executable),
            "an MZ header"
        );
        assert_eq!(
            judge_file(&url("evil.d%D0%B5sktop")),
            Err(Refusal::Launcher),
            "by content"
        );
        assert_eq!(judge_file(&url("tool")), Err(Refusal::Executable));
        assert_eq!(judge_file(&url("run")), Err(Refusal::Executable));
        assert_eq!(
            judge_file(&url("report.pdf")),
            Ok(encode_file_url(pdf.to_str().expect("utf-8 path")))
        );
        assert!(judge_file(&url("notes.md")).is_ok());
        assert_eq!(
            judge_file(&format!("{}#install", url("note.txt"))),
            Ok(format!(
                "{}#install",
                encode_file_url(note.to_str().expect("utf-8"))
            )),
            "a fragment carries over to the URL that opens",
        );
        assert_eq!(
            judge_file(&url("note.txt")),
            Ok(encode_file_url(note.to_str().expect("utf-8 path"))),
            "an ordinary file opens, under the URL of the file that was checked",
        );
        // A directory opens in the file manager, exec bit and all.
        assert!(judge_file(&format!("file://{}", dir.display())).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_file_url_encodes_back_to_exactly_its_path() {
        let path = "/home/user/My Notes/50%/файл#1.md";
        let url = encode_file_url(path);
        assert!(url.is_ascii());
        assert_eq!(
            url,
            "file:///home/user/My%20Notes/50%25/%D1%84%D0%B0%D0%B9%D0%BB%231.md"
        );
        let decoded: Option<Vec<u8>> = percent_decoded(&url["file://".len()..]).collect();
        assert_eq!(decoded.as_deref(), Some(path.as_bytes()));
    }

    #[test]
    fn an_ordinary_target_opens() {
        for url in [
            "https://example.com/a?q=1#f",
            "https://xn--pypal-4ve.com/",
            "https://de.wikipedia.org/wiki/Grüße",
            "http://localhost:3000/",
            "http://[::1]:8080/x",
            "mailto:someone@example.com",
            "file:///home/user/notes/a.md",
            "file://localhost/home/user/notes/a.md",
            "file:///home/user/My%20Notes/a.md",
            "https://fa.wikipedia.org/wiki/کتاب\u{200C}خانه",
            "https://emojipedia.org/red-heart/❤\u{FE0F}",
            "https://example.com/100%25",
            "file:///home/user/notes/100%25.pdf",
            "file:///home/user/manual.html#install",
        ] {
            assert_eq!(judge(url), Ok(()), "{url}");
        }
    }

    #[test]
    fn allowed_schemes_pass() {
        assert!(has_allowed_scheme("https://example.com"));
        assert!(has_allowed_scheme("http://example.com"));
        assert!(has_allowed_scheme("file:///etc/hosts"));
        assert!(has_allowed_scheme("mailto:a@b.com"));
    }

    #[test]
    fn disallowed_schemes_rejected() {
        assert!(!has_allowed_scheme("javascript:alert(1)"));
        assert!(!has_allowed_scheme("data:text/html,x"));
        assert!(!has_allowed_scheme("ftp://example.com"));
        assert!(!has_allowed_scheme("example.com"));
        assert!(!has_allowed_scheme("./relative/page.md"));
    }

    #[test]
    fn vet_refuses_what_xdg_open_must_never_see() {
        // A disallowed scheme or a leading dash never reaches xdg-open.
        assert!(vet("javascript:alert(1)").is_err());
        assert!(vet("-flag").is_err());
        assert!(vet("page.md").is_err());
        assert_eq!(
            vet("https://example.com/").ok().as_deref(),
            Some("https://example.com/")
        );
    }

    #[test]
    fn the_opener_passes_the_clicks_token_and_never_an_inherited_one() {
        let envs = |command: &Command| -> Vec<(String, Option<String>)> {
            let mut envs: Vec<_> = command
                .get_envs()
                .map(|(key, value)| {
                    let text = |s: &std::ffi::OsStr| s.to_string_lossy().into_owned();
                    (text(key), value.map(text))
                })
                .collect();
            envs.sort();
            envs
        };
        let with = opener("https://example.com/", Some("tok-1"));
        assert_eq!(
            envs(&with),
            [
                ("DESKTOP_STARTUP_ID".to_string(), None),
                (
                    "XDG_ACTIVATION_TOKEN".to_string(),
                    Some("tok-1".to_string())
                ),
            ]
        );
        let without = opener("https://example.com/", None);
        assert_eq!(
            envs(&without),
            [
                ("DESKTOP_STARTUP_ID".to_string(), None),
                ("XDG_ACTIVATION_TOKEN".to_string(), None),
            ],
            "cleared, not inherited"
        );
        assert_eq!(with.get_program(), "xdg-open");
        let args: Vec<_> = with.get_args().collect();
        assert_eq!(args, ["https://example.com/"]);
    }
}
