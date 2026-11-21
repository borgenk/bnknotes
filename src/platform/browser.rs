//! Opening a clicked link in the user's default browser.
//!
//! The app resolves a click to a destination ([`crate::platform::link`] finds the
//! ones the text carries no markup for) and hands it here. We shell out to
//! `xdg-open`, the freedesktop way to reach whatever the user set as their default
//! handler, rather than hardcoding a browser.

use std::process::{Command, Stdio};

use crate::platform::error::{Error, Result};
use crate::platform::host::{self, Host};

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
    /// A file: target's path has a percent sign that is not a plain escape, or an
    /// escape for %, / or a control character. xdg-open decodes the path and uses it
    /// as a printf format, and an encoded slash reads as two different paths.
    FilePath,
}

impl Refusal {
    fn reason(self) -> &'static str {
        match self {
            Refusal::Dash => "it starts with a dash",
            Refusal::Scheme => "its scheme is not one the editor opens",
            Refusal::Hidden => "it carries a character that does not show",
            Refusal::Host => "its host is not the one it reads as",
            Refusal::FilePath => "its path could be read more than one way",
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
    if file && !escapes_are_plain(url) {
        return Err(Refusal::FilePath);
    }
    Ok(())
}

/// Whether every percent sign in url starts a two-digit escape that decodes to
/// something other than %, / or a control character.
fn escapes_are_plain(url: &str) -> bool {
    let bytes = url.as_bytes();
    let hex = |b: u8| char::from(b).to_digit(16);
    bytes.iter().enumerate().all(|(i, &b)| {
        if b != b'%' {
            return true;
        }
        let (Some(&hi), Some(&lo)) = (bytes.get(i + 1), bytes.get(i + 2)) else {
            return false;
        };
        match (hex(hi), hex(lo)) {
            (Some(hi), Some(lo)) => {
                let decoded = (hi * 16 + lo) as u8;
                !matches!(decoded, b'%' | b'/') && !decoded.is_ascii_control()
            }
            _ => false,
        }
    })
}

/// Open `url` in the default browser via `xdg-open`, detached from the app.
///
/// `xdg-open` is a thin shell wrapper, so the URL is judged first: a clicked link
/// can only ever open a web, file or mail target that is what it reads as, never a
/// `javascript:`/`data:`-style payload. `xdg-open` launches the handler and exits
/// almost immediately; it is reaped on a short-lived detached thread so it never
/// lingers as a zombie and the launch never blocks the event loop.
pub fn open_url(url: &str) -> Result<()> {
    if let Err(refusal) = judge(url) {
        return Err(Error::msg(format!(
            "refusing to open {url:?}: {}",
            refusal.reason()
        )));
    }
    let mut child = Command::new("xdg-open")
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}

/// Whether [`open_url`] would launch `url`. The app checks this before offering a
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
            ("file:///home/user/notes/100%25.pdf", Refusal::FilePath),
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
    fn open_url_refuses_unvetted_targets_without_spawning() {
        // A disallowed scheme or a leading dash never reaches xdg-open.
        assert!(open_url("javascript:alert(1)").is_err());
        assert!(open_url("-flag").is_err());
        assert!(open_url("page.md").is_err());
    }
}
