//! The command line. An argument bnknotes does not take is an error, so a mistyped
//! flag does not open the editor as if it was never given.

use std::ffi::OsString;
use std::path::PathBuf;

pub const USAGE: &str = "usage: bnknotes [--demo-diff] [PATH] | bnknotes --gpu-probe";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mode {
    /// path is a note or a vault directory. demo_diff opens a demonstration diff
    /// review, for tuning the view without running an AI action.
    Run {
        path: Option<PathBuf>,
        demo_diff: bool,
    },
    /// Report what the compositor offers the GPU presentation path, then exit.
    GpuProbe,
}

impl Mode {
    /// args is the command line without the program name. The error is the first
    /// argument bnknotes does not take: an unknown flag, an empty argument, or a
    /// second path.
    pub fn parse<I>(args: I) -> Result<Self, String>
    where
        I: IntoIterator,
        I::Item: Into<OsString>,
    {
        let (mut path, mut demo_diff, mut probe) = (None, false, false);
        for arg in args {
            let arg: OsString = arg.into();
            match arg.to_str() {
                Some("--gpu-probe") => probe = true,
                Some("--demo-diff") => demo_diff = true,
                _ if path.is_some()
                    || arg.is_empty()
                    || arg.as_encoded_bytes().first() == Some(&b'-') =>
                {
                    return Err(arg.to_string_lossy().into_owned());
                }
                _ => path = Some(PathBuf::from(arg)),
            }
        }
        Ok(if probe {
            Self::GpuProbe
        } else {
            Self::Run { path, demo_diff }
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::cli::*;

    fn parse(list: &[&str]) -> Result<Mode, String> {
        Mode::parse(list.iter().copied())
    }

    fn run(path: Option<&str>, demo_diff: bool) -> Result<Mode, String> {
        Ok(Mode::Run {
            path: path.map(PathBuf::from),
            demo_diff,
        })
    }

    #[test]
    fn a_bare_launch_opens_the_editor_on_nothing() {
        assert_eq!(parse(&[]), run(None, false));
    }

    #[test]
    fn a_path_and_the_demo_flag_combine_in_either_order() {
        assert_eq!(parse(&["notes.md"]), run(Some("notes.md"), false));
        assert_eq!(parse(&["--demo-diff"]), run(None, true));
        assert_eq!(
            parse(&["notes.md", "--demo-diff"]),
            run(Some("notes.md"), true)
        );
        assert_eq!(
            parse(&["--demo-diff", "notes.md"]),
            run(Some("notes.md"), true)
        );
    }

    #[test]
    fn the_gpu_probe_wins_over_everything_given_with_it() {
        assert_eq!(parse(&["--gpu-probe"]), Ok(Mode::GpuProbe));
        assert_eq!(parse(&["notes.md", "--gpu-probe"]), Ok(Mode::GpuProbe));
        assert_eq!(parse(&["--gpu-probe", "--demo-diff"]), Ok(Mode::GpuProbe));
    }

    #[test]
    fn anything_else_is_refused_and_named() {
        assert_eq!(parse(&["--help"]), Err("--help".to_string()));
        assert_eq!(parse(&["-v"]), Err("-v".to_string()));
        assert_eq!(
            parse(&["a.md", "b.md"]),
            Err("b.md".to_string()),
            "one path only"
        );
        assert_eq!(parse(&[""]), Err(String::new()), "an empty argument too");
        // Dev flags are answered before this parse, and only in a dev build.
        assert_eq!(parse(&["--perf"]), Err("--perf".to_string()));
    }

    #[test]
    fn a_path_that_is_not_utf8_is_kept_as_bytes() {
        use std::os::unix::ffi::{OsStrExt, OsStringExt};

        let bytes = b"/home/user/notes/\xff.md".to_vec();
        let parsed = Mode::parse([OsString::from_vec(bytes.clone())]);
        let Ok(Mode::Run {
            path: Some(path), ..
        }) = parsed
        else {
            panic!("a non-UTF-8 path opens the editor on it: {parsed:?}");
        };
        assert_eq!(path.as_os_str().as_bytes(), bytes.as_slice());
    }
}
