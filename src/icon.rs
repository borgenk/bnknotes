//! Vault chip icons: a picture you paste, for the rail to wear.
//!
//! A chip says which vault it is with a color and two letters. An icon is the
//! third thing it can say, and the best judge of what is worth saying is the person
//! looking at it: snip the part of a picture you mean with whatever tool puts a
//! picture on the clipboard, then paste it onto the chip.
//!
//! The pasted picture is kept beside the config rather than in the vault. It is
//! chrome, not a note: a vault's attachment folder holds what its notes point at,
//! and an icon is pointed at by the rail. One file per vault, replaced when a new
//! one is pasted and deleted when the icon is cleared, so the folder holds exactly
//! the icons in use.
//!
//! A chip is square and a snip is whatever shape it was: the square shown is the
//! middle of the picture, worked out once when it is pasted (the crop the bake
//! reads) rather than every time it is drawn.

use std::path::{Path, PathBuf};

use crate::config::{config_dir, vault_name};
use crate::file;
use crate::platform::error::{Error, Result};
use crate::render::image::{self, Image, Limits};

/// A chip's picture: the file it was pasted into, and the square of it the chip
/// shows. The square is in the picture's own pixels, clamped into it when it is
/// baked, so a file edited to another size afterwards still gives a chip.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VaultIcon {
    pub source: PathBuf,
    pub x: u32,
    pub y: u32,
    pub side: u32,
}

/// The folder the pasted icons are kept in, beside the config and the state file.
/// `None` when the config directory does not resolve, and then an icon cannot be
/// kept at all.
fn icons_dir() -> Option<PathBuf> {
    Some(config_dir()?.join("icons"))
}

/// Keep `bytes` as `vault`'s icon and describe the chip it makes.
///
/// The bytes are decoded first, so what is kept is a picture the editor can draw
/// back rather than whatever was on the clipboard, and the dimension limits apply
/// here as everywhere. One file per vault: the name is the vault's own plus a hash
/// of its path, so two vaults called `notes` do not fight over it, and the extension
/// follows what the bytes actually are.
pub fn paste(vault: &Path, bytes: &[u8]) -> Result<VaultIcon> {
    let image = Image::decode(bytes, Limits::default())?;
    let ext = image::extension(bytes).ok_or_else(|| Error::msg("not a picture this reads"))?;
    let dir = icons_dir().ok_or_else(|| Error::msg("no XDG_CONFIG_HOME or HOME to keep it in"))?;
    std::fs::create_dir_all(&dir)?;
    let source = dir.join(format!("{}.{ext}", icon_name(vault)));
    file::save_atomic(&source, bytes)?;
    let (x, y, side) = center_square(image.width, image.height);
    Ok(VaultIcon { source, x, y, side })
}

/// Delete the file behind `icon`, for an icon being cleared or replaced. Best
/// effort: a file already gone is the state wanted, and one that will not delete
/// costs a stray picture in a folder, not a chip.
pub fn discard(icon: &VaultIcon) {
    let _ = std::fs::remove_file(&icon.source);
}

/// The biggest square in the middle of a `w` by `h` picture. A chip is square and a
/// snip is whatever shape the pointer drew, so the middle is what it shows: the
/// thing being snipped is what was being aimed at.
fn center_square(w: u32, h: u32) -> (u32, u32, u32) {
    let side = w.min(h);
    ((w - side) / 2, (h - side) / 2, side)
}

/// The file name for `vault`'s icon, without an extension: its own name, so the
/// folder can be read, plus a hash of its full path, so two vaults with one name
/// still get one file each.
fn icon_name(vault: &Path) -> String {
    let name: String = vault_name(vault)
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect();
    format!("{name}-{:08x}", fnv1a(vault.as_os_str().as_encoded_bytes()))
}

/// FNV-1a, 32 bits. A file name has to mean the same thing at the next launch, so
/// this is spelled out rather than taken from the standard library's hasher, whose
/// output is explicitly not promised to be stable.
fn fnv1a(bytes: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for b in bytes {
        hash ^= *b as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

#[cfg(test)]
mod tests {
    use crate::icon::*;

    #[test]
    fn the_middle_square_of_a_wide_picture() {
        // A snip is whatever shape the pointer drew; the chip shows the middle of it.
        assert_eq!(center_square(300, 120), (90, 0, 120));
    }

    #[test]
    fn the_middle_square_of_a_tall_picture() {
        assert_eq!(center_square(120, 300), (0, 90, 120));
    }

    #[test]
    fn a_square_picture_is_all_of_it() {
        assert_eq!(center_square(200, 200), (0, 0, 200));
    }

    #[test]
    fn an_odd_size_leaves_the_spare_pixel_at_the_end() {
        // 301 wide, 120 square: 90 either side and one over, rather than a crop that
        // runs a pixel past the picture.
        let (x, y, side) = center_square(301, 120);
        assert_eq!((x, y, side), (90, 0, 120));
        assert!(x + side <= 301);
    }

    #[test]
    fn a_one_pixel_picture_is_that_pixel() {
        assert_eq!(center_square(1, 1), (0, 0, 1));
    }

    #[test]
    fn a_vaults_icon_is_named_after_it() {
        let name = icon_name(Path::new("/home/user/notes"));
        assert!(name.starts_with("notes-"), "{name}");
    }

    #[test]
    fn two_vaults_of_one_name_get_two_names() {
        // The whole reason the path is hashed in: a rail can hold two vaults called
        // `notes`, and one file between them would put one's picture on the other.
        let a = icon_name(Path::new("/home/user/work/notes"));
        let b = icon_name(Path::new("/home/user/personal/notes"));
        assert_ne!(a, b);
        assert!(a.starts_with("notes-") && b.starts_with("notes-"));
    }

    #[test]
    fn a_name_is_the_same_at_the_next_launch() {
        // It names a file that outlives the process, so it cannot be a hash that is
        // free to change between runs or releases.
        assert_eq!(
            icon_name(Path::new("/home/user/notes")),
            icon_name(Path::new("/home/user/notes"))
        );
        assert_eq!(fnv1a(b"/home/user/notes"), fnv1a(b"/home/user/notes"));
    }

    #[test]
    fn a_name_holds_nothing_a_path_would_mind() {
        // A vault named with a slash or a space must not become a name that walks out
        // of the icons folder or needs quoting to look at.
        let name = icon_name(Path::new("/home/user/My Notes & Things"));
        assert!(
            name.chars().all(|c| c.is_alphanumeric() || c == '-'),
            "{name}"
        );
    }

    #[test]
    fn fnv_tells_paths_apart() {
        assert_ne!(fnv1a(b"/home/user/a"), fnv1a(b"/home/user/b"));
        assert_ne!(fnv1a(b""), fnv1a(b"x"));
    }
}
