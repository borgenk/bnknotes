//! Keyboard translation backed by libxkbcommon (libxkbcommon.so.0).
//!
//! The compositor hands us an XKB keymap over wl_keyboard.keymap; we give it to
//! libxkbcommon and let it own layout, dead keys, and modifier state. Each
//! pressed key becomes a [`KeyAction`]: a few layout-independent special keys
//! dispatch on the physical evdev keycode, and everything else asks xkb for the
//! UTF-8 string the key produces under the current modifiers.
//!
//! Trimmed to the actions this milestone needs (character input, Backspace,
//! left/right).

use core::ffi::{c_char, c_int, c_void};
use core::ptr::NonNull;

use crate::error::{Error, Result};

/// Wayland passes raw evdev keycodes; XKB expects them offset by +8.
const EVDEV_OFFSET: u32 = 8;

mod keycode {
    pub const BACKSPACE: u32 = 14;
    pub const ENTER: u32 = 28;
    pub const KP_ENTER: u32 = 96;
    pub const HOME: u32 = 102;
    pub const UP: u32 = 103;
    pub const LEFT: u32 = 105;
    pub const RIGHT: u32 = 106;
    pub const END: u32 = 107;
    pub const DOWN: u32 = 108;
    pub const DELETE: u32 = 111;
}

const XKB_KEYMAP_FORMAT_TEXT_V1: u32 = 1;
/// Query the effective modifier state (depressed, latched, or locked).
const XKB_STATE_MODS_EFFECTIVE: u32 = 1 << 0;
/// libxkbcommon's canonical names for the Shift and Control modifiers.
const XKB_MOD_NAME_SHIFT: &[u8] = b"Shift\0";
const XKB_MOD_NAME_CTRL: &[u8] = b"Control\0";

#[link(name = "xkbcommon")]
#[allow(non_snake_case)]
extern "C" {
    fn xkb_context_new(flags: u32) -> *mut c_void;
    fn xkb_context_unref(ctx: *mut c_void);
    fn xkb_keymap_new_from_string(
        ctx: *mut c_void,
        string: *const c_char,
        format: u32,
        flags: u32,
    ) -> *mut c_void;
    fn xkb_keymap_unref(km: *mut c_void);
    fn xkb_state_new(km: *mut c_void) -> *mut c_void;
    fn xkb_state_unref(st: *mut c_void);
    fn xkb_state_update_mask(
        st: *mut c_void,
        depressed_mods: u32,
        latched_mods: u32,
        locked_mods: u32,
        depressed_layout: u32,
        latched_layout: u32,
        locked_layout: u32,
    ) -> u32;
    fn xkb_state_key_get_utf8(st: *mut c_void, key: u32, buf: *mut c_char, size: usize) -> c_int;
    fn xkb_state_mod_name_is_active(st: *mut c_void, name: *const c_char, type_: u32) -> c_int;
    fn xkb_state_key_get_one_sym(st: *mut c_void, key: u32) -> u32;
    fn xkb_keymap_key_repeats(keymap: *mut c_void, key: u32) -> c_int;
}

/// What a key press should do to the editor. The cursor-motion variants are the
/// ones a held Shift turns into selection-extending moves; the editing variants
/// ignore Shift.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyAction {
    Char(char),
    Enter,
    Backspace,
    Delete,
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    SelectAll,
    Copy,
    Cut,
    Paste,
    Undo,
    Redo,
    None,
}

/// Owns the libxkbcommon context, and (once a keymap arrives) the keymap and
/// state. Until then character input returns `None`, which is the correct
/// startup state before wl_keyboard.keymap.
pub struct Xkb {
    ctx: NonNull<c_void>,
    keymap: Option<NonNull<c_void>>,
    state: Option<NonNull<c_void>>,
}

impl Xkb {
    pub fn new() -> Result<Self> {
        // SAFETY: xkb_context_new accepts any flags and returns null on failure.
        let ctx = unsafe { xkb_context_new(0) };
        let ctx = NonNull::new(ctx).ok_or_else(|| Error::msg("xkb_context_new returned null"))?;
        Ok(Self {
            ctx,
            keymap: None,
            state: None,
        })
    }

    /// Install the keymap the compositor sent (its raw, NUL-terminated bytes).
    pub fn load_keymap(&mut self, bytes: &[u8], format: u32) -> Result<()> {
        if format != XKB_KEYMAP_FORMAT_TEXT_V1 {
            return Err(Error::msg("compositor sent an unsupported keymap format"));
        }
        // xkb_keymap_new_from_string wants a C string. The payload is
        // NUL-terminated; strip trailing NULs, reject an interior one, then add
        // exactly one terminator.
        let end = bytes.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
        let body = &bytes[..end];
        if body.contains(&0) {
            return Err(Error::msg("keymap contains an embedded NUL"));
        }
        let mut cstr = Vec::with_capacity(body.len() + 1);
        cstr.extend_from_slice(body);
        cstr.push(0);

        // SAFETY: ctx is valid; cstr is NUL-terminated and lives across the call.
        let keymap = unsafe {
            xkb_keymap_new_from_string(self.ctx.as_ptr(), cstr.as_ptr() as *const c_char, format, 0)
        };
        let keymap =
            NonNull::new(keymap).ok_or_else(|| Error::msg("xkb_keymap_new_from_string failed"))?;

        // SAFETY: keymap is valid.
        let state = unsafe { xkb_state_new(keymap.as_ptr()) };
        let state = match NonNull::new(state) {
            Some(s) => s,
            None => {
                // SAFETY: keymap is valid and we are releasing the ref we hold.
                unsafe { xkb_keymap_unref(keymap.as_ptr()) };
                return Err(Error::msg("xkb_state_new failed"));
            }
        };

        self.replace(keymap, state);
        Ok(())
    }

    /// Forward the modifier mask from wl_keyboard.modifiers to xkb so that
    /// Shift, Caps Lock, and layout groups affect the next key lookup.
    pub fn update_modifiers(&self, depressed: u32, latched: u32, locked: u32, group: u32) {
        let Some(state) = self.state else { return };
        // SAFETY: state is valid (NonNull, owned).
        unsafe {
            xkb_state_update_mask(state.as_ptr(), depressed, latched, locked, 0, 0, group);
        }
    }

    /// Whether Shift is currently active, so navigation keys extend a selection.
    pub fn shift_active(&self) -> bool {
        self.mod_active(XKB_MOD_NAME_SHIFT)
    }

    /// Whether Control is currently active, gating the editor shortcuts.
    pub fn ctrl_active(&self) -> bool {
        self.mod_active(XKB_MOD_NAME_CTRL)
    }

    fn mod_active(&self, name: &[u8]) -> bool {
        let Some(state) = self.state else {
            return false;
        };
        // SAFETY: state is valid; the name is a static NUL-terminated string.
        let active = unsafe {
            xkb_state_mod_name_is_active(
                state.as_ptr(),
                name.as_ptr() as *const c_char,
                XKB_STATE_MODS_EFFECTIVE,
            )
        };
        active > 0
    }

    /// Whether the keymap marks `keycode` as auto-repeating (text and editing
    /// keys do; modifiers and the like do not).
    pub fn key_repeats(&self, keycode: u32) -> bool {
        let Some(keymap) = self.keymap else {
            return false;
        };
        // SAFETY: keymap is valid; the call only reads the per-key repeat flag.
        let repeats = unsafe { xkb_keymap_key_repeats(keymap.as_ptr(), keycode + EVDEV_OFFSET) };
        repeats != 0
    }

    /// Translate a Wayland (evdev) keycode into an editor action.
    pub fn action(&self, keycode: u32) -> KeyAction {
        match keycode {
            keycode::BACKSPACE => return KeyAction::Backspace,
            keycode::DELETE => return KeyAction::Delete,
            keycode::ENTER | keycode::KP_ENTER => return KeyAction::Enter,
            keycode::LEFT => return KeyAction::Left,
            keycode::RIGHT => return KeyAction::Right,
            keycode::UP => return KeyAction::Up,
            keycode::DOWN => return KeyAction::Down,
            keycode::HOME => return KeyAction::Home,
            keycode::END => return KeyAction::End,
            _ => {}
        }

        // Control combos: read the layout's keysym so the shortcut follows the
        // letter, not the physical key, and never insert a control character.
        if self.ctrl_active() {
            return self.ctrl_action(keycode);
        }

        let Some(state) = self.state else {
            return KeyAction::None;
        };
        let mut buf = [0u8; 16];
        // SAFETY: state is valid; buf is writable and large enough for any
        // single key's UTF-8 output.
        let n = unsafe {
            xkb_state_key_get_utf8(
                state.as_ptr(),
                keycode + EVDEV_OFFSET,
                buf.as_mut_ptr() as *mut c_char,
                buf.len(),
            )
        };
        if n <= 0 {
            return KeyAction::None;
        }
        // Like snprintf, the return is the required length; clamp before slicing.
        let n = (n as usize).min(buf.len());
        let Ok(s) = core::str::from_utf8(&buf[..n]) else {
            return KeyAction::None;
        };
        // Only single, non-control scalars become input. Compose/dead-key
        // sequences that yield multiple scalars are dropped for now.
        let mut chars = s.chars();
        match (chars.next(), chars.next()) {
            (Some(c), None) if (c as u32) >= 0x20 && c != '\x7f' => KeyAction::Char(c),
            _ => KeyAction::None,
        }
    }

    /// Map a Ctrl+key combo to an editor shortcut, by the key's layout keysym.
    fn ctrl_action(&self, keycode: u32) -> KeyAction {
        let Some(state) = self.state else {
            return KeyAction::None;
        };
        // SAFETY: state is valid; the call returns the effective keysym.
        let sym = unsafe { xkb_state_key_get_one_sym(state.as_ptr(), keycode + EVDEV_OFFSET) };
        // Fold an upper-case keysym (Ctrl+Shift+letter) to lower-case ASCII.
        let sym = if (0x41..=0x5A).contains(&sym) {
            sym + 0x20
        } else {
            sym
        };
        match sym {
            0x61 => KeyAction::SelectAll, // a
            0x63 => KeyAction::Copy,      // c
            0x78 => KeyAction::Cut,       // x
            0x76 => KeyAction::Paste,     // v
            // Ctrl+Z undoes; Ctrl+Shift+Z and Ctrl+Y redo.
            0x7A if self.shift_active() => KeyAction::Redo, // Z
            0x7A => KeyAction::Undo,                        // z
            0x79 => KeyAction::Redo,                        // y
            _ => KeyAction::None,
        }
    }

    fn replace(&mut self, keymap: NonNull<c_void>, state: NonNull<c_void>) {
        if let Some(old) = self.state.replace(state) {
            // SAFETY: old came from xkb_state_new and is freed once.
            unsafe { xkb_state_unref(old.as_ptr()) };
        }
        if let Some(old) = self.keymap.replace(keymap) {
            // SAFETY: old came from xkb_keymap_new_* and is freed once.
            unsafe { xkb_keymap_unref(old.as_ptr()) };
        }
    }
}

impl Drop for Xkb {
    fn drop(&mut self) {
        // SAFETY: each pointer came from its matching xkb_*_new and is freed
        // exactly once; we hold the only references.
        unsafe {
            if let Some(s) = self.state.take() {
                xkb_state_unref(s.as_ptr());
            }
            if let Some(k) = self.keymap.take() {
                xkb_keymap_unref(k.as_ptr());
            }
            xkb_context_unref(self.ctx.as_ptr());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_keys_dispatch_without_a_keymap() {
        let xkb = Xkb::new().expect("xkb context");
        assert_eq!(xkb.action(keycode::BACKSPACE), KeyAction::Backspace);
        assert_eq!(xkb.action(keycode::DELETE), KeyAction::Delete);
        assert_eq!(xkb.action(keycode::ENTER), KeyAction::Enter);
        assert_eq!(xkb.action(keycode::KP_ENTER), KeyAction::Enter);
        assert_eq!(xkb.action(keycode::LEFT), KeyAction::Left);
        assert_eq!(xkb.action(keycode::RIGHT), KeyAction::Right);
        assert_eq!(xkb.action(keycode::UP), KeyAction::Up);
        assert_eq!(xkb.action(keycode::DOWN), KeyAction::Down);
        assert_eq!(xkb.action(keycode::HOME), KeyAction::Home);
        assert_eq!(xkb.action(keycode::END), KeyAction::End);
    }

    #[test]
    fn character_keys_without_a_keymap_are_none() {
        // 'a' is evdev keycode 30; with no keymap loaded there is no state.
        let xkb = Xkb::new().expect("xkb context");
        assert_eq!(xkb.action(30), KeyAction::None);
    }

    #[test]
    fn shift_is_inactive_without_a_keymap() {
        let xkb = Xkb::new().expect("xkb context");
        assert!(!xkb.shift_active());
    }
}
