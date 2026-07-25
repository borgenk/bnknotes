//! What a key press means to the editor: the binding table, and nothing else.
//!
//! [`crate::platform::xkb`] answers only what the machine knows about a key (the
//! keysym it resolves to, the text it types, which modifiers are held). Turning that
//! into an editor action is a decision about *this* app, so it lives here, at the app
//! layer, where you would look for it to rebind something.
//!
//! Two dispatches, in order:
//!
//!   1. **Physical keys** (Backspace, Enter, the arrows, Home/End, Page Up/Down)
//!      dispatch on the evdev keycode, because their meaning does not move with the
//!      layout: the Left arrow is the Left arrow on every keyboard. Most of them jump
//!      further when Ctrl is held, so Ctrl+Left moves a word and Ctrl+Home reaches the
//!      document start.
//!   2. **Ctrl combos** dispatch on the *keysym* ([`ctrl_action_for_sym`]), so a
//!      shortcut follows the letter the user sees rather than the key's position: Ctrl+W
//!      closes the tab on QWERTY and on Dvorak alike.
//!
//! Anything else is text, and goes through the compose machine as a [`KeyAction::Char`].

use crate::platform::xkb::{keycode, Xkb};

/// What a key press should do to the editor. The cursor-motion variants (Left, Right,
/// Up, Down, Home, End, the word/document variants, and the page moves) are the ones a
/// held Shift turns into selection-extending moves; the editing variants ignore Shift.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyAction {
    Char(char),
    Enter,
    /// Insert indentation (a soft tab); see the app's handling.
    Tab,
    Backspace,
    Delete,
    /// Delete to the previous/next word boundary (Ctrl+Backspace / Ctrl+Delete).
    DeleteWordBack,
    DeleteWordForward,
    /// Delete the whole line(s) the cursor or selection touches (Ctrl+Shift+K).
    DeleteLine,
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    /// Move one word left/right (Ctrl+Left / Ctrl+Right).
    WordLeft,
    WordRight,
    /// Move to the very start/end of the buffer (Ctrl+Home / Ctrl+End).
    DocStart,
    DocEnd,
    /// Move up/down one viewport height (Page Up / Page Down).
    PageUp,
    PageDown,
    SelectAll,
    Copy,
    Cut,
    Paste,
    Undo,
    Redo,
    Save,
    /// Open (or, while open, close) the find bar (Ctrl+F).
    Find,
    /// Open (or, while open, close) the palette (Ctrl+K).
    Palette,
    /// Open (or, while open, close) the palette on the vault's contents
    /// (Ctrl+Shift+F).
    SearchVault,
    /// Open the find bar with its replace row (Ctrl+H).
    Replace,
    /// Leave a transient mode such as the find bar (Escape).
    Escape,
    /// Pane management: toggle the file tree (Ctrl+B), toggle the second content pane
    /// (Ctrl+\), focus the tree / editor A / editor B (Ctrl+0/1/2), and close the
    /// active tab (Ctrl+W).
    ToggleTree,
    ToggleSecondPane,
    FocusTree,
    FocusPaneA,
    FocusPaneB,
    CloseTab,
    /// Fold or unfold the heading section the cursor sits in (Ctrl+period), and
    /// unfold every section in the document (Ctrl+Shift+period).
    ToggleFold,
    UnfoldAll,
    /// Format the cursor's line, or every line the selection touches, as a task
    /// item; a line already a task toggles its checkbox (Ctrl+L).
    ToggleTaskList,
    None,
}

/// The editor action for a Wayland (evdev) keycode, read against `xkb`'s current
/// layout and modifier state.
pub fn action(xkb: &Xkb, keycode: u32) -> KeyAction {
    // The navigation and deletion keys do bigger jumps when Ctrl is held: Ctrl+Arrow
    // moves by word, Ctrl+Home/End to the document ends, and Ctrl+Backspace/Delete
    // remove a whole word.
    let ctrl = xkb.ctrl_active();
    match keycode {
        keycode::BACKSPACE => {
            return ctrl_or(ctrl, KeyAction::DeleteWordBack, KeyAction::Backspace);
        }
        keycode::DELETE => {
            return ctrl_or(ctrl, KeyAction::DeleteWordForward, KeyAction::Delete);
        }
        keycode::ESC => return KeyAction::Escape,
        keycode::ENTER | keycode::KP_ENTER => return KeyAction::Enter,
        // Ctrl+Tab falls through to the (empty) Ctrl handler so it does not insert;
        // plain Tab inserts indentation.
        keycode::TAB if !ctrl => return KeyAction::Tab,
        keycode::LEFT => return ctrl_or(ctrl, KeyAction::WordLeft, KeyAction::Left),
        keycode::RIGHT => return ctrl_or(ctrl, KeyAction::WordRight, KeyAction::Right),
        keycode::UP => return KeyAction::Up,
        keycode::DOWN => return KeyAction::Down,
        keycode::HOME => return ctrl_or(ctrl, KeyAction::DocStart, KeyAction::Home),
        keycode::END => return ctrl_or(ctrl, KeyAction::DocEnd, KeyAction::End),
        keycode::PAGEUP => return KeyAction::PageUp,
        keycode::PAGEDOWN => return KeyAction::PageDown,
        _ => {}
    }

    // Any other Ctrl combo is a shortcut, keyed by the layout's keysym so it follows
    // the letter, not the position. Ctrl never types a character.
    if ctrl {
        return xkb.key_sym(keycode).map_or(KeyAction::None, |sym| {
            ctrl_action_for_sym(sym, xkb.shift_active())
        });
    }

    // Everything else is text, composed if the layout composes it.
    xkb.key_text(keycode)
        .map_or(KeyAction::None, KeyAction::Char)
}

/// The shortcut table for a Ctrl-held key, keyed by its layout keysym so a non-QWERTY
/// layout binds the letter the user sees, not the position. Held apart from the keymap
/// state so the table itself is testable.
fn ctrl_action_for_sym(sym: u32, shift: bool) -> KeyAction {
    // Fold an upper-case keysym (Ctrl+Shift+letter) to lower-case ASCII.
    let sym = if (0x41..=0x5A).contains(&sym) {
        sym + 0x20
    } else {
        sym
    };
    match sym {
        0x61 => KeyAction::SelectAll,            // a
        0x63 => KeyAction::Copy,                 // c
        0x66 if shift => KeyAction::SearchVault, // Ctrl+Shift+F: search the vault
        0x66 => KeyAction::Find,                 // f
        0x68 => KeyAction::Replace,              // h
        0x6C => KeyAction::ToggleTaskList,       // l: format the line(s) as a task
        0x73 => KeyAction::Save,                 // s
        0x78 => KeyAction::Cut,                  // x
        0x76 => KeyAction::Paste,                // v
        0x6B if shift => KeyAction::DeleteLine,  // Ctrl+Shift+K
        0x6B => KeyAction::Palette,              // k: open the palette
        0x62 => KeyAction::ToggleTree,           // b: toggle sidebar
        0x77 => KeyAction::CloseTab,             // w: close the active tab
        0x5C => KeyAction::ToggleSecondPane,     // backslash: split
        // Period folds the section at the cursor.
        // Shifted it unfolds the whole document, which most layouts report as the
        // `>` keysym rather than a shifted period, so both spellings bind.
        0x3E => KeyAction::UnfoldAll,          // '>'
        0x2E if shift => KeyAction::UnfoldAll, // period, where shift leaves it alone
        0x2E => KeyAction::ToggleFold,         // period
        0x30 => KeyAction::FocusTree,          // 0: focus tree
        0x31 => KeyAction::FocusPaneA,         // 1: focus editor A
        0x32 => KeyAction::FocusPaneB,         // 2: focus editor B
        // Ctrl+Z undoes; Ctrl+Shift+Z and Ctrl+Y redo.
        0x7A if shift => KeyAction::Redo, // Z
        0x7A => KeyAction::Undo,          // z
        0x79 => KeyAction::Redo,          // y
        _ => KeyAction::None,
    }
}

/// The Ctrl-held action when `ctrl`, else the plain one. The navigation and delete
/// keys that jump further with Ctrl (word/document motion, word delete) share this
/// shape.
fn ctrl_or(ctrl: bool, with_ctrl: KeyAction, plain: KeyAction) -> KeyAction {
    if ctrl {
        with_ctrl
    } else {
        plain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_keys_dispatch_without_a_keymap() {
        // With no keymap there is no modifier state, so Ctrl reads as inactive and the
        // navigation keys take their plain (non-word) meaning. These dispatch on the
        // keycode alone, which is exactly why they still resolve here.
        let xkb = Xkb::new().expect("xkb context");
        assert_eq!(action(&xkb, keycode::BACKSPACE), KeyAction::Backspace);
        assert_eq!(action(&xkb, keycode::DELETE), KeyAction::Delete);
        assert_eq!(action(&xkb, keycode::ESC), KeyAction::Escape);
        assert_eq!(action(&xkb, keycode::ENTER), KeyAction::Enter);
        assert_eq!(action(&xkb, keycode::KP_ENTER), KeyAction::Enter);
        assert_eq!(action(&xkb, keycode::TAB), KeyAction::Tab);
        assert_eq!(action(&xkb, keycode::LEFT), KeyAction::Left);
        assert_eq!(action(&xkb, keycode::RIGHT), KeyAction::Right);
        assert_eq!(action(&xkb, keycode::UP), KeyAction::Up);
        assert_eq!(action(&xkb, keycode::DOWN), KeyAction::Down);
        assert_eq!(action(&xkb, keycode::HOME), KeyAction::Home);
        assert_eq!(action(&xkb, keycode::END), KeyAction::End);
        assert_eq!(action(&xkb, keycode::PAGEUP), KeyAction::PageUp);
        assert_eq!(action(&xkb, keycode::PAGEDOWN), KeyAction::PageDown);
    }

    #[test]
    fn character_keys_without_a_keymap_are_none() {
        // 'a' is evdev keycode 30; with no keymap there is no state to type from.
        let xkb = Xkb::new().expect("xkb context");
        assert_eq!(action(&xkb, 30), KeyAction::None);
    }

    #[test]
    fn ctrl_shortcuts_map_by_keysym() {
        // The letters the editor binds, by their lower-case keysym.
        let sym = |c: char| c as u32;
        assert_eq!(ctrl_action_for_sym(sym('w'), false), KeyAction::CloseTab);
        assert_eq!(ctrl_action_for_sym(sym('b'), false), KeyAction::ToggleTree);
        assert_eq!(ctrl_action_for_sym(sym('s'), false), KeyAction::Save);
        assert_eq!(ctrl_action_for_sym(sym('f'), false), KeyAction::Find);
        assert_eq!(
            ctrl_action_for_sym(sym('l'), false),
            KeyAction::ToggleTaskList
        );
        assert_eq!(ctrl_action_for_sym(sym('1'), false), KeyAction::FocusPaneA);
        // An unbound key is inert rather than falling through to another action.
        assert_eq!(ctrl_action_for_sym(sym('q'), false), KeyAction::None);
    }

    #[test]
    fn ctrl_shift_folds_the_keysym_and_picks_the_shifted_action() {
        let sym = |c: char| c as u32;
        // Ctrl+Shift+letter arrives upper-case; the table folds it, so a binding
        // without a shifted meaning still fires (Ctrl+Shift+W closes the tab).
        assert_eq!(ctrl_action_for_sym(sym('W'), true), KeyAction::CloseTab);
        // Where shift does mean something, it selects the shifted action.
        assert_eq!(ctrl_action_for_sym(sym('z'), false), KeyAction::Undo);
        assert_eq!(ctrl_action_for_sym(sym('Z'), true), KeyAction::Redo);
        assert_eq!(ctrl_action_for_sym(sym('k'), false), KeyAction::Palette);
        assert_eq!(ctrl_action_for_sym(sym('K'), true), KeyAction::DeleteLine);
        // Ctrl+F finds in the document, Ctrl+Shift+F across the vault. The shifted arm
        // comes first, since the shifted key would otherwise fall through to the find bar.
        assert_eq!(ctrl_action_for_sym(sym('f'), false), KeyAction::Find);
        assert_eq!(ctrl_action_for_sym(sym('F'), true), KeyAction::SearchVault);
    }

    #[test]
    fn every_bound_ctrl_letter_is_distinct() {
        // A duplicate keysym in the table would silently shadow the earlier arm, so a
        // rebind that collides fails here rather than in the user's hands.
        let bound: Vec<KeyAction> = "acfhklsxvbw\\012zy."
            .chars()
            .map(|c| ctrl_action_for_sym(c as u32, false))
            .collect();
        for (i, a) in bound.iter().enumerate() {
            assert_ne!(*a, KeyAction::None, "arm {i} should be bound");
            assert_eq!(
                bound.iter().filter(|b| *b == a).count(),
                1,
                "{a:?} is bound twice"
            );
        }
    }
}
