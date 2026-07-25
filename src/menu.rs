//! The context menu: a small panel of actions a right-click opens over the chrome,
//! and the editor's first pop-up.
//!
//! It is drawn into the window's own display list, after everything else so it
//! lands on top, the way the find bar floats over a pane. No second surface and no
//! `xdg_popup`: the menu is small and opens beside the rail at the window's edge,
//! so it has nowhere to spill, and the damage diff already carries it for free.
//!
//! While a menu is open it takes every press, whether or not the press is on it (a
//! pop-up grab in miniature): the one on an item runs it, and any other dismisses.
//! [`Menu::layout`] is the geometry both the painter and the pointer read, so an
//! item is clicked exactly where it is drawn, and there is no stored rectangle to
//! go stale when the window resizes under an open menu.

use crate::platform::freetype::{FontStyle, Fonts};
use crate::platform::geom::Rect;
use crate::platform::shape;
use crate::scale::Scale;
use crate::theme::{MENU_ITEM_PAD_X, MENU_ITEM_VPAD, MENU_PAD};

/// What a menu item does when it is clicked. The action names its subject, so a
/// menu holds everything needed to act on the thing it was opened over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MenuAction {
    /// Save rail vault `i` into the config file, so it is a vault at every launch.
    SaveVault(usize),
    /// Put the clipboard's picture on rail vault `i`'s chip.
    PasteIcon(usize),
    /// Take rail vault `i`'s picture away.
    ClearIcon(usize),
    /// Format the line(s) pane `pane` has under its caret or selection as task
    /// items, toggling the box on lines already tasks: the menu twin of Ctrl+L.
    ToggleTaskList { pane: usize },
    /// Run AI action `i` (an index into [`crate::ai::ACTIONS`]) over the focused pane.
    Ai(usize),
    /// Open the palette to type a free-form AI instruction for the focused pane.
    AiCustomPrompt,
    /// Pin or unpin tab `tab` of pane `pane`: the right-click on a tab toggles it.
    TogglePin { pane: usize, tab: usize },
}

/// One row of a menu: what it says, and what it does. The label is owned because a
/// menu entry can be composed at open time (an AI action carries its provider's name).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MenuItem {
    pub label: String,
    pub action: MenuAction,
}

/// An open context menu: its items and the point it was opened at. A right-click
/// with nothing to offer opens no menu at all, so a live one always has something
/// to click.
#[derive(Clone, Debug)]
pub struct Menu {
    items: Vec<MenuItem>,
    /// Where the right-click landed: the panel's top-left corner, before the clamp
    /// that keeps it inside the window.
    at: (i32, i32),
    /// Which item the pointer is over, for the hover fill.
    hovered: Option<usize>,
}

impl Menu {
    /// A menu of `items` opened at window point `at`.
    pub fn new(items: Vec<MenuItem>, at: (i32, i32)) -> Self {
        Self {
            items,
            at,
            hovered: None,
        }
    }

    pub fn items(&self) -> &[MenuItem] {
        &self.items
    }

    pub fn hovered(&self) -> Option<usize> {
        self.hovered
    }

    /// Point the menu's hover at `item`, reporting whether that changed anything
    /// (and so whether the frame has to be repainted).
    pub fn set_hovered(&mut self, item: Option<usize>) -> bool {
        let changed = self.hovered != item;
        self.hovered = item;
        changed
    }

    /// The action of item `i`, if there is one there.
    pub fn action(&self, i: usize) -> Option<MenuAction> {
        self.items.get(i).map(|item| item.action)
    }

    /// Where the menu lands in a `window`-sized window: the panel at the click
    /// point, sized to its widest label, and pushed back inside the window when the
    /// click was too near an edge for it to fit.
    pub fn layout(
        &self,
        fonts: &Fonts,
        body_size: u32,
        scale: Scale,
        window: (i32, i32),
    ) -> MenuLayout {
        let face = fonts.face(body_size, FontStyle::Regular);
        let m = fonts.metrics(body_size);
        let pad = scale.px(MENU_PAD);
        let widest = self
            .items
            .iter()
            .map(|item| shape::text_advance(face, &item.label).ceil() as i32)
            .max()
            .unwrap_or(0);
        let item_h = m.ascent + m.descent + 2 * scale.px(MENU_ITEM_VPAD);
        let w = widest + 2 * scale.px(MENU_ITEM_PAD_X) + 2 * pad;
        let h = item_h * self.items.len() as i32 + 2 * pad;

        // Shift, rather than flip, a menu that would hang off an edge: the panel is
        // small, so sliding it back keeps it under the pointer that opened it. A
        // window too small to hold it at all pins it to the top-left, where the
        // items are at least reachable.
        let x = (self.at.0).min(window.0 - w).max(0);
        let y = (self.at.1).min(window.1 - h).max(0);
        MenuLayout {
            panel: Rect { x, y, w, h },
            items: Rect {
                x: x + pad,
                y: y + pad,
                w: w - 2 * pad,
                h: item_h,
            },
            count: self.items.len(),
            baseline_offset: (item_h - (m.ascent + m.descent)) / 2 + m.ascent,
        }
    }
}

/// Where an open menu's panel and its items are on screen. The items are one
/// column of equal rows, so the layout holds the first and steps down to the rest
/// rather than carrying a rectangle per item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MenuLayout {
    /// The panel behind the items, border and padding included.
    pub panel: Rect,
    /// The first item's row; each one below it is `items.h` further down.
    items: Rect,
    count: usize,
    /// The text baseline inside an item's row, from the row's top.
    baseline_offset: i32,
}

impl MenuLayout {
    /// Item `i`'s row, whether or not the menu has that many items; only
    /// [`Self::item_at`] and the painter ask, and both stay within the count.
    pub fn item_rect(&self, i: usize) -> Rect {
        Rect {
            y: self.items.y + i as i32 * self.items.h,
            ..self.items
        }
    }

    /// The item at window point `(px, py)`, if the point is on one. A point on the
    /// panel's padding is on no item, which is how a click there dismisses the menu
    /// without acting.
    pub fn item_at(&self, px: f32, py: f32) -> Option<usize> {
        (0..self.count).find(|&i| self.item_rect(i).contains(px, py))
    }

    /// The baseline for item `i`'s label: its row, centered on the text.
    pub fn baseline(&self, i: usize) -> i32 {
        self.item_rect(i).y + self.baseline_offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A menu of `n` items at `at`, which is all the geometry cares about.
    fn menu(n: usize, at: (i32, i32)) -> Menu {
        Menu::new(
            (0..n)
                .map(|i| MenuItem {
                    label: "Save as vault".to_string(),
                    action: MenuAction::SaveVault(i),
                })
                .collect(),
            at,
        )
    }

    /// The app's real font stack at the body size: the panel is sized from real
    /// measured text, so the geometry is tested against it rather than a stand-in.
    fn test_fonts() -> Fonts {
        Fonts::with_config(
            &crate::platform::freetype::FontConfig::default(),
            &[crate::theme::FONT_SIZE],
            1.5,
        )
        .expect("open the default fonts")
    }

    /// The layout of `menu` in a `window`-sized window at scale 1.
    fn layout(menu: &Menu, window: (i32, i32)) -> MenuLayout {
        menu.layout(&test_fonts(), crate::theme::FONT_SIZE, Scale::ONE, window)
    }

    #[test]
    fn the_panel_opens_at_the_click_point() {
        let m = menu(1, (120, 200));
        let l = layout(&m, (1000, 800));
        assert_eq!((l.panel.x, l.panel.y), (120, 200));
    }

    #[test]
    fn a_panel_that_would_hang_off_an_edge_slides_back_in() {
        // A right-click near the bottom-right corner: the panel stays whole and
        // inside, rather than half of it landing outside the window.
        let m = menu(2, (995, 795));
        let l = layout(&m, (1000, 800));
        assert!(l.panel.x + l.panel.w <= 1000);
        assert!(l.panel.y + l.panel.h <= 800);
    }

    #[test]
    fn a_window_too_small_pins_the_panel_to_the_corner() {
        // Nothing fits, so the clamp must not push the panel off the other side
        // (a negative origin would put the items out of reach entirely).
        let m = menu(3, (10, 10));
        let l = layout(&m, (20, 20));
        assert_eq!((l.panel.x, l.panel.y), (0, 0));
    }

    #[test]
    fn items_stack_down_the_panel_without_overlapping() {
        let m = menu(3, (100, 100));
        let l = layout(&m, (1000, 800));
        for i in 0..2 {
            let (a, b) = (l.item_rect(i), l.item_rect(i + 1));
            assert_eq!(a.y + a.h, b.y);
            assert_eq!(a.h, b.h);
        }
        // Every item, padding included, sits inside the panel that backs them.
        let last = l.item_rect(2);
        assert!(l.items.y > l.panel.y);
        assert!(last.y + last.h < l.panel.y + l.panel.h);
    }

    #[test]
    fn item_at_finds_the_row_under_the_pointer() {
        let m = menu(2, (100, 100));
        let l = layout(&m, (1000, 800));
        for i in 0..2 {
            let r = l.item_rect(i);
            let mid = (r.x as f32 + r.w as f32 / 2.0, r.y as f32 + r.h as f32 / 2.0);
            assert_eq!(l.item_at(mid.0, mid.1), Some(i));
        }
    }

    #[test]
    fn item_at_misses_the_panels_padding_and_everything_outside() {
        let m = menu(1, (100, 100));
        let l = layout(&m, (1000, 800));
        // The panel's own padding is on no item: a click there dismisses.
        assert_eq!(l.item_at(l.panel.x as f32, l.panel.y as f32), None);
        // Well outside it.
        assert_eq!(l.item_at(500.0, 500.0), None);
    }

    #[test]
    fn a_label_sits_on_its_own_row() {
        let m = menu(2, (100, 100));
        let l = layout(&m, (1000, 800));
        for i in 0..2 {
            let r = l.item_rect(i);
            assert!(l.baseline(i) > r.y && l.baseline(i) < r.y + r.h);
        }
    }

    #[test]
    fn the_panel_is_wide_enough_for_its_labels() {
        let m = menu(1, (100, 100));
        let l = layout(&m, (1000, 800));
        let fonts = test_fonts();
        let face = fonts.face(crate::theme::FONT_SIZE, FontStyle::Regular);
        // The row holds the label plus its padding, and the panel holds the row.
        let text = shape::text_advance(face, "Save as vault").ceil() as i32;
        assert!(l.item_rect(0).w > text);
        assert!(l.panel.w > l.item_rect(0).w);
    }

    #[test]
    fn hover_reports_only_real_changes() {
        let mut m = menu(2, (0, 0));
        assert!(m.set_hovered(Some(1)));
        assert!(!m.set_hovered(Some(1)));
        assert!(m.set_hovered(None));
        assert_eq!(m.hovered(), None);
    }

    #[test]
    fn an_items_action_names_its_subject() {
        let m = menu(2, (0, 0));
        assert_eq!(m.action(1), Some(MenuAction::SaveVault(1)));
        assert_eq!(m.action(2), None);
    }
}
