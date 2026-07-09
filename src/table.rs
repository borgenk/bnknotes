//! Grid-cursor navigation for markdown tables: the byte-offset moves that let a
//! caret behave as if it lived in a rendered cell, with the pipes and padding
//! hidden. Pure functions of the rope, shared by the editing model (which moves
//! and edits) and the layout (which positions the caret and hit-tests clicks), so
//! the two never disagree on where a cell begins or ends.
//!
//! A caret in a table always sits in a cell's trimmed content range; the moves
//! here step from cell to cell over the structural bytes (pipes and their padding)
//! in one jump, and report when a move would leave the table so the caller can
//! step the caret out into the surrounding prose.

use crate::markdown;
use crate::rope::Rope;

/// A table block's logical-line extent: the header row, the delimiter row just
/// below it, and one past the last body row (the body is `delim + 1 .. end`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Block {
    pub header: usize,
    pub delim: usize,
    pub end: usize,
}

/// One cell the caret can sit in: its row (logical line), column index, the number
/// of cells in that row, and the trimmed content byte range `[start, end]`
/// (absolute). The caret clamps into `[start, end]`; an empty cell has `start ==
/// end`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Cell {
    pub block: Block,
    pub row: usize,
    pub col: usize,
    pub cols: usize,
    pub start: usize,
    pub end: usize,
}

/// The content text of logical line `line` (without its trailing newline).
fn line_text(rope: &Rope, line: usize) -> String {
    let (s, e) = rope.line_content_range(line);
    rope.byte_slice(s..e)
}

/// The table block containing logical line `i`, or `None`. Detection scans the
/// contiguous run of pipe-bearing lines around `i`: the table begins at the first
/// line whose successor is a delimiter row (a valid header) and runs to the end of
/// that pipe run. Lines above the header (prose that happens to share the run) and
/// lines with no pipe are not in a table.
pub fn block_at(rope: &Rope, i: usize) -> Option<Block> {
    let n = rope.len_lines();
    if i >= n || !markdown::is_table_row(&line_text(rope, i)) {
        return None;
    }
    let mut run_start = i;
    while run_start > 0 && markdown::is_table_row(&line_text(rope, run_start - 1)) {
        run_start -= 1;
    }
    let mut run_end = i;
    while run_end + 1 < n && markdown::is_table_row(&line_text(rope, run_end + 1)) {
        run_end += 1;
    }
    let mut header = run_start;
    loop {
        if header >= run_end {
            return None;
        }
        if markdown::table_head(&line_text(rope, header), &line_text(rope, header + 1)).is_some() {
            break;
        }
        header += 1;
    }
    if i < header {
        return None;
    }
    Some(Block {
        header,
        delim: header + 1,
        end: run_end + 1,
    })
}

/// The absolute, trimmed content byte ranges of table row `line`'s cells.
pub fn row_cells(rope: &Rope, line: usize) -> Vec<(usize, usize)> {
    let (ls, _) = rope.line_content_range(line);
    let text = line_text(rope, line);
    let mut cells = Vec::new();
    markdown::table_row_cells(&text, &mut cells);
    cells.iter().map(|&(s, e)| (ls + s, ls + e)).collect()
}

/// The cell the caret at `offset` sits in, or `None` when `offset` is not in a
/// table's header or body row (the delimiter row has no cells). An offset in the
/// structure before or after a cell snaps to the nearest cell.
pub fn cell_at(rope: &Rope, offset: usize) -> Option<Cell> {
    let line = rope.byte_to_line(offset.min(rope.len()));
    let block = block_at(rope, line)?;
    if line == block.delim {
        return None;
    }
    let cells = row_cells(rope, line);
    if cells.is_empty() {
        return None;
    }
    let cols = cells.len();
    // The first cell whose content ends at or past `offset`; a trailing offset
    // falls to the last cell.
    let col = cells
        .iter()
        .position(|&(_, e)| offset <= e)
        .unwrap_or(cols - 1);
    let (start, end) = cells[col];
    Some(Cell {
        block,
        row: line,
        col,
        cols,
        start,
        end,
    })
}

/// The drawn row after `row` (skipping the delimiter), or `None` at the last row.
fn next_row(block: Block, row: usize) -> Option<usize> {
    let candidate = if row == block.header {
        block.delim + 1
    } else {
        row + 1
    };
    (candidate < block.end).then_some(candidate)
}

/// The drawn row before `row` (skipping the delimiter), or `None` at the header.
fn prev_row(block: Block, row: usize) -> Option<usize> {
    if row == block.header {
        None
    } else if row == block.delim + 1 {
        Some(block.header)
    } else {
        Some(row - 1)
    }
}

/// The content start of `row`'s first cell (its line start if the row is empty).
fn first_cell_start(rope: &Rope, row: usize) -> usize {
    row_cells(rope, row)
        .first()
        .map_or_else(|| rope.line_to_byte(row), |c| c.0)
}

/// The content end of `row`'s last cell (its content end if the row is empty).
fn last_cell_end(rope: &Rope, row: usize) -> usize {
    row_cells(rope, row)
        .last()
        .map_or_else(|| rope.line_content_range(row).1, |c| c.1)
}

/// The content start of the cell after `cell` in reading order (next column, or
/// the first column of the next row), or `None` when `cell` is the table's last.
pub fn next_cell_start(rope: &Rope, cell: Cell) -> Option<usize> {
    let cells = row_cells(rope, cell.row);
    if cell.col + 1 < cells.len() {
        return Some(cells[cell.col + 1].0);
    }
    next_row(cell.block, cell.row).map(|r| first_cell_start(rope, r))
}

/// The content start of the cell before `cell` in reading order, or `None` when
/// `cell` is the table's first.
pub fn prev_cell_start(rope: &Rope, cell: Cell) -> Option<usize> {
    if cell.col > 0 {
        return Some(row_cells(rope, cell.row)[cell.col - 1].0);
    }
    prev_row(cell.block, cell.row).map(|r| first_cell_start(rope, r))
}

/// The content end of the cell before `cell` in reading order, or `None` when
/// `cell` is the table's first (used by a Left move at a cell's start).
pub fn prev_cell_end(rope: &Rope, cell: Cell) -> Option<usize> {
    if cell.col > 0 {
        return Some(row_cells(rope, cell.row)[cell.col - 1].1);
    }
    prev_row(cell.block, cell.row).map(|r| last_cell_end(rope, r))
}

/// The offset just past the table (leaving it to the right or below): the start of
/// the line after the block, clamped to the buffer.
pub fn exit_after(rope: &Rope, block: Block) -> usize {
    rope.line_to_byte(block.end).min(rope.len())
}

/// The offset just before the table (leaving it to the left or above): the content
/// end of the line above the header, or 0 when the table opens the document.
pub fn exit_before(rope: &Rope, block: Block) -> usize {
    if block.header == 0 {
        0
    } else {
        rope.line_content_range(block.header - 1).1
    }
}

/// The byte range to remove from `row` to delete its column `col`: the cell's
/// content plus one adjacent separator pipe, so the row loses exactly one column.
/// `None` when the row has no such column. The first column drops its trailing
/// separator, a later column its leading one, so a trimmed table stays trimmed.
pub fn column_cut(rope: &Rope, row: usize, col: usize) -> Option<(usize, usize)> {
    let cells = row_cells(rope, row);
    if col >= cells.len() {
        return None;
    }
    if col == 0 {
        let start = cells[0].0;
        let end = cells.get(1).map_or(rope.line_content_range(row).1, |c| c.0);
        Some((start, end))
    } else {
        Some((cells[col - 1].1, cells[col].1))
    }
}

/// The text of a fresh empty table row of `cols` columns, without a leading or
/// trailing newline: outer pipes and one space of room per cell, so a typed
/// character lands inside the cell.
pub fn empty_row(cols: usize) -> String {
    let mut s = String::from("|");
    for _ in 0..cols.max(1) {
        s.push_str("  |");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rope(s: &str) -> Rope {
        Rope::from_str(s)
    }

    const DOC: &str =
        "intro\n\n| A | B | C |\n| - | :-: | -: |\n| ab | cd | ef |\n| gh | ij | kl |\n\nafter\n";

    /// The line index of the first row of the sample table's header.
    fn header_line(r: &Rope) -> usize {
        r.byte_to_line(r.byte_slice(0..r.len()).find("| A ").unwrap())
    }

    #[test]
    fn detects_the_block_from_any_of_its_rows_and_not_the_prose() {
        let r = rope(DOC);
        let h = header_line(&r);
        let block = block_at(&r, h).expect("a block");
        assert_eq!(block.header, h);
        assert_eq!(block.delim, h + 1);
        assert_eq!(block.end, h + 4, "header + 2 body rows span 4 lines");
        // Every row of the table sees the same block; the delimiter included.
        for row in [h, h + 1, h + 2, h + 3] {
            assert_eq!(block_at(&r, row), Some(block), "row {row}");
        }
        // Prose above and below, and the blank lines, are not in the table.
        for line in [0, 1, h - 1, block.end, block.end + 1] {
            assert_eq!(block_at(&r, line), None, "line {line}");
        }
    }

    #[test]
    fn cell_at_finds_the_column_and_skips_the_delimiter() {
        let r = rope(DOC);
        let h = header_line(&r);
        let (ls, _) = r.line_content_range(h + 2); // "| ab | cd | ef |"
        let cells = row_cells(&r, h + 2);
        assert_eq!(cells.len(), 3);
        // A byte inside "cd" resolves to column 1 of that row.
        let mid = cells[1].0;
        let cell = cell_at(&r, mid).expect("a cell");
        assert_eq!((cell.row, cell.col, cell.cols), (h + 2, 1, 3));
        assert_eq!((cell.start, cell.end), cells[1]);
        // The delimiter row has no cells.
        assert_eq!(cell_at(&r, r.line_to_byte(h + 1)), None);
        let _ = ls;
    }

    #[test]
    fn cell_steps_walk_columns_then_rows_then_off_the_ends() {
        let r = rope(DOC);
        let h = header_line(&r);
        // Start in the header's first cell.
        let first = cell_at(&r, row_cells(&r, h)[0].0).unwrap();
        let c1 = row_cells(&r, h)[1].0;
        assert_eq!(next_cell_start(&r, first), Some(c1), "to the next column");
        // Last cell of the header steps to the first cell of the first body row
        // (skipping the delimiter).
        let last_header = cell_at(&r, row_cells(&r, h)[2].0).unwrap();
        assert_eq!(
            next_cell_start(&r, last_header),
            Some(row_cells(&r, h + 2)[0].0),
            "wraps to the next row, past the delimiter"
        );
        // The very last cell steps off the table.
        let last = cell_at(&r, row_cells(&r, h + 3)[2].0).unwrap();
        assert_eq!(next_cell_start(&r, last), None);
        // The first cell has no previous.
        assert_eq!(prev_cell_start(&r, first), None);
        assert_eq!(prev_cell_end(&r, first), None);
    }

    #[test]
    fn exits_land_just_outside_the_block() {
        let r = rope(DOC);
        let h = header_line(&r);
        let block = block_at(&r, h).unwrap();
        // Before the table: the end of the blank line above the header.
        assert_eq!(exit_before(&r, block), r.line_content_range(h - 1).1);
        // After the table: the start of the blank line below the last body row.
        assert_eq!(exit_after(&r, block), r.line_to_byte(block.end));
    }

    #[test]
    fn a_table_opening_the_document_exits_before_to_the_start() {
        let r = rope("| a | b |\n| - | - |\n| 1 | 2 |\n");
        let block = block_at(&r, 0).unwrap();
        assert_eq!(exit_before(&r, block), 0);
    }

    #[test]
    fn empty_row_has_the_right_column_count() {
        // Wrapped in a fresh line, an empty row must split into `cols` cells.
        let r = rope(&format!("x\n{}\n", empty_row(3)));
        assert_eq!(row_cells(&r, 1).len(), 3);
    }

    #[test]
    fn column_cut_targets_a_cell_and_one_separator() {
        let r = rope("| a | b | c |\n| - | - | - |\n");
        let cells = row_cells(&r, 0);
        // The first column drops its trailing separator: start to next cell start.
        assert_eq!(column_cut(&r, 0, 0), Some((cells[0].0, cells[1].0)));
        // A later column drops its leading separator: previous end to this end.
        assert_eq!(column_cut(&r, 0, 1), Some((cells[0].1, cells[1].1)));
        assert_eq!(column_cut(&r, 0, 9), None, "out of range");
    }
}
