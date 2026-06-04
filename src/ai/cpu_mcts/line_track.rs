use super::line_constants::{CELL_TO_LINES, DELTA, LINE_CELL_MASK};

#[derive(Clone, Copy)]
pub enum StepResult {
    Continue,
    Win,
    Draw,
}

/// Per-line counter state for one game position.
///
/// Layout:
///   att_cnt[3]: 76 lines × 2 bits for attacker stone count (0-3)
///   def_cnt[3]: same for defender
///   stones:     occupied cells (att | def)
///   att_reach_cells: empty cells where att already has 3-in-a-row
///   def_reach_cells: same for def
///
/// After each `apply()` the att/def roles are swapped so the caller never
/// has to track whose turn it is explicitly.
#[derive(Clone, Copy)]
pub struct LineTrack {
    pub att_cnt: [u64; 3],
    pub def_cnt: [u64; 3],
    pub stones: u64,
    pub att_reach_cells: u64,
    pub def_reach_cells: u64,
}

impl LineTrack {
    pub fn new(att: u64, def: u64) -> Self {
        let mut lt = LineTrack {
            att_cnt: [0u64; 3],
            def_cnt: [0u64; 3],
            stones: 0,
            att_reach_cells: 0,
            def_reach_cells: 0,
        };
        // Replay existing stones.  att first, def second so that after
        // replay att/def are in the correct perspective.
        let mut a = att;
        while a != 0 {
            let cell = a.trailing_zeros() as u8;
            lt.apply_init(cell, true);
            a &= a - 1;
        }
        let mut d = def;
        while d != 0 {
            let cell = d.trailing_zeros() as u8;
            lt.apply_init(cell, false);
            d &= d - 1;
        }
        // Recompute reach cells from scratch (apply_init doesn't track reach).
        lt.recompute_reach();
        lt
    }

    /// Place a stone during initialization (no reach tracking, no swap).
    fn apply_init(&mut self, cell: u8, is_att: bool) {
        self.stones |= 1u64 << cell;
        let lines = CELL_TO_LINES[cell as usize];
        let mut k = 0;
        while k < 13 {
            let line_id = lines[k];
            if line_id == 0xFF { break; }
            let bit_pos = line_id as usize * 2;
            let word = bit_pos / 64;
            let offset = bit_pos % 64;
            if is_att {
                self.att_cnt[word] += 1u64 << offset;
            } else {
                self.def_cnt[word] += 1u64 << offset;
            }
            k += 1;
        }
    }

    /// Recompute att_reach_cells and def_reach_cells from scratch.
    fn recompute_reach(&mut self) {
        self.att_reach_cells = 0;
        self.def_reach_cells = 0;
        let mut line_id = 0usize;
        while line_id < 76 {
            let ac = self.att_count_of(line_id as u8);
            let dc = self.def_count_of(line_id as u8);
            if ac == 3 && dc == 0 {
                let empty = LINE_CELL_MASK[line_id] & !self.stones;
                self.att_reach_cells |= empty;
            }
            if dc == 3 && ac == 0 {
                let empty = LINE_CELL_MASK[line_id] & !self.stones;
                self.def_reach_cells |= empty;
            }
            line_id += 1;
        }
    }

    /// Apply att's move at `cell`. Returns game outcome.
    /// After a Continue result the att/def roles are swapped.
    #[inline]
    pub fn apply(&mut self, cell: u8) -> StepResult {
        let action_mask = 1u64 << cell;

        // Win: att completes 4-in-a-row (was reach cell).
        if self.att_reach_cells & action_mask != 0 {
            self.stones |= action_mask;
            return StepResult::Win;
        }

        // Update att counts.
        for i in 0..3 {
            self.att_cnt[i] += DELTA[cell as usize][i];
        }
        self.stones |= action_mask;

        // Draw: board full (top layer filled).
        if (self.stones >> 48) & 0xFFFF == 0xFFFF {
            return StepResult::Draw;
        }

        // Update reach cells.
        self.update_reach(cell);

        // Swap sides.
        core::mem::swap(&mut self.att_cnt, &mut self.def_cnt);
        core::mem::swap(&mut self.att_reach_cells, &mut self.def_reach_cells);

        StepResult::Continue
    }

    /// Valid moves: bottom-most empty cell in each column.
    #[inline]
    pub fn valid_mask(&self) -> u64 {
        !self.stones & ((self.stones << 16) | 0xFFFF)
    }

    #[inline]
    fn update_reach(&mut self, cell: u8) {
        let lines = CELL_TO_LINES[cell as usize];
        let mut k = 0;
        while k < 13 {
            let line_id = lines[k];
            if line_id == 0xFF { break; }
            if self.att_count_of(line_id) == 3 {
                let empty = LINE_CELL_MASK[line_id as usize] & !self.stones;
                self.att_reach_cells |= empty;
            }
            k += 1;
        }
        // att played `cell`, so def can no longer reach there.
        self.def_reach_cells &= !( 1u64 << cell );
    }

    #[inline]
    pub fn att_count_of(&self, line_id: u8) -> u8 {
        let bit_pos = line_id as usize * 2;
        let word   = bit_pos / 64;
        let offset = bit_pos % 64;
        ((self.att_cnt[word] >> offset) & 0b11) as u8
    }

    #[inline]
    pub fn def_count_of(&self, line_id: u8) -> u8 {
        let bit_pos = line_id as usize * 2;
        let word   = bit_pos / 64;
        let offset = bit_pos % 64;
        ((self.def_cnt[word] >> offset) & 0b11) as u8
    }
}
