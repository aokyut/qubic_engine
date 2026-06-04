pub mod arena;
pub mod envelope;
pub mod line_constants;
pub mod line_track;
pub mod rollout;
pub mod rng;
pub mod search;

use std::time::Instant;

use crate::ai::EvalAndActF;
use crate::board::uboard::{Action, UBoard};
use crate::board::{Board, GetAction};
use arena::{expand, Arena};
use envelope::MctsConfig;
use line_track::LineTrack;
use rng::Xorshift64;
use search::search;

pub struct CpuMcts {
    pub max_iter:         u32,
    pub explore_c:        f32,
    pub arena_capacity:   usize,
    pub expand_threshold: u32,
}

impl CpuMcts {
    pub fn new(max_iter: u32, expand_threshold: u32) -> Self {
        Self {
            max_iter,
            explore_c: 1.414,
            arena_capacity: 2_000_000,
            expand_threshold: expand_threshold,
        }
    }

    /// Run MCTS and return (action_bitmask, visit_count, value) per root child.
    pub fn run(&self, board: UBoard) -> Vec<(Action, u32, f32)> {
        let (att, def) = board;
        let mut arena = Arena::new(self.arena_capacity);
        let root_idx = 0u32;

        let initial_state = LineTrack::new(att, def);
        let cfg = MctsConfig { explore_c: self.explore_c, expand_threshold: self.expand_threshold };
        let mut rng = Xorshift64::new(0x6c62272e07bb0142);

        // Expand root before iterations.
        let valid = initial_state.valid_mask();
        expand(&mut arena, root_idx, valid);

        for _ in 0..self.max_iter {
            search(&mut arena, root_idx, initial_state, &mut rng, &cfg);
        }

        // Collect root children results.
        let root = arena.get(root_idx);
        let nc = root.num_children;
        let fc = root.first_child;
        (0..nc).map(|i| {
            let child = arena.get(fc + i as u32);
            let action = 1u64 << child.action;
            let na = child.visits;
            let val = if na > 0 {
                1.0 - child.total_val / na as f32
            } else {
                0.0
            };
            (action, na, val)
        }).collect()
    }
}

impl GetAction for CpuMcts {
    fn get_action(&self, b: &Board) -> u8 {
        let (att, def) = b.get_att_def();
        let t = Instant::now();
        let mut results = self.run((att, def));
        let t = t.elapsed().as_millis();
        results.sort_by(|a, b| b.1.cmp(&a.1));
        if cfg!(feature="view"){
            for (action, num, val) in results.iter(){
                println!("[Action:{:>2}]val:{val:.3}, na:{num:>10}", action.trailing_zeros() % 16);
            }
            println!("time:{t}ms, node:{}node, nps:{}[n/s]", self.max_iter, 1_000 * self.max_iter as u128 / t);
        }
        if results.is_empty() { return 0; }
        results[0].0.trailing_zeros() as u8
    }
}

impl EvalAndActF for CpuMcts {
    fn eval_and_act(&self, b: &Board) -> (u8, f32) {
        let (att, def) = b.get_att_def();
        let t = Instant::now();
        let mut results = self.run((att, def));
        let t = t.elapsed().as_millis();
        results.sort_by(|a, b| b.1.cmp(&a.1));
        if results.is_empty() { return (0, 0.5); }
        (results[0].0.trailing_zeros() as u8, results[0].2)
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::line_constants::{CELL_TO_LINES, LINE_CELL_MASK};
    use super::line_track::{LineTrack, StepResult};
    use super::rollout::rollout_parity;
    use super::rng::Xorshift64;

    // Helper: verify every cell belongs to at least 1 line.
    #[test]
    fn cell_to_lines_non_empty() {
        for cell in 0u8..64 {
            assert!(CELL_TO_LINES[cell as usize][0] != 0xFF,
                "cell {cell} has no lines");
        }
    }

    // Verify line 0 (X-line, y=0, z=0) has cells 0,1,2,3.
    #[test]
    fn line_cell_mask_line0() {
        let mask = LINE_CELL_MASK[0];
        assert_eq!(mask, 0b1111u64, "line 0 mask = {:b}", mask);
    }

    // Verify Z-line z=0,y=0,x=0 (line 32) cells 0,16,32,48.
    #[test]
    fn line_cell_mask_z_line() {
        let mask = LINE_CELL_MASK[32];
        let expected = (1u64) | (1u64 << 16) | (1u64 << 32) | (1u64 << 48);
        assert_eq!(mask, expected);
    }

    // 4 in a row → Win.
    #[test]
    fn four_in_a_row_win() {
        let mut lt = LineTrack::new(0, 0);
        // Place 3 att stones at cells 0,1,2 (X-line, y=0, z=0).
        assert!(matches!(lt.apply(0), StepResult::Continue));
        // swap → def plays something far away
        assert!(matches!(lt.apply(60), StepResult::Continue));
        assert!(matches!(lt.apply(1), StepResult::Continue));
        assert!(matches!(lt.apply(61), StepResult::Continue));
        assert!(matches!(lt.apply(2), StepResult::Continue));
        assert!(matches!(lt.apply(62), StepResult::Continue));
        // Now att has 3 in a row on line 0; att_reach_cells should include cell 3.
        let result = lt.apply(3);
        assert!(matches!(result, StepResult::Win), "expected Win after 4th stone");
    }

    // 3 in a row sets reach cells.
    #[test]
    fn reach_cells_set_after_three() {
        let mut lt = LineTrack::new(0, 0);
        lt.apply(0);  // att
        lt.apply(60); // def
        lt.apply(1);  // att
        lt.apply(61); // def
        lt.apply(2);  // att → now att has 3 on line 0
        // After swap, att_reach (for next att = def's turn perspective) ≠ 0?
        // Actually: after att plays 2, update_reach fires → att_reach_cells |= cell 3.
        // Then swap. From the new att's (def's) perspective, def_reach_cells has cell 3.
        assert!(lt.def_reach_cells & (1u64 << 3) != 0,
            "cell 3 should be in def_reach after att 3-in-a-row");
    }

    // Rollout from empty board stays in [0,1].
    #[test]
    fn rollout_range() {
        let mut rng = Xorshift64::new(42);
        for _ in 0..100 {
            let lt = LineTrack::new(0, 0);
            let v = rollout_parity(lt, &mut rng);
            assert!(v == 0.0 || v == 0.5 || v == 1.0,
                "rollout returned {v}");
        }
    }

    // Full MCTS run produces valid results.
    #[test]
    fn mcts_run_basic() {
        let mcts = CpuMcts::new(200, 1);
        let results = mcts.run((0u64, 0u64));
        assert!(!results.is_empty());
        let total_visits: u32 = results.iter().map(|r| r.1).sum();
        assert!(total_visits > 0);
        for &(action, na, val) in &results {
            assert!(action != 0);
            assert!(val >= 0.0 && val <= 1.0, "val={val}");
        }
    }
}
