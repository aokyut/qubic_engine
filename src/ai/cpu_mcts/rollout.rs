use super::line_track::{LineTrack, StepResult};
use super::rng::Xorshift64;

/// Rollout returning value relative to the *initial* att side at entry.
pub fn rollout_parity(mut state: LineTrack, rng: &mut Xorshift64) -> f32 {
    let mut parity: f32 = 1.0; // 1.0 means we're tracking from original att
    loop {
        let valid = state.valid_mask();
        if valid == 0 {
            return 0.5;
        }

        if state.att_reach_cells & valid != 0 {
            return parity; // current att wins
        }


        let must_block = state.def_reach_cells & valid;
        let action_mask = if must_block != 0 {
            if must_block.count_ones() >= 2 {
                return 1.0 - parity; // current def (opponent) wins
            }
            must_block
        } else {
            let safe = valid & !(state.def_reach_cells >> 16);
            let candidates = if safe != 0 { safe } else { valid };
            nth_set_bit(candidates, rng.next() % candidates.count_ones() as u64)
        };

        let cell = action_mask.trailing_zeros() as u8;
        match state.apply(cell) {
            StepResult::Win  => return parity,       // the side that just played wins = current parity
            StepResult::Draw => return 0.5,
            StepResult::Continue => {
                parity = 1.0 - parity; // sides swapped
            }
        }
    }
}

#[inline]
fn nth_set_bit(mut mask: u64, mut n: u64) -> u64 {
    while n > 0 {
        mask &= mask - 1;
        n -= 1;
    }
    mask & mask.wrapping_neg()
}
