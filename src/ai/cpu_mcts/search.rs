use super::arena::{expand, Arena, Node};
use super::envelope::{select_child, update_envelope, MctsConfig};
use super::line_track::{LineTrack, StepResult};
use super::rollout::rollout_parity;
use super::rng::Xorshift64;

/// Recursive MCTS search.
///
/// Returns a value in [0,1] from the perspective of the side whose turn it
/// is at `node_idx` (the att side of `state`).
/// 1.0 = that side wins, 0.0 = that side loses.
pub fn search(
    arena: &mut Arena,
    node_idx: u32,
    state: LineTrack,        // owned copy — no undo needed
    rng: &mut Xorshift64,
    cfg: &MctsConfig,
) -> f32 {
    let node = arena.get(node_idx);

    if node.is_terminal() {
        // Terminal value is stored as total_val/visits (always 0.0 or 0.5 or 1.0).
        return node.q();
    }

    // --- EXPAND (deferred by threshold) or ROLLOUT ---
    if !node.is_expanded() {
        if node.visits < cfg.expand_threshold {
            let v = rollout_parity(state, rng);
            let node = arena.get_mut(node_idx);
            node.visits += 1;
            node.total_val += v;
            return v;
        }
        let valid = state.valid_mask();
        expand(arena, node_idx, valid);
    }

    // --- SELECT child ---
    let child_offset = select_child(arena.get(node_idx), node_idx, arena, cfg);
    let fc = arena.get(node_idx).first_child;
    let child_idx = fc + child_offset as u32;
    let child_action = arena.get(child_idx).action;

    // --- ADVANCE state ---
    let mut next_state = state;
    let step = next_state.apply(child_action);

    let v = match step {
        StepResult::Win => {
            // child_action completed a win for the current att.
            // Value for the child (opposite perspective): 0.0
            // Value for current node (who played winning move): 1.0
            let cn = arena.get_mut(child_idx);
            cn.visits += 1;
            cn.total_val += 0.0; // from child's view, the mover won = 1.0 for mover
            // Actually: after apply(), state is swapped. Child's att is the loser.
            // We store from child's perspective: child just became the losing side = 0.0.
            cn.flags |= 0b001;
            1.0 // caller's perspective: won
        }
        StepResult::Draw => {
            let cn = arena.get_mut(child_idx);
            cn.visits += 1;
            cn.total_val += 0.5;
            cn.flags |= 0b001;
            0.5
        }
        StepResult::Continue => {
            // Recurse.  next_state has sides swapped, so the returned value
            // is from the child's att perspective.  We flip for the parent.
            let child_v = search(arena, child_idx, next_state, rng, cfg);
            1.0 - child_v
        }
    };

    // --- BACKPROP into current node ---
    let node = arena.get_mut(node_idx);
    node.visits += 1;
    node.total_val += v;

    update_envelope(arena, node_idx, child_offset, cfg);

    v
}
