use super::arena::{Arena, Node};

pub struct MctsConfig {
    pub explore_c:        f32,
    pub expand_threshold: u32,
}

const ENVELOPE_THRESHOLD: u32 = 64;

#[inline]
pub fn ucb_of(child: &Node, parent_visits: u32, c: f32) -> f32 {
    if child.visits == 0 {
        return f32::INFINITY;
    }
    let q = 1.0 - child.total_val / child.visits as f32;
    let n = child.visits as f32;
    q + c * ((parent_visits as f32 + 1.0).ln() / n).sqrt()
}

/// UCB as a linear function of x = sqrt(ln(N)):  UCB(x) = a*x + b
#[inline]
fn compute_ab(child: &Node, c: f32) -> (f32, f32) {
    if child.visits == 0 {
        return (f32::INFINITY, 0.0);
    }
    let n = child.visits as f32;
    let q_prime = 1.0 - child.total_val / n;
    let a = c / n.sqrt();
    (a, q_prime)
}

pub fn select_child(node: &Node, node_idx: u32, arena: &Arena, cfg: &MctsConfig) -> u8 {
    if node.is_envelope() {
        select_envelope(node, node_idx, arena, cfg)
    } else {
        select_full(node, arena, cfg)
    }
}

fn select_full(node: &Node, arena: &Arena, cfg: &MctsConfig) -> u8 {
    let n = node.visits;
    let mut best_ucb = f32::NEG_INFINITY;
    let mut best_offset = 0u8;
    let mut i = 0u8;
    while i < node.num_children {
        let child = arena.get(node.first_child + i as u32);
        let ucb = ucb_of(child, n, cfg.explore_c);
        if ucb > best_ucb {
            best_ucb = ucb;
            best_offset = i;
        }
        i += 1;
    }
    best_offset
}

fn select_envelope(node: &Node, node_idx: u32, arena: &Arena, cfg: &MctsConfig) -> u8 {
    let n = node.visits;
    let x = ((n as f32 + 1.0).ln()).sqrt();

    let best = arena.get(node.first_child + node.best_child as u32);
    let best_ucb   = ucb_of(best, n, cfg.explore_c);
    let second_ucb = node.second_a * x + node.second_b;

    if best_ucb >= second_ucb {
        node.best_child
    } else {
        select_full(node, arena, cfg)
    }
}

pub fn update_envelope(arena: &mut Arena, node_idx: u32, visited_offset: u8, cfg: &MctsConfig) {
    let node = arena.get(node_idx);

    if !node.is_envelope() {
        if node.visits >= ENVELOPE_THRESHOLD {
            initialize_envelope(arena, node_idx, cfg);
        }
        return;
    }

    let n = node.visits;
    let x = ((n as f32 + 1.0).ln()).sqrt();
    let (best_offset, second_offset, third_offset) = {
        let nd = arena.get(node_idx);
        (nd.best_child, nd.second_idx, nd.third_idx)
    };

    let needs_rebuild = if visited_offset == best_offset {
        let best   = arena.get(arena.get(node_idx).first_child + best_offset as u32);
        let best_u = ucb_of(best, n, cfg.explore_c);
        let sec_u  = arena.get(node_idx).second_a * x + arena.get(node_idx).second_b;
        best_u < sec_u
    } else if visited_offset == second_offset {
        let fc = arena.get(node_idx).first_child;
        let second = arena.get(fc + second_offset as u32);
        let (a, b) = compute_ab(second, cfg.explore_c);
        let sec_u = a * x + b;
        let best  = arena.get(fc + best_offset as u32);
        let best_u = ucb_of(best, n, cfg.explore_c);
        if sec_u > best_u {
            true
        } else {
            arena.get_mut(node_idx).second_a = a;
            arena.get_mut(node_idx).second_b = b;
            false
        }
    } else if visited_offset == third_offset {
        let fc = arena.get(node_idx).first_child;
        let third = arena.get(fc + third_offset as u32);
        let (a, b) = compute_ab(third, cfg.explore_c);
        let thi_u = a * x + b;
        let sec_u = arena.get(node_idx).second_a * x + arena.get(node_idx).second_b;
        if thi_u > sec_u {
            true
        } else {
            arena.get_mut(node_idx).third_a = a;
            arena.get_mut(node_idx).third_b = b;
            false
        }
    } else {
        n & 0x3FF == 0 // periodic full rebuild
    };

    if needs_rebuild || n & 0x3FF == 0 {
        reinitialize_envelope(arena, node_idx, cfg);
    }
}

fn initialize_envelope(arena: &mut Arena, node_idx: u32, cfg: &MctsConfig) {
    reinitialize_envelope(arena, node_idx, cfg);
    arena.get_mut(node_idx).flags |= 0b010;
}

fn reinitialize_envelope(arena: &mut Arena, node_idx: u32, cfg: &MctsConfig) {
    let n  = arena.get(node_idx).visits;
    let nc = arena.get(node_idx).num_children as u32;
    let fc = arena.get(node_idx).first_child;

    // Sort top-3 by UCB (selection sort over nc, typically ≤16).
    let mut best   = (f32::NEG_INFINITY, 0u8);
    let mut second = (f32::NEG_INFINITY, 0u8);
    let mut third  = (f32::NEG_INFINITY, 0u8);

    let mut i = 0u32;
    while i < nc {
        let child = arena.get(fc + i);
        let u = ucb_of(child, n, cfg.explore_c);
        if u > best.0 {
            third = second;
            second = best;
            best = (u, i as u8);
        } else if u > second.0 {
            third = second;
            second = (u, i as u8);
        } else if u > third.0 {
            third = (u, i as u8);
        }
        i += 1;
    }

    let second_offset = if second.0 == f32::NEG_INFINITY { best.1 } else { second.1 };
    let third_offset  = if third.0  == f32::NEG_INFINITY { second_offset } else { third.1 };

    let second_ab = compute_ab(arena.get(fc + second_offset as u32), cfg.explore_c);
    let third_ab  = compute_ab(arena.get(fc + third_offset  as u32), cfg.explore_c);

    let node = arena.get_mut(node_idx);
    node.best_child  = best.1;
    node.second_idx  = second_offset;
    node.third_idx   = third_offset;
    node.second_a    = second_ab.0;
    node.second_b    = second_ab.1;
    node.third_a     = third_ab.0;
    node.third_b     = third_ab.1;
}
