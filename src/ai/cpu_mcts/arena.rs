/// Node in the MCTS tree.  40 bytes.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Node {
    pub visits:      u32,  // total visits
    pub total_val:   f32,  // sum of rollout values (att-side perspective)
    pub first_child: u32,  // index of first child in arena
    pub num_children: u8,
    pub action:      u8,   // cell index that led to this node
    /// bit0: terminal, bit1: envelope_active, bit2: expanded
    pub flags:       u8,
    pub best_child:  u8,   // offset within children when envelope active
    pub second_idx:  u8,
    pub third_idx:   u8,
    pub _pad:        [u8; 2],
    pub second_a:    f32,
    pub second_b:    f32,
    pub third_a:     f32,
    pub third_b:     f32,
}

impl Node {
    #[inline] pub fn is_terminal(&self) -> bool  { self.flags & 0b001 != 0 }
    #[inline] pub fn is_envelope(&self) -> bool   { self.flags & 0b010 != 0 }
    #[inline] pub fn is_expanded(&self) -> bool   { self.flags & 0b100 != 0 }
    #[inline] pub fn q(&self) -> f32 {
        if self.visits == 0 { 0.5 } else { self.total_val / self.visits as f32 }
    }
}

/// Flat node pool.
pub struct Arena {
    pub nodes: Vec<Node>,
}

impl Arena {
    pub fn new(capacity: usize) -> Self {
        let mut nodes = Vec::with_capacity(capacity);
        nodes.push(Node::default()); // root placeholder at index 0
        Self { nodes }
    }

    #[inline]
    pub fn alloc(&mut self, node: Node) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(node);
        idx
    }

    #[inline]
    pub fn get(&self, idx: u32) -> &Node {
        unsafe { self.nodes.get_unchecked(idx as usize) }
    }

    #[inline]
    pub fn get_mut(&mut self, idx: u32) -> &mut Node {
        unsafe { self.nodes.get_unchecked_mut(idx as usize) }
    }

    pub fn reset(&mut self) {
        self.nodes.clear();
        self.nodes.push(Node::default());
    }
}

/// Expand `node_idx` children using `valid_mask`.
pub fn expand(arena: &mut Arena, node_idx: u32, valid: u64) {
    if arena.get(node_idx).is_expanded() { return; }

    let first_child = arena.nodes.len() as u32;
    let mut mask = valid;
    let mut count = 0u8;
    while mask != 0 {
        let cell = mask.trailing_zeros() as u8;
        arena.alloc(Node { action: cell, ..Default::default() });
        mask &= mask - 1;
        count += 1;
    }

    let node = arena.get_mut(node_idx);
    node.first_child = first_child;
    node.num_children = count;
    node.flags |= 0b100;
}
