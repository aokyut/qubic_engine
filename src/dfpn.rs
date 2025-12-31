use crate::board::{
    self, get_2row_mask, get_put_reach_mask, get_reach_mask, pprint_board, pprint_u64, Board,
};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

pub type UBoard = (u64, u64);
pub type Action = u64;
pub type HalfBoard = u64;

pub struct ZobristHash {}

impl ZobristHash {}

pub struct NodeNumber {
    pn: f32,
    dn: f32,
}

impl NodeNumber {
    fn new() -> Self {
        return NodeNumber { pn: 1.0, dn: 1.0 };
    }
    fn from(pn: f32, dn: f32) -> Self {
        return NodeNumber { pn, dn };
    }
}

#[derive(Clone, Debug, Copy)]
pub enum MateType {
    NoMate,
    Three(u64),
    Two(u64),
}

#[derive(Clone, Copy)]
pub enum Player {
    Attack,
    Defence,
}

impl Player {
    fn next(&self) -> Self {
        match self {
            Player::Attack => Player::Defence,
            _ => Player::Attack,
        }
    }
}

#[derive(Clone, Debug)]
struct MaskActionIterator {
    mask: u64,
    att: HalfBoard,
}

impl MaskActionIterator {
    fn new(att: HalfBoard, action_mask: Action) -> Self {
        return MaskActionIterator {
            mask: action_mask,
            att: att,
        };
    }
}

impl Iterator for MaskActionIterator {
    type Item = (Action, HalfBoard);

    fn next(&mut self) -> Option<Self::Item> {
        if self.mask == 0 {
            return None;
        }
        // 最下位ビット（LSB）を取り出す: x & -x
        let action = self.mask & self.mask.wrapping_neg();
        // 取り出したビットを消す
        self.mask ^= action;
        // board = att | action
        let board = self.att | action;
        Some((action, board))
    }
}

const TSS_MOVE_MASK: [u64; 64] = [
    0x9009950d953f953f,
    0x200a2206222f222f,
    0x40054406444f444f,
    0x90099a0b9acf9acf,
    0x109001d011f111f1,
    0xa0a00660a6f3a6f3,
    0x5050066056fc56fc,
    0x809008b088f888f8,
    0x09010d101f111f11,
    0x0a0a06603f6a3f6a,
    0x05050660cf65cf65,
    0x09080b808f888f88,
    0x9009d059f359f359,
    0xa0026022f222f222,
    0x50046044f444f444,
    0x9009b0a9fca9fca9,
    0x0001953f953f953f,
    0x000b222f222f222f,
    0x000d444f444f444f,
    0x00089acf9acf9acf,
    0x101111f111f111f1,
    0xa0b3a6f3a6f3a6f3,
    0x50dc56fc56fc56fc,
    0x808888f888f888f8,
    0x11011f111f111f11,
    0x3b0a3f6a3f6a3f6a,
    0xcd05cf65cf65cf65,
    0x88088f888f888f88,
    0x1000f359f359f359,
    0xb000f222f222f222,
    0xd000f444f444f444,
    0x8000fca9fca9fca9,
    0x953f953f953f9009,
    0x222f222f222f200e,
    0x444f444f444f4007,
    0x9acf9acf9acf9009,
    0x11f111f111f11190,
    0xa6f3a6f3a6f3a6e0,
    0x56fc56fc56fc5670,
    0x88f888f888f88890,
    0x1f111f111f110911,
    0x3f6a3f6a3f6a0e6a,
    0xcf65cf65cf650765,
    0x8f888f888f880988,
    0xf359f359f3599009,
    0xf222f222f222e002,
    0xf444f444f4447004,
    0xfca9fca9fca99009,
    0x953f953f0537950d,
    0x222f222f02222202,
    0x444f444f04444404,
    0x9acf9acf0ace9a0b,
    0x11f111f1007000d0,
    0xa6f3a6f300200020,
    0x56fc56fc00400040,
    0x88f888f800e000b0,
    0x1f111f1107000d00,
    0x3f6a3f6a02000200,
    0xcf65cf6504000400,
    0x8f888f880e000b00,
    0xf359f3597350d059,
    0xf222f22222202022,
    0xf444f44444404044,
    0xfca9fca9eca0b0a9,
];

pub fn tss_expand_alpha((att, def): UBoard, mask: u64) -> (bool, Vec<((Action, Action), UBoard)>) {
    let mut board_vec = Vec::new();
    let action_mask = mask & (get_2row_mask(att, def) | get_put_reach_mask(att, def));

    for (act, n_att) in MaskActionIterator::new(att, action_mask) {
        let def_reach_mask = get_reach_mask(def, n_att);
        if def_reach_mask != 0 {
            continue;
        }

        let reach_mask_ = get_reach_mask(n_att, def);
        if reach_mask_.count_ones() > 1 {
            return (true, vec![((act, act), (n_att, def))]);
        }
        // reach_mask.count_ones() = 1
        assert_eq!(reach_mask_.count_ones(), 1);
        let n_def = reach_mask_ | def;
        let n_reach_mask = get_reach_mask(n_att, n_def);

        if n_reach_mask != 0 {
            return (true, vec![((act, act), (n_att, n_def))]);
        }

        let mut reach_mask = get_reach_mask(n_def, n_att);
        if reach_mask != 0 {
            let (mut att, mut def) = (n_att, n_def);
            loop {
                if reach_mask.count_ones() > 1 {
                    break;
                }

                let n_att = reach_mask | att;
                let att_reach_mask = get_reach_mask(n_att, def);
                if att_reach_mask == 0 {
                    break;
                }
                if att_reach_mask.count_ones() > 1 {
                    return (true, vec![((act, act), (n_att, def))]);
                }
                let n_def = att_reach_mask | def;

                let n_reach_mask = get_reach_mask(n_att, n_def);
                if n_reach_mask != 0 {
                    return (true, vec![((act, act), (n_att, n_def))]);
                }
                let reach_mask_ = get_reach_mask(n_def, n_att);
                if reach_mask_ == 0 {
                    board_vec.push(((reach_mask, att_reach_mask), (n_att, n_def)));
                    break;
                }
                reach_mask = reach_mask_;
                (att, def) = (n_att, n_def);
            }
        } else {
            board_vec.push(((act, reach_mask_), (n_att, n_def)))
        }
    }

    return (false, board_vec);
}

pub fn tss_expand((att, def): UBoard) -> (bool, Vec<(Action, UBoard)>) {
    let mut board_vec = Vec::new();
    let action_mask = get_2row_mask(att, def) | get_put_reach_mask(att, def);
    for (act, n_att) in MaskActionIterator::new(att, action_mask) {
        let def_reach_mask = get_reach_mask(def, n_att);
        if def_reach_mask != 0 {
            continue;
        }

        let reach_mask = get_reach_mask(n_att, def);
        if reach_mask.count_ones() > 1 {
            return (true, vec![(act, (n_att, def))]);
        }
        let n_def = (!reach_mask + 1) & reach_mask | def;
        let n_reach_mask = get_reach_mask(n_att, n_def);

        if n_reach_mask != 0 {
            return (true, vec![(act, (n_att, n_def))]);
        }

        let mut reach_mask = get_reach_mask(n_def, n_att);
        if reach_mask != 0 {
            let (mut att, mut def) = (n_att, n_def);
            loop {
                if reach_mask.count_ones() > 1 {
                    break;
                }
                let n_att = (!reach_mask).wrapping_add(1) & reach_mask | att;
                let att_reach_mask = get_reach_mask(n_att, def);
                if att_reach_mask == 0 {
                    break;
                }
                let def_reach_mask = get_reach_mask(def, n_att);
                if def_reach_mask != 0 {
                    break;
                }
                if att_reach_mask.count_ones() > 1 {
                    return (true, vec![(act, (n_att, def))]);
                }
                let n_def = (!att_reach_mask + 1) & att_reach_mask | def;

                let n_reach_mask = get_reach_mask(n_att, n_def);
                if n_reach_mask != 0 {
                    return (true, vec![(act, (n_att, n_def))]);
                }
                reach_mask = get_reach_mask(n_def, n_att);
                if reach_mask == 0 {
                    board_vec.push((act, (n_att, n_def)));
                }
                (att, def) = (n_att, n_def);
            }
        } else {
            board_vec.push((act, (n_att, n_def)))
        }
    }

    return (false, board_vec);
}

pub fn pprint_uboard((att, def): UBoard) {
    pprint_board(&Board::from(att, def, board::Player::Black));
}

#[derive(Debug, Clone)]
pub struct Status {
    pub valid_nodes: usize,
    pub reach_boards: usize,
    pub path_size: usize,
    pub att: u64,
    pub def: u64,
}

impl Status {
    pub fn from(vnode: usize, reach_boards: usize, path_size: usize, att: u64, def: u64) -> Self {
        return Status {
            valid_nodes: vnode,
            reach_boards: reach_boards,
            path_size,
            att,
            def,
        };
    }
}

fn count_diff(n_att: u64, n_def: u64, att: u64, def: u64) -> usize {
    return ((n_att | n_def).count_ones() - (att | def).count_ones()) as usize;
}

/// Semi Horizontal Mate Search
pub fn threat_space_search((att, def): UBoard) -> Option<u64> {
    use std::collections::VecDeque;
    // 相手の手番にのみリーチがある時はfalse
    if get_reach_mask(att, def) != 0 {
        let mask = get_reach_mask(att, def);
        return Some((!mask + 1) & mask);
    }
    let root_def_reach = get_reach_mask(def, att);
    let mut def_reach = root_def_reach;
    let (mut att, mut def) = (att, def);
    while def_reach != 0 {
        if def_reach.count_ones() > 1 {
            return None;
        }
        let n_att = def_reach | att;
        let att_reach = get_reach_mask(n_att, def);
        if att_reach == 0 {
            return None;
        }
        let n_def_reach = get_reach_mask(def, n_att);
        if n_def_reach != 0 {
            return None;
        }
        if att_reach.count_ones() > 1 {
            return Some(root_def_reach);
        }
        let n_def = (!att_reach + 1) & att_reach | def;
        let n_reach_mask = get_reach_mask(n_att, n_def);
        if n_reach_mask != 0 {
            return Some(root_def_reach);
        }
        def_reach = get_reach_mask(n_def, n_att);
        (att, def) = (n_att, n_def);
        if def_reach == 0 {
            break;
        }
    }

    let mut hash = HashSet::new();
    let mut expands: VecDeque<(u64, UBoard)>;
    let (end_flag, root_expands) = tss_expand((att, def));
    if end_flag {
        let (n_att, n_def) = root_expands[0].1;
        return Some(root_expands[0].0);
    }

    expands = root_expands.into_iter().collect();

    let mut count_reach_board = 0;
    let mut count_valid_board = 0;

    loop {
        if expands.len() == 0 {
            break;
        }
        let (act, tar_board) = expands.pop_front().unwrap();
        if hash.get(&tar_board).is_some() {
            continue;
        }

        // println!(
        //     "expand: att:{}, def:{}, [{}]",
        //     tar_board.0,
        //     tar_board.1,
        //     (tar_board.0 | tar_board.1).count_ones()
        // );
        // pprint_uboard(tar_board);

        let (end_flag, new_nodes) = tss_expand(tar_board);

        count_reach_board += get_valid_boards(tar_board.0, tar_board.1).len();
        count_valid_board += get_valid_boards(tar_board.0, tar_board.1).len();
        if end_flag {
            let (a, (n_att, n_def)) = new_nodes[0];
            return Some(act);
        }
        hash.insert(tar_board);
        for (_, new_board) in new_nodes {
            // println!("->");
            // pprint_uboard(new_board);
            expands.push_back((act, new_board));
        }
    }

    return None;
}

/// Horizontal TSS mate search
/// 最短手数であることを保証する
pub fn threat_space_search_horizontal((att, def): UBoard) -> Option<Vec<Action>> {
    use std::collections::VecDeque;
    let mask = get_reach_mask(att, def);
    if mask != 0 {
        return Some(vec![(!mask + 1) & mask]);
    }

    let mut hash: HashMap<HalfBoard, UBoard> = HashMap::new();
    let mut expands: VecDeque<(Action, Action, UBoard)> = VecDeque::new();
    let mask = get_reach_mask(def, att);
    let reach_action =
        (get_2row_mask(att, def) | get_put_reach_mask(att, def)) & !get_put_reach_mask(def, att);
    let action_mask = if mask == 0 {
        reach_action
    } else {
        reach_action & mask
    };

    for (action, n_att) in MaskActionIterator::new(att, action_mask) {
        let reach_mask = get_reach_mask(n_att, def);
        // reach_mask.count_ones() == 1
        let def_action = ((!reach_mask + 1) & reach_mask);
        let n_def = def ^ ((!reach_mask + 1) & reach_mask);
        let next_reach_mask = get_reach_mask(n_att, n_def);
        if next_reach_mask != 0 {
            return Some(vec![
                action,
                def_action,
                (!next_reach_mask + 1) & next_reach_mask,
            ]);
        }
        let ndef_reach_mask = get_reach_mask(n_def, n_att);
        if ndef_reach_mask.count_ones() > 1 {
            continue;
        }
        expands.push_back((ndef_reach_mask, action, (n_att, n_def)));
        hash.insert(n_att, (att, def));
    }

    while let Some((def_reach_mask, action, (att, def))) = expands.pop_front() {
        let reach_action = (get_2row_mask(att, def) | get_put_reach_mask(att, def))
            & !get_put_reach_mask(def, att);
        let action_mask = if def_reach_mask == 0 {
            reach_action
        } else {
            reach_action & def_reach_mask
        };

        // println!(
        //     "expand: att:{}, def:{}, [{}]",
        //     att,
        //     def,
        //     (att | def).count_ones()
        // );
        // pprint_uboard((att, def));
        // pprint_u64(get_2row_mask(att, def));
        // println!("");
        // pprint_u64(get_put_reach_mask(att, def));
        // println!("");
        // pprint_u64(get_put_reach_mask(def, att));

        for (_, n_att) in MaskActionIterator::new(att, action_mask) {
            if hash.get(&n_att).is_some() {
                continue;
            }
            let reach_mask = get_reach_mask(n_att, def);
            let n_def = def ^ ((!reach_mask + 1) & reach_mask);
            let next_reach = get_reach_mask(n_att, n_def);
            if next_reach != 0 {
                hash.insert(n_att, (att, def));
                return Some(restoration_tssh_path(hash, att));
            }
            let n_def_reach = get_reach_mask(n_def, n_att);
            if n_def_reach.count_ones() > 1 {
                continue;
            }
            expands.push_back(((n_def_reach, action, (n_att, n_def))));
            // println!("->");
            // pprint_uboard((n_att, n_def));
            hash.insert(n_att, (att, def));
        }
    }

    return None;
}

#[inline]
fn restoration_tssh_path(hashmap: HashMap<HalfBoard, UBoard>, last_att: HalfBoard) -> Vec<Action> {
    let (mut att, mut def) = hashmap.get(&last_att).unwrap();
    let mut actions = vec![att ^ last_att];
    while let Some((prev_att, prev_def)) = hashmap.get(&att) {
        actions.push(prev_def ^ def);
        actions.push(prev_att ^ att);
        (att, def) = (*prev_att, *prev_def);
    }
    return actions;
}

pub fn get_reach_boards(att: u64, def: u64) -> Vec<(Action, HalfBoard)> {
    let stone = att | def;
    let mut action_mask = get_put_reach_mask(att, def);
    action_mask |= get_2row_mask(att, def);
    let mut v = Vec::new();
    loop {
        if action_mask == 0 {
            return v;
        }
        let action = (!action_mask + 1) & action_mask;
        action_mask ^= action;
        v.push((action, att | action));
    }
}

pub fn get_valid_boards(att: u64, def: u64) -> Vec<(Action, HalfBoard)> {
    let stone = att | def;
    let mut action_mask = (!stone) & ((stone << 16) | 0xffff);
    let mut v = Vec::new();
    loop {
        if action_mask == 0 {
            return v;
        }
        let action = (!action_mask + 1) & action_mask;
        action_mask ^= action;
        v.push((action, att | action));
    }
}

pub fn valid_boards(att: u64, def: u64) -> Vec<(Action, HalfBoard)> {
    let stone = att | def;
    let mut action_mask = (!stone) & ((stone << 16) | 0xffff);
    let mut v = Vec::new();
    loop {
        if action_mask == 0 {
            return v;
        }
        let action = (!action_mask + 1) & action_mask;
        action_mask = action_mask ^ action;
        v.push((action, att | action));
    }
}

pub fn get_valid_action_mask(att: u64, def: u64) -> u64 {
    let stone = att | def;
    return (!stone) & ((stone << 16) | 0xffff);
}

#[inline]
pub fn get_action_from_mask(action_mask: u64) -> u64 {
    (!action_mask + 1) & action_mask
}

pub fn fix_pns((att, def): UBoard, is_att: bool, depth: usize) -> (f32, f32) {
    if depth == 0 {
        return (1.0, 1.0);
    }
    let action_mask = get_valid_action_mask(att, def);
    if is_att {
        let mut pn = f32::INFINITY;
        let mut dn = 0.0;
        for (action, n_att) in MaskActionIterator::new(att, action_mask) {
            if threat_space_search((def, n_att)).is_some() {
                continue;
            }
            if threat_space_search((n_att, def)).is_none() {
                continue;
            }
            let (ch_pn, ch_dn) = fix_pns((n_att, def), !is_att, depth - 1);
            // println!(
            //     "[att:{depth}] action:{}, {ch_pn}/{ch_dn}",
            //     action.trailing_ones() % 16,
            // );
            if ch_pn == 0.0 {
                return (ch_pn, ch_dn);
            }
            if pn > ch_pn {
                pn = ch_pn;
            }
            dn += ch_dn;
        }
        return (pn, dn);
    } else {
        let mut pn = 0.0;
        let mut dn = f32::INFINITY;
        for (action, n_def) in MaskActionIterator::new(def, action_mask) {
            if threat_space_search((att, n_def)).is_some() {
                continue;
            }
            let (ch_pn, ch_dn) = fix_pns((att, n_def), !is_att, depth - 1);
            // println!(
            //     "[def:{depth}] action:{}, {ch_pn}/{ch_dn}",
            //     action.trailing_ones() % 16,
            // );
            if ch_dn == 0.0 {
                return (ch_pn, ch_dn);
            }
            if dn > ch_dn {
                dn = ch_dn;
            }
            pn += ch_pn;
        }
        return (pn, dn);
    }
}

pub type Pn = f32;
pub type Dn = f32;

pub fn proof_number_search_att(
    (att, def): UBoard,
    th_pn: Pn,
    th_dn: Dn,
    matetype: MateType,
    depth: usize,
    hashmap: &mut HashMap<UBoard, (Option<Vec<UBoard>>, Pn, Dn, MateType)>,
    max_node: usize,
) {
    // ノードを展開していない場合
    let no_child = {
        let node = hashmap.get(&(att, def)).unwrap();

        if cfg!(feature = "view") {
            println!(
                "[pns_att] th_pn:{}, th_dn:{}, pn:{}, dn:{}, size:{}, att:{att}, def:{def}",
                th_pn,
                th_dn,
                node.1,
                node.2,
                hashmap.len()
            );
        }
        node.0.is_none()
    };
    if no_child {
        let mut valid_boards = Vec::new();
        let reach_mask = get_reach_mask(def, att);
        let action_mask = if reach_mask != 0 {
            reach_mask
        } else {
            get_valid_action_mask(att, def)
        };
        for (action, n_att) in MaskActionIterator::new(att, action_mask) {
            match hashmap.entry((n_att, def)) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    if threat_space_search_horizontal((def, n_att)).is_some() {
                        continue;
                    }
                    let mate_thread = threat_space_search_horizontal((n_att, def));
                    if mate_thread.is_none() {
                        continue;
                    }
                    let mate_thread_size = mate_thread.unwrap().len() as f32;

                    valid_boards.push((n_att, def));
                    let (pn, dn) = if depth == 0 {
                        fix_pns((n_att, def), false, 2)
                    } else {
                        (mate_thread_size / 16.0, 1.0)
                    };
                    // println!("call fix_pns: {}/{}", pn, dn);
                    entry.insert((None, pn, dn, MateType::NoMate));
                }
                _ => valid_boards.push((n_att, def)),
            };
        }
        hashmap.insert((att, def), (Some(valid_boards), 1.0, 1.0, MateType::NoMate));
    }

    loop {
        // cal pn, dn
        if max_node < hashmap.len() {
            return;
        }
        if cfg!(feature = "view") {
            println!("[att:{depth}]cal pn, dn");
            pprint_uboard((att, def));
        }

        let mut pn = f32::INFINITY;
        let mut th_pn_next = f32::INFINITY;
        let mut next_boards = None;
        let mut next_dn = 0.0;
        let mut dn = 0.0;
        {
            let children = hashmap.get(&(att, def)).unwrap().0.clone().unwrap();
            for &(n_att, def) in children.iter() {
                let &(_, ch_pn, ch_dn, _) = hashmap.get(&(n_att, def)).unwrap();
                if cfg!(feature = "view") {
                    println!(
                        "action:{}, ch_pn:{ch_pn}, ch_dn:{ch_dn}",
                        (n_att ^ att).trailing_zeros() % 16
                    );
                }
                if th_pn_next > ch_pn {
                    if pn > ch_pn {
                        (pn, th_pn_next) = (ch_pn, pn);
                        next_boards = Some((n_att, def));
                        next_dn = ch_dn;
                    } else {
                        th_pn_next = ch_pn;
                    }
                }
                dn += ch_dn;
            }

            hashmap.insert(
                (att, def),
                (Some(children.clone()), pn, dn, MateType::NoMate),
            );
        }

        if cfg!(feature = "view") {
            println!("pn:{pn}/{th_pn}, dn:{dn}/{th_dn}, next_pn:{th_pn_next}")
        };

        if pn == f32::INFINITY || pn == 0.0 || pn > th_pn {
            return;
        }
        if dn == f32::INFINITY || dn == 0.0 || dn > th_dn {
            return;
        }
        if th_pn_next > th_pn {
            th_pn_next = th_pn;
        }

        proof_number_search_def(
            next_boards.unwrap(),
            th_pn_next,
            th_dn - dn + next_dn,
            MateType::NoMate,
            depth + 1,
            hashmap,
            max_node,
        );
    }
}

pub fn proof_number_search_def(
    (att, def): UBoard,
    th_pn: Pn,
    th_dn: Dn,
    matetype: MateType,
    depth: usize,
    hashmap: &mut HashMap<UBoard, (Option<Vec<UBoard>>, Pn, Dn, MateType)>,
    max_node: usize,
) {
    // ノードを展開していない場合
    let no_child = {
        let node = hashmap.get(&(att, def)).unwrap();
        if cfg!(feature = "view") {
            println!(
                "[pns_def] th_pn:{}, th_dn:{}, pn:{}, dn:{}, size:{}, att:{att}, def:{def}",
                th_pn,
                th_dn,
                node.1,
                node.2,
                hashmap.len()
            );
        }
        node.0.is_none()
    };
    if no_child {
        let mut valid_boards = Vec::new();
        let reach_mask = get_reach_mask(att, def);
        let action_mask = if reach_mask != 0 {
            reach_mask
        } else {
            get_valid_action_mask(att, def)
        };
        for (action, n_def) in MaskActionIterator::new(def, action_mask) {
            match hashmap.entry((att, n_def)) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    if threat_space_search_horizontal((att, n_def)).is_some() {
                        continue;
                    }
                    valid_boards.push((att, n_def));
                    entry.insert((None, 1.0, 1.0, MateType::NoMate));
                }
                _ => valid_boards.push((att, n_def)),
            };
        }
        hashmap.insert((att, def), (Some(valid_boards), 1.0, 1.0, MateType::NoMate));
    }

    loop {
        if max_node < hashmap.len() {
            return;
        }
        if cfg!(feature = "view") {
            println!("[def:{depth}] cal pn, dn");
            pprint_uboard((att, def));
        }
        // cal pn, dn
        let mut pn = 0.0;
        let mut next_boards = None;
        let mut next_pn = 0.0;
        let mut dn = f32::INFINITY;
        let mut th_dn_next = f32::INFINITY;
        {
            let children = hashmap.get(&(att, def)).unwrap().0.clone().unwrap();
            for &(att, n_def) in children.iter() {
                let &(_, ch_pn, ch_dn, _) = hashmap.get(&(att, n_def)).unwrap();
                if cfg!(feature = "view") {
                    println!(
                        "action:{}, ch_pn:{ch_pn}, ch_dn:{ch_dn}",
                        (n_def ^ def).trailing_zeros() % 16
                    );
                }
                if th_dn_next > ch_dn {
                    if dn > ch_dn {
                        (dn, th_dn_next) = (ch_dn, dn);
                        next_boards = Some((att, n_def));
                        next_pn = ch_pn;
                    } else {
                        th_dn_next = dn;
                    }
                }
                pn += ch_pn;
            }

            hashmap.insert(
                (att, def),
                (Some(children.clone()), pn, dn, MateType::NoMate),
            );
        }

        if cfg!(feature = "view") {
            println!("[def] pn:{pn}/{th_pn}, dn:{dn}/{th_dn}, next_dn:{th_dn_next}")
        };
        if pn == f32::INFINITY || pn == 0.0 || pn > th_pn {
            return;
        }
        if dn == f32::INFINITY || dn == 0.0 || dn > th_dn {
            return;
        }

        if th_dn < th_dn_next {
            th_dn_next = th_dn;
        }
        proof_number_search_att(
            next_boards.unwrap(),
            th_pn - pn + next_pn,
            th_dn_next,
            MateType::NoMate,
            depth + 1,
            hashmap,
            max_node,
        );
    }
}

#[derive(Clone, Debug)]
pub struct ProofNumberSearchStatus {
    pub size: usize,
    pub typ: MateType,
}

pub type PnsHashMap = HashMap<UBoard, (Option<Vec<UBoard>>, f32, f32, MateType)>;

pub fn child_len(hmap: &PnsHashMap, b: &UBoard) -> usize {
    let (att, def) = *b;
    let mut count = 0;
    for (c_att, c_def) in hmap.keys() {
        if att & c_att == att && def & c_def == def {
            count += 1;
        }
    }
    return count;
}

pub fn proof_number_search(b: Board) -> ProofNumberSearchStatus {
    let (att, def) = b.get_att_def();
    let res = threat_space_search((att, def));
    if let Some(action) = res {
        // println!("this has threat mate");
        return ProofNumberSearchStatus {
            size: 0,
            typ: MateType::Three((action.trailing_zeros() % 16) as u64),
        };
    }

    let mut hashmap = HashMap::new();

    hashmap.insert((att, def), (None, 1.0, 1.0, MateType::NoMate));

    proof_number_search_att(
        (att, def),
        f32::INFINITY,
        f32::INFINITY,
        MateType::NoMate,
        0,
        &mut hashmap,
        50_000,
    );
    let action_mask = get_valid_action_mask(att, def);
    for (action, n_att) in MaskActionIterator::new(att, action_mask) {
        if let Some((_, pn, _, _)) = hashmap.get(&(n_att, def)) {
            if cfg!(feature = "view") {
                println!(
                    "action:{}, size:{}",
                    action.trailing_zeros() % 16,
                    child_len(&hashmap, &(n_att, def))
                );
            }
            if *pn == 0.0 {
                return ProofNumberSearchStatus {
                    size: hashmap.len(),
                    typ: MateType::Two((action.trailing_zeros() % 16) as u64),
                };
            }
        }
    }
    return ProofNumberSearchStatus {
        size: hashmap.len(),
        typ: MateType::NoMate,
    };
}
