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
    fn new(att: HalfBoard, action_mask: u64) -> Self {
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

pub fn tss_expand_alpha((att, def): UBoard) -> (bool, Vec<(Action, UBoard)>) {
    let mut board_vec = Vec::new();
    let def_reach_mask = get_reach_mask(def, att);
    if def_reach_mask.count_ones() > 1 {
        return (false, vec![]);
    }
    let action_mask = if def_reach_mask != 0 {
        def_reach_mask & (get_2row_mask(att, def) | get_put_reach_mask(att, def))
    } else {
        get_2row_mask(att, def) | get_put_reach_mask(att, def)
    };
    for (act, n_att) in MaskActionIterator::new(att, action_mask) {
        // println!("->{}", act.trailing_zeros() % 16);
        // pprint_uboard((n_att, def));
        let def_reach_mask = get_reach_mask(def, n_att);
        // 次の手番で相手が４を作れる時
        if def_reach_mask != 0 {
            // println!("次の手番で相手が４を作れちゃう");
            // pprint_u64(def_reach_mask);
            continue;
        }

        let reach_mask = get_reach_mask(n_att, def);
        if reach_mask.count_ones() > 1 {
            return (true, vec![(act, (n_att, def))]);
        }
        // reach_mask.count_ones() == 1
        let n_def = (!reach_mask + 1) & reach_mask | def;
        let n_reach_mask = get_reach_mask(n_att, n_def);

        if n_reach_mask != 0 {
            return (true, vec![(act, (n_att, n_def))]);
        }

        board_vec.push((act, (n_att, n_def)));
    }

    return (false, board_vec);
}

pub fn tss_expand((att, def): UBoard) -> (bool, Vec<(Action, UBoard)>) {
    let mut board_vec = Vec::new();
    let action_mask = get_2row_mask(att, def) | get_put_reach_mask(att, def);
    for (act, n_att) in MaskActionIterator::new(att, action_mask) {
        // println!("->{}", act.trailing_zeros() % 16);
        // pprint_uboard((n_att, def));
        let def_reach_mask = get_reach_mask(def, n_att);
        // 次の手番で相手が４を作れる時
        if def_reach_mask != 0 {
            // println!("次の手番で相手が４を作れちゃう");
            // pprint_u64(def_reach_mask);
            continue;
        }

        let reach_mask = get_reach_mask(n_att, def);
        if reach_mask.count_ones() > 1 {
            return (true, vec![(act, (n_att, def))]);
        }
        // reach_mask.count_ones() == 1
        let n_def = (!reach_mask + 1) & reach_mask | def;
        let n_reach_mask = get_reach_mask(n_att, n_def);

        if n_reach_mask != 0 {
            return (true, vec![(act, (n_att, n_def))]);
        }

        let mut reach_mask = get_reach_mask(n_def, n_att);
        if reach_mask != 0 {
            // println!("flag1");
            // pprint_u64(reach_mask);
            let (mut att, mut def) = (n_att, n_def);
            loop {
                if reach_mask.count_ones() > 1 {
                    break;
                }
                let n_att = (!reach_mask).wrapping_add(1) & reach_mask | att;
                // println!("warikomi->");
                // pprint_uboard((n_att, def));
                let att_reach_mask = get_reach_mask(n_att, def);
                if att_reach_mask == 0 {
                    break;
                }
                if att_reach_mask.count_ones() > 1 {
                    // println!("flag2");
                    return (true, vec![(act, (n_att, def))]);
                }
                let n_def = (!att_reach_mask + 1) & att_reach_mask | def;
                // println!("warikomi2->");
                // pprint_uboard((n_att, n_def));

                let n_reach_mask = get_reach_mask(n_att, n_def);
                if n_reach_mask != 0 {
                    // println!("flag2");
                    return (true, vec![(act, (n_att, n_def))]);
                }
                reach_mask = get_reach_mask(n_def, n_att);
                if reach_mask == 0 {
                    board_vec.push((act, (n_att, n_def)));
                }
                (att, def) = (n_att, n_def);
            }
        } else {
            // println!("push back!");
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
/// Horizontal Mate Search
pub fn threat_space_search_alpha((att, def): UBoard) -> Option<(u64, Status)> {
    use std::collections::VecDeque;
    // どちらの手番にもリーチは存在しないことを仮定する
    if get_reach_mask(def, att) != 0 {
        let mask = get_reach_mask(def, att);
        return None;
        // return Some(((!mask + 1) & mask, Status::from(0, 0, 1, att, def)));
    }
    if get_reach_mask(att, def) != 0 {
        let mask = get_reach_mask(att, def);
        return Some(((!mask + 1) & mask, Status::from(0, 0, 1, att, def)));
    }
    assert!(get_reach_mask(def, att) == 0);
    assert!(get_reach_mask(att, def) == 0);

    let mut hash = HashSet::new();
    let mut expands: VecDeque<(u64, UBoard)>;
    let (end_flag, root_expands) = tss_expand((att, def));
    if end_flag {
        let (n_att, n_def) = root_expands[0].1;
        return Some((
            root_expands[0].0,
            Status::from(0, 0, count_diff(n_att, n_def, att, def), n_att, n_def),
        ));
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

        // println!("expand: att:{}, def:{}", tar_board.0, tar_board.1);
        // pprint_uboard(tar_board);

        let (end_flag, new_nodes) = tss_expand(tar_board);

        count_reach_board += get_reach_boards(tar_board.0, tar_board.1).len();
        count_valid_board += get_valid_boards(tar_board.0, tar_board.1).len();
        if end_flag {
            let (a, (n_att, n_def)) = new_nodes[0];
            return Some((
                act,
                Status::from(
                    count_valid_board,
                    count_reach_board,
                    count_diff(n_att, n_def, att, def),
                    n_att,
                    n_def,
                ),
            ));
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

/// Horizontal Mate Search
pub fn threat_space_search((att, def): UBoard) -> Option<u64> {
    use std::collections::VecDeque;
    // どちらの手番にもリーチは存在しないことを仮定する
    if get_reach_mask(def, att) != 0 {
        let mask = get_reach_mask(def, att);
        return None;
    }
    if get_reach_mask(att, def) != 0 {
        let mask = get_reach_mask(att, def);
        return Some((!mask + 1) & mask);
    }
    assert!(get_reach_mask(def, att) == 0);
    assert!(get_reach_mask(att, def) == 0);

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

        // println!("expand: att:{}, def:{}", tar_board.0, tar_board.1);
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

pub type Pn = f32;
pub type Dn = f32;

pub fn proof_number_search_att(
    (att, def): UBoard,
    th_pn: Pn,
    th_dn: Dn,
    matetype: MateType,
    hashmap: &mut HashMap<UBoard, (Option<Vec<UBoard>>, Pn, Dn, MateType)>,
) {
    // ノードを展開していない場合
    let no_child = {
        let node = hashmap.get(&(att, def)).unwrap();
        // println!(
        //     "[pns_att] th_pn:{}, th_dn:{}, pn:{}, dn:{}, size:{}, att:{att}, def:{def}",
        //     th_pn,
        //     th_dn,
        //     node.1,
        //     node.2,
        //     hashmap.len()
        // );
        node.0.is_none()
    };
    if no_child {
        let mut valid_boards = Vec::new();
        let action_mask = get_valid_action_mask(att, def);
        for (action, n_att) in MaskActionIterator::new(att, action_mask) {
            match hashmap.entry((n_att, def)) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    if threat_space_search((def, n_att)).is_some() {
                        continue;
                    }
                    if threat_space_search((n_att, def)).is_none() {
                        continue;
                    }
                    valid_boards.push((n_att, def));
                    entry.insert((None, 1.0, 1.0, MateType::NoMate));
                }
                _ => valid_boards.push((n_att, def)),
            };
        }
        hashmap.insert((att, def), (Some(valid_boards), 1.0, 1.0, MateType::NoMate));
    }

    loop {
        // cal pn, dn
        // println!("[att]cal pn, dn");
        // pprint_uboard((att, def));
        let mut pn = f32::INFINITY;
        let mut next_boards = None;
        let mut next_dn = 0.0;
        let mut dn = 0.0;
        let mut th_pn_next = th_pn;
        {
            let children = hashmap.get(&(att, def)).unwrap().0.clone().unwrap();
            for &(n_att, def) in children.iter() {
                let &(_, ch_pn, ch_dn, _) = hashmap.get(&(n_att, def)).unwrap();
                // println!(
                //     "action:{}, ch_pn:{ch_pn}, ch_dn:{ch_dn}",
                //     (n_att ^ att).trailing_zeros() % 16
                // );
                if ch_pn < pn {
                    if th_pn_next > pn {
                        th_pn_next = pn;
                    }
                    pn = ch_pn;
                    next_boards = Some((n_att, def));
                    next_dn = ch_dn;
                } else if ch_pn == pn {
                    if th_pn_next > pn {
                        th_pn_next = pn;
                    }
                }
                dn += ch_dn;
            }

            hashmap.insert(
                (att, def),
                (Some(children.clone()), pn, dn, MateType::NoMate),
            );
        }

        // println!("pn:{pn}/{th_pn}, dn:{dn}/{th_dn}, next_pn:{th_pn_next}");

        if pn == f32::INFINITY || pn == 0.0 || pn > th_pn {
            // println!("flag1");
            return;
        }
        if dn == f32::INFINITY || dn == 0.0 || dn > th_dn {
            // println!("flag2");
            return;
        }

        proof_number_search_def(
            next_boards.unwrap(),
            th_pn_next,
            th_dn - dn + next_dn,
            MateType::NoMate,
            hashmap,
        );
    }
}

pub fn proof_number_search_def(
    (att, def): UBoard,
    th_pn: Pn,
    th_dn: Dn,
    matetype: MateType,
    hashmap: &mut HashMap<UBoard, (Option<Vec<UBoard>>, Pn, Dn, MateType)>,
) {
    // ノードを展開していない場合
    let no_child = {
        let node = hashmap.get(&(att, def)).unwrap();
        // println!(
        //     "[pns_def] th_pn:{}, th_dn:{}, pn:{}, dn:{}, size:{}, att:{att}, def:{def}",
        //     th_pn,
        //     th_dn,
        //     node.1,
        //     node.2,
        //     hashmap.len()
        // );
        node.0.is_none()
    };
    if no_child {
        let mut valid_boards = Vec::new();
        let action_mask = get_valid_action_mask(att, def);
        for (action, n_def) in MaskActionIterator::new(def, action_mask) {
            match hashmap.entry((att, n_def)) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    if threat_space_search((att, n_def)).is_some() {
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
        // println!("[def] cal pn, dn");
        // pprint_uboard((att, def));
        // cal pn, dn
        let mut pn = 0.0;
        let mut next_boards = None;
        let mut next_pn = 0.0;
        let mut dn = f32::INFINITY;
        let mut th_dn_next = th_dn;
        {
            let children = hashmap.get(&(att, def)).unwrap().0.clone().unwrap();
            for &(att, n_def) in children.iter() {
                let &(_, ch_pn, ch_dn, _) = hashmap.get(&(att, n_def)).unwrap();
                // println!(
                //     "action:{}, ch_pn:{ch_pn}, ch_dn:{ch_dn}",
                //     (n_def ^ def).trailing_zeros() % 16
                // );
                if ch_dn < dn {
                    if th_dn_next > dn {
                        th_dn_next = dn;
                    }
                    dn = ch_dn;
                    next_boards = Some((att, n_def));
                    next_pn = ch_pn;
                } else if ch_dn == dn {
                    if th_dn_next > dn {
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

        // println!("[def] pn:{pn}/{th_pn}, dn:{dn}/{th_dn}, next_dn:{th_dn_next}");

        if pn == f32::INFINITY || pn == 0.0 || pn > th_pn {
            return;
        }
        if dn == f32::INFINITY || dn == 0.0 || dn > th_dn {
            return;
        }

        proof_number_search_att(
            next_boards.unwrap(),
            th_pn - pn + next_pn,
            th_dn_next,
            MateType::NoMate,
            hashmap,
        );
    }
}

#[derive(Clone, Debug)]
pub struct ProofNumberSearchStatus {
    pub size: usize,
    pub typ: MateType,
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
        &mut hashmap,
    );
    let action_mask = get_valid_action_mask(att, def);
    for (action, n_att) in MaskActionIterator::new(att, action_mask) {
        if let Some((_, pn, _, _)) = hashmap.get(&(n_att, def)) {
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
