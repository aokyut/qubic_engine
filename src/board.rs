// use std::collections::VecDeque;
use indicatif::{ProgressBar, ProgressStyle};
use proconio::input;
use rand::Rng;
use std::fmt;
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
};

#[derive(PartialEq, Eq, Hash, Debug)]
pub enum Player {
    White,
    Black,
}

impl Clone for Player {
    fn clone(&self) -> Self {
        match self {
            Player::Black => Player::Black,
            Player::White => Player::White,
        }
    }
}

impl Player {
    pub fn next(&self) -> Self {
        match self {
            Player::Black => Player::White,
            Player::White => Player::Black,
        }
    }

    pub fn from_u64(board: u64) -> Self {
        if board % 2 == 0 {
            return Player::Black;
        } else {
            return Player::White;
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            Player::Black => String::from("Black"),
            Player::White => String::from("White"),
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct Board {
    pub black: u64,
    pub white: u64,
    pub player: Player,
}

impl Board {
    pub fn new() -> Self {
        return Board {
            black: 0,
            white: 0,
            player: Player::Black,
        };
    }

    pub fn from(black: u64, white: u64, player: Player) -> Self {
        return Board {
            black: black,
            white: white,
            player: player,
        };
    }

    pub fn next(&self, action_id: u8) -> Self {
        if cfg!(feature = "render") {
            assert!(action_id < 16);
        }
        let board = self.black | self.white;
        let action_bitboard: u64 =
            (0x0001000100010001u64 << action_id) & ((!board << 16) ^ (!board));
        match self.player {
            Player::Black => Board {
                black: self.black | action_bitboard,
                white: self.white,
                player: Player::White,
            },
            Player::White => Board {
                black: self.black,
                white: self.white | action_bitboard,
                player: Player::Black,
            },
        }
    }

    pub fn is_draw(&self) -> bool {
        return (self.black | self.white) == 0xffffffffffffffff;
    }

    pub fn is_win(&self) -> bool {
        match self.player {
            Player::White => _is_win_board(self.black),
            Player::Black => _is_win_board(self.white),
        }
    }

    pub fn clone(&self) -> Self {
        return Board {
            black: self.black,
            white: self.white,
            player: self.player.clone(),
        };
    }

    pub fn action_mask(&self) -> u64 {
        let board = self.black | self.white;
        return board >> 48;
    }

    pub fn valid_actions(&self) -> Vec<u8> {
        let mut actions = Vec::<u8>::new();
        let board = (self.black | self.white) >> 48;
        for i in 0..16u8 {
            if (board >> i) & 1 == 0 {
                actions.push(i);
            }
        }

        return actions;
    }

    pub fn has_mate(&self, depth: u8) -> (bool, u8) {
        if depth == 1 {
            for action in self.valid_actions() {
                let next_board = self.next(action);
                if next_board.is_win() {
                    return (true, action);
                }
            }
        } else {
            for action in self.valid_actions() {
                let next_board = self.next(action);
                if next_board.is_win() {
                    return (true, action);
                } else {
                    let val = -next_board._minimax_action(depth - 1);
                    if val == 1 {
                        return (true, action);
                    }
                }
            }
        }
        return (false, 0);
    }

    pub fn minimax_action(&self, depth: u8) -> u8 {
        let mut rng = rand::thread_rng();
        if depth == 1 {
            let actions = self.valid_actions();
            for action in actions.iter() {
                let next_board = self.next(*action);
                if next_board.is_win() {
                    return *action;
                } else if next_board.is_draw() {
                    return *action;
                }
            }
            return actions[rng.gen::<usize>() % actions.len()];
        } else {
            let mut actions: Vec<u8> = Vec::new();
            let mut max_val: i8 = -2;
            for action in self.valid_actions() {
                let next_board = self.next(action);
                if next_board.is_win() {
                    return action;
                } else if next_board.is_draw() {
                    return action;
                } else {
                    let val = -next_board._minimax_action(depth - 1);
                    if val == 1 {
                        return action;
                    }
                    if max_val < val {
                        max_val = val;
                        actions = vec![action];
                    } else if max_val == val {
                        actions.push(action);
                    }
                }
            }

            // println!("actions:{:#?}, val:{}", actions, max_val);
            return actions[rng.gen::<usize>() % actions.len()];
        }
    }

    pub fn to_string(&self) -> String {
        return format!("{},{}", self.black, self.white);
    }

    pub fn to_board_string(&self) -> String {
        let mut s = String::new();
        for i in 0..64 {
            if (self.black >> i) & 1 == 1 {
                s += "O";
            } else if (self.white >> i) & 1 == 1 {
                s += "X";
            } else {
                s += "-";
            }
        }
        match self.player {
            Player::Black => s += "B",
            Player::White => s += "W",
        }
        return s;
    }

    fn _minimax_action(&self, depth: u8) -> i8 {
        if depth == 0 {
            return 0;
        }
        let mut max_val = -2;
        for action in self.valid_actions() {
            let next_board = self.next(action);
            if next_board.is_win() {
                return 1;
            } else if next_board.is_draw() {
                return 0;
            }
            let val = -next_board._minimax_action(depth - 1);
            if max_val < val {
                max_val = val;
            }
        }
        return max_val;
    }

    pub fn is_black(&self) -> bool {
        match self.player {
            Player::Black => true,
            Player::White => false,
        }
    }

    pub fn get_att_def(&self) -> (u64, u64) {
        match self.player {
            Player::Black => (self.black, self.white),
            Player::White => (self.white, self.black),
        }
    }

    pub fn to_u128(&self) -> u128 {
        match self.player {
            Player::Black => (self.black as u128) + ((self.white as u128) << 64),
            Player::White => (self.white as u128) + ((self.black as u128) << 64),
        }
    }

    pub fn hash(&self) -> u128 {
        let mut bitboard = self.to_u128();
        let mut min_bitboard = bitboard;

        let hbitboard = Board::hflip(bitboard);
        if min_bitboard > hbitboard {
            min_bitboard = hbitboard;
        }

        for _ in 0..3 {
            bitboard = Board::rot(bitboard);
            if min_bitboard > bitboard {
                min_bitboard = bitboard;
            }
            let hbitboard = Board::hflip(bitboard);
            if min_bitboard > hbitboard {
                min_bitboard = hbitboard;
            }
        }

        return min_bitboard;
    }

    pub fn hflip(bitboard: u128) -> u128 {
        let mask: u128 = 0x11111111111111111111111111111111;
        return ((bitboard >> 3) & mask)
            | ((bitboard >> 1) & mask << 1)
            | ((bitboard << 1) & mask << 2)
            | ((bitboard << 3) & mask << 3);
    }

    pub fn dflip(bitboard: u128) -> u128 {
        let mask1: u128 = 0x0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a;
        let mask1_: u128 = 0xa5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5;
        let mask2: u128 = 0x00cc00cc00cc00cc00cc00cc00cc00cc;
        let mask2_: u128 = 0xcc33cc33cc33cc33cc33cc33cc33cc33;
        let bitboard = (bitboard & mask1_) | ((bitboard >> 3) & mask1) | ((bitboard & mask1) << 3);
        return (bitboard & mask2_) | ((bitboard >> 6) & mask2) | ((bitboard & mask2) << 6);
    }

    pub fn rot(bitboard: u128) -> u128 {
        return Board::dflip(Board::hflip(bitboard));
    }
}

pub fn _is_win_board(bit: u64) -> bool {
    (bit & (bit >> 1) & (bit >> 2) & (bit >> 3) & 0x1111111111111111)
        | (bit & (bit >> 4) & (bit >> 8) & (bit >> 12) & 0x000f000f000f000f)
        | (bit & (bit >> 16) & (bit >> 32) & (bit >> 48) & 0x000000000000ffff)
        | (bit & (bit >> 5) & (bit >> 10) & (bit >> 15) & 0x0001000100010001)
        | (bit & (bit >> 3) & (bit >> 6) & (bit >> 9) & 0x0008000800080008)
        | (bit & (bit >> 17) & (bit >> 34) & (bit >> 51) & 0x1111)
        | (bit & (bit >> 15) & (bit >> 30) & (bit >> 45) & 0x8888)
        | (bit & (bit >> 20) & (bit >> 40) & (bit >> 60) & 0x000f)
        | (bit & (bit >> 12) & (bit >> 24) & (bit >> 36) & 0xf000)
        | (bit & (bit >> 21) & (bit >> 42) & (bit >> 63))
        | (bit & (bit >> 19) & (bit >> 38) & (bit >> 57) & 0x0008)
        | (bit & (bit >> 13) & (bit >> 26) & (bit >> 39) & 0x1000)
        | (bit & (bit >> 11) & (bit >> 22) & (bit >> 33) & 0x8000)
        > 0
}

pub const fn count_2row_(
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    b1: u64,
    b2: u64,
    b3: u64,
    b4: u64,
) -> u64 {
    return a1 & a2 & b3 & b4
        | a1 & b2 & a3 & b4
        | a1 & b2 & b3 & a4
        | b1 & a2 & a3 & b4
        | b1 & a2 & b3 & a4
        | b1 & b2 & a3 & a4;
}

pub fn count_2row(s: u64, b: u64) -> u32 {
    let x = (count_2row_(s, s >> 1, s >> 2, s >> 3, b, b >> 1, b >> 2, b >> 3)
        & 0x1111_1111_1111_1111)
        .count_ones();
    let y = (count_2row_(s, s >> 4, s >> 8, s >> 12, b, b >> 4, b >> 8, b >> 12)
        & 0x000f_000f_000f_000f)
        .count_ones();
    let z = count_2row_(s, s >> 16, s >> 32, s >> 48, b, b >> 16, b >> 32, b >> 48).count_ones();
    let xy = (count_2row_(s, s >> 5, s >> 10, s >> 15, b, b >> 5, b >> 10, b >> 15)
        & 0x0001_0001_0001_0001)
        .count_ones();
    let xy_ = (count_2row_(s, s >> 3, s >> 6, s >> 9, b, b >> 3, b >> 6, b >> 9)
        & 0x0008_0008_0008_0008)
        .count_ones();
    let xz = (count_2row_(s, s >> 17, s >> 34, s >> 51, b, b >> 17, b >> 34, b >> 51)
        & 0x0000_0000_0000_1111)
        .count_ones();
    let xz_ = (count_2row_(s, s >> 15, s >> 30, s >> 45, b, b >> 15, b >> 30, b >> 45) & 0x8888)
        .count_ones();
    let yz = (count_2row_(s, s >> 20, s >> 40, s >> 60, b, b >> 20, b >> 40, b >> 60)
        & 0x0000_0000_0000_000f)
        .count_ones();
    let yz_ = (count_2row_(s, s >> 12, s >> 24, s >> 36, b, b >> 12, b >> 24, b >> 36) & 0xf000)
        .count_ones();
    let xyz1 = count_2row_(s, s >> 21, s >> 42, s >> 63, b, b >> 21, b >> 42, b >> 63);
    let xyz2 = count_2row_(s, s >> 19, s >> 38, s >> 57, b, b >> 19, b >> 38, b >> 57) & 0x0008;
    let xyz3 = count_2row_(s, s >> 13, s >> 26, s >> 39, b, b >> 13, b >> 26, b >> 39) & 0x1000;
    let xyz4 = count_2row_(s, s >> 11, s >> 22, s >> 33, b, b >> 11, b >> 22, b >> 33) & 0x8000;

    let xyz = (xyz1 | xyz2 | xyz3 | xyz4).count_ones();
    return x + y + z + xy + xy_ + xz + xz_ + yz + yz_ + xyz;
}

fn count_1row_(a1: u64, a2: u64, a3: u64, a4: u64, b1: u64, b2: u64, b3: u64, b4: u64) -> u64 {
    return a1 & b2 & b3 & b4 | b1 & a2 & b3 & b4 | b1 & b2 & a3 & b4 | b1 & b2 & b3 & a4;
}

/// * a: right 3 shift attacker
/// * b: right 2 shift attacker
/// * c: right 1 shift attacker
/// * d: blank mask
/// * e: left 1 shift attacker
/// * f: left 2 shift attacker
/// * g: left 3 shift attacker
/// * a, b, c and e, f, g are masked.
const fn _get_reach_mask(a: u64, b: u64, c: u64, d: u64, e: u64, f: u64, g: u64) -> u64 {
    return d & (b & c & (a | e) | e & f & (c | g));
}

pub fn count_1row(s: u64, b: u64) -> u32 {
    let x = (count_1row_(s, s >> 1, s >> 2, s >> 3, b, b >> 1, b >> 2, b >> 3)
        & 0x1111_1111_1111_1111)
        .count_ones();
    let y = (count_1row_(s, s >> 4, s >> 8, s >> 12, b, b >> 4, b >> 8, b >> 12)
        & 0x000f_000f_000f_000f)
        .count_ones();
    let z = count_1row_(s, s >> 16, s >> 32, s >> 48, b, b >> 16, b >> 32, b >> 48).count_ones();
    let xy = (count_1row_(s, s >> 5, s >> 10, s >> 15, b, b >> 5, b >> 10, b >> 15)
        & 0x0001_0001_0001_0001)
        .count_ones();
    let xy_ = (count_1row_(s, s >> 3, s >> 6, s >> 9, b, b >> 3, b >> 6, b >> 9)
        & 0x0008_0008_0008_0008)
        .count_ones();
    let xz = (count_1row_(s, s >> 17, s >> 34, s >> 51, b, b >> 17, b >> 34, b >> 51)
        & 0x0000_0000_0000_1111)
        .count_ones();
    let xz_ = (count_1row_(s, s >> 15, s >> 30, s >> 45, b, b >> 15, b >> 30, b >> 45) & 0x8888)
        .count_ones();
    let yz = (count_1row_(s, s >> 20, s >> 40, s >> 60, b, b >> 20, b >> 40, b >> 60)
        & 0x0000_0000_0000_000f)
        .count_ones();
    let yz_ = (count_1row_(s, s >> 12, s >> 24, s >> 36, b, b >> 12, b >> 24, b >> 36) & 0xf000)
        .count_ones();
    let xyz1 = count_1row_(s, s >> 21, s >> 42, s >> 63, b, b >> 21, b >> 42, b >> 63);
    let xyz2 = count_1row_(s, s >> 19, s >> 38, s >> 57, b, b >> 19, b >> 38, b >> 57) & 0x0008;
    let xyz3 = count_1row_(s, s >> 13, s >> 26, s >> 39, b, b >> 13, b >> 26, b >> 39) & 0x1000;
    let xyz4 = count_1row_(s, s >> 11, s >> 22, s >> 33, b, b >> 11, b >> 22, b >> 33) & 0x8000;

    let xyz = (xyz1 | xyz2 | xyz3 | xyz4).count_ones();
    return x + y + z + xy + xy_ + xz + xz_ + yz + yz_ + xyz;
}

pub fn count_3row(s: u64, b: u64) -> u32 {
    let (s, b) = (b, s);

    let x = (count_1row_(s, s >> 1, s >> 2, s >> 3, b, b >> 1, b >> 2, b >> 3)
        & 0x1111_1111_1111_1111)
        .count_ones();
    let y = (count_1row_(s, s >> 4, s >> 8, s >> 12, b, b >> 4, b >> 8, b >> 12)
        & 0x000f_000f_000f_000f)
        .count_ones();
    let z = count_1row_(s, s >> 16, s >> 32, s >> 48, b, b >> 16, b >> 32, b >> 48).count_ones();
    let xy = (count_1row_(s, s >> 5, s >> 10, s >> 15, b, b >> 5, b >> 10, b >> 15)
        & 0x0001_0001_0001_0001)
        .count_ones();
    let xy_ = (count_1row_(s, s >> 3, s >> 6, s >> 9, b, b >> 3, b >> 6, b >> 9)
        & 0x0008_0008_0008_0008)
        .count_ones();
    let xz = (count_1row_(s, s >> 17, s >> 34, s >> 51, b, b >> 17, b >> 34, b >> 51)
        & 0x0000_0000_0000_1111)
        .count_ones();
    let xz_ = (count_1row_(s, s >> 15, s >> 30, s >> 45, b, b >> 15, b >> 30, b >> 45) & 0x8888)
        .count_ones();
    let yz = (count_1row_(s, s >> 20, s >> 40, s >> 60, b, b >> 20, b >> 40, b >> 60)
        & 0x0000_0000_0000_000f)
        .count_ones();
    let yz_ = (count_1row_(s, s >> 12, s >> 24, s >> 36, b, b >> 12, b >> 24, b >> 36) & 0xf000)
        .count_ones();
    let xyz1 = count_1row_(s, s >> 21, s >> 42, s >> 63, b, b >> 21, b >> 42, b >> 63);
    let xyz2 = count_1row_(s, s >> 19, s >> 38, s >> 57, b, b >> 19, b >> 38, b >> 57) & 0x0008;
    let xyz3 = count_1row_(s, s >> 13, s >> 26, s >> 39, b, b >> 13, b >> 26, b >> 39) & 0x1000;
    let xyz4 = count_1row_(s, s >> 11, s >> 22, s >> 33, b, b >> 11, b >> 22, b >> 33) & 0x8000;

    let xyz = (xyz1 | xyz2 | xyz3 | xyz4).count_ones();
    return x + y + z + xy + xy_ + xz + xz_ + yz + yz_ + xyz;
}

const fn _get_1row_mask(a: u64, b: u64, c: u64, d: u64) -> u64 {
    return a & b & c & d;
}

pub const fn get_1row_mask(s: u64, b: u64) -> u64 {
    let (b1, b2, b3, b4, b6, b8, b9, b11, b12, b13, b15) = (
        b >> 1,
        b >> 2,
        b >> 3,
        b >> 4,
        b >> 6,
        b >> 8,
        b >> 9,
        b >> 11,
        b >> 12,
        b >> 13,
        b >> 15,
    );
    let (b16, b17, b19, b20, b21, b22, b24, b26, b30) = (
        b >> 16,
        b >> 17,
        b >> 19,
        b >> 20,
        b >> 21,
        b >> 22,
        b >> 24,
        b >> 26,
        b >> 30,
    );
    let (b32, b33, b34, b36, b38, b39, b40, b42, b45) = (
        b >> 32,
        b >> 33,
        b >> 34,
        b >> 36,
        b >> 38,
        b >> 39,
        b >> 40,
        b >> 42,
        b >> 45,
    );
    let (b48, b49, b57, b60, b63) = (b >> 48, b >> 49, b >> 57, b >> 60, b >> 63);
    0
}

pub fn get_reach_mask_alpha(a: u64, d: u64) -> u64 {
    let b = a;
    let stone = a | d;
    let s = !stone;
    let blank = s & ((stone << 16) | 0xffff);

    let x = (count_1row_(s, s >> 1, s >> 2, s >> 3, b, b >> 1, b >> 2, b >> 3)
        & 0x1111_1111_1111_1111)
        * 0xf
        & blank;
    let y = (count_1row_(s, s >> 4, s >> 8, s >> 12, b, b >> 4, b >> 8, b >> 12)
        & 0x000f_000f_000f_000f);
    let y = (y * 0x1111) & blank;
    let z = count_1row_(s, s >> 16, s >> 32, s >> 48, b, b >> 16, b >> 32, b >> 48);
    let z = (z * 0x0001_0001_0001_0001) & blank;
    let xy = (count_1row_(s, s >> 5, s >> 10, s >> 15, b, b >> 5, b >> 10, b >> 15)
        & 0x0001_0001_0001_0001);
    let xy = (xy * 0x8421) & blank;
    let xy_ =
        (count_1row_(s, s >> 3, s >> 6, s >> 9, b, b >> 3, b >> 6, b >> 9) & 0x0008_0008_0008_0008);
    let xy_ = (xy_ * 0x249) & blank;
    let xz = (count_1row_(s, s >> 17, s >> 34, s >> 51, b, b >> 17, b >> 34, b >> 51)
        & 0x0000_0000_0000_1111);
    let xz = (xz * 0x0008_0004_0002_0001) & blank;
    let xz_ = (count_1row_(s, s >> 15, s >> 30, s >> 45, b, b >> 15, b >> 30, b >> 45) & 0x8888);
    let xz_ = (xz_ * 0x0000_2000_4000_8001) & blank;
    let yz = (count_1row_(s, s >> 20, s >> 40, s >> 60, b, b >> 20, b >> 40, b >> 60)
        & 0x0000_0000_0000_000f);
    let yz = (yz * 0x1000_0100_0010_0001) & blank;
    let yz_ = (count_1row_(s, s >> 12, s >> 24, s >> 36, b, b >> 12, b >> 24, b >> 36) & 0xf000);
    let yz_ = (yz_ * 0x0000_0010_0100_1001) & blank;

    let xyz1 = count_1row_(s, s >> 21, s >> 42, s >> 63, b, b >> 21, b >> 42, b >> 63);
    let xyz1 = (xyz1 * 0x8000_0400_0020_0001) & blank;
    let xyz2 = count_1row_(s, s >> 19, s >> 38, s >> 57, b, b >> 19, b >> 38, b >> 57) & 0x0008;
    let xyz2 = (xyz2 * 0x0200_0040_0008_0001) & blank;
    let xyz3 = count_1row_(s, s >> 13, s >> 26, s >> 39, b, b >> 13, b >> 26, b >> 39) & 0x1000;
    let xyz3 = (xyz3 * 0x0000_0080_0400_2001) & blank;
    let xyz4 = count_1row_(s, s >> 11, s >> 22, s >> 33, b, b >> 11, b >> 22, b >> 33) & 0x8000;
    let xyz4 = (xyz4 * 0x0000_0002_0040_0801) & blank;

    // println!("flag");
    // pprint_u64(xyz4);

    return x | y | z | xy | xy_ | yz | yz_ | xz | xz_ | xyz1 | xyz2 | xyz3 | xyz4;
}

pub fn get_reach_mask(a: u64, d: u64) -> u64 {
    let stone = a | d;
    let blank = !(stone) & ((stone << 16) | 0xffff);
    let x = _get_reach_mask(
        (a >> 3) & 0x1111_1111_1111_1111,
        (a >> 2) & 0x3333_3333_3333_3333,
        (a >> 1) & 0x7777_7777_7777_7777,
        blank,
        (a << 1) & 0xeeee_eeee_eeee_eeee,
        (a << 2) & 0xcccc_cccc_cccc_cccc,
        (a << 3) & 0x8888_8888_8888_8888,
    );
    let y = _get_reach_mask(
        (a >> 12) & 0x000f_000f_000f_000f,
        (a >> 8) & 0x00ff_00ff_00ff_00ff,
        (a >> 4) & 0x0fff_0fff_0fff_0fff,
        blank,
        (a << 4) & 0xfff0_fff0_fff0_fff0,
        (a << 8) & 0xff00_ff00_ff00_ff00,
        (a << 12) & 0xf000_f000_f000_f000,
    );
    let z = _get_reach_mask(a >> 48, a >> 32, a >> 16, blank, a << 16, a << 32, a << 48);
    let xy = _get_reach_mask(
        (a >> 15) & 0x0001_0001_0001_0001,
        (a >> 10) & 0x0033_0033_0033_0033,
        (a >> 5) & 0x0777_0777_0777_0777,
        blank,
        (a << 5) & 0xeee0_eee0_eee0_eee0,
        (a << 10) & 0xcc00_cc00_cc00_cc00,
        (a << 15) & 0x8000_8000_8000_8000,
    );
    let yx = _get_reach_mask(
        (a >> 9) & 0x0008_0008_0008_0008,
        (a >> 6) & 0x00cc_00cc_00cc_00cc,
        (a >> 3) & 0x0eee_0eee_0eee_0eee,
        blank,
        (a << 3) & 0x7770_7770_7770_7770,
        (a << 6) & 0x3300_3300_3300_3300,
        (a << 9) & 0x1000_1000_1000_1000,
    );
    let xz = _get_reach_mask(
        (a >> 51) & 0x0000_0000_0000_1111,
        (a >> 34) & 0x0000_0000_3333_3333,
        (a >> 17) & 0x0000_7777_7777_7777,
        blank,
        (a << 17) & 0xeeee_eeee_eeee_0000,
        (a << 34) & 0xcccc_cccc_0000_0000,
        (a << 51) & 0x8888_0000_0000_0000,
    );
    let zx = _get_reach_mask(
        (a >> 45) & 0x0000_0000_0000_8888,
        (a >> 30) & 0x0000_0000_cccc_cccc,
        (a >> 15) & 0x0000_eeee_eeee_eeee,
        blank,
        (a << 15) & 0x7777_7777_7777_0000,
        (a << 30) & 0x3333_3333_0000_0000,
        (a << 45) & 0x1111_0000_0000_0000,
    );
    let yz = _get_reach_mask(
        (a >> 60) & 0xf,
        (a >> 40) & 0x00ff_00ff,
        (a >> 20) & 0x0fff_0fff_0fff,
        blank,
        (a << 20) & 0xfff0_fff0_fff0_0000,
        (a << 40) & 0xff00_ff00_0000_0000,
        (a << 60) & 0xf000_0000_0000_0000,
    );
    let zy = _get_reach_mask(
        (a >> 36) & 0xf000,
        (a >> 24) & 0xff00_ff00,
        (a >> 12) & 0xfff0_fff0_fff0,
        blank,
        (a << 12) & 0x0fff_0fff_0fff_0000,
        (a << 24) & 0x00ff_00ff_0000_0000,
        (a << 36) & 0x000f_0000_0000_0000,
    );
    let xyz = _get_reach_mask(
        (a >> 63) & 0x1,
        (a >> 42) & 0x0033_0033,
        (a >> 21) & 0x0777_0777_0777,
        blank,
        (a << 21) & 0xeee0_eee0_eee0_0000,
        (a << 42) & 0xcc00_cc00_0000_0000,
        (a << 63) & 0x8000_0000_0000_0000,
    );
    let yzx = _get_reach_mask(
        (a >> 57) & 0x8,
        (a >> 38) & 0x00cc_00cc,
        (a >> 19) & 0x0eee_0eee_0eee,
        blank,
        (a << 19) & 0x7770_7770_7770_0000,
        (a << 38) & 0x3300_3300_0000_0000,
        (a << 57) & 0x1000_0000_0000_0000,
    );
    let xzy = _get_reach_mask(
        (a >> 39) & 0x1000,
        (a >> 26) & 0x3300_3300,
        (a >> 13) & 0x7770_7770_7770,
        blank,
        (a << 13) & 0x0eee_0eee_0eee_0000,
        (a << 26) & 0x00cc_00cc_0000_0000,
        (a << 39) & 0x0008_0000_0000_0000,
    );
    let zyx = _get_reach_mask(
        (a >> 33) & 0x8000,
        (a >> 22) & 0xcc00_cc00,
        (a >> 11) & 0xeee0_eee0_eee0,
        blank,
        (a << 11) & 0x0777_0777_0777_0000,
        (a << 22) & 0x0033_0033_0000_0000,
        (a << 33) & 0x0001_0000_0000_0000,
    );

    return x | y | z | xy | yx | xz | zx | yz | zy | xyz | xzy | yzx | zyx;
}

pub fn search_four(board: &Board) -> Option<u8> {
    let (att, def) = board.get_att_def();
    let reach_mask = get_reach_mask(att, def);
    if reach_mask == 0 {
        return None;
    } else {
        let action = (reach_mask | (reach_mask >> 16) | (reach_mask >> 32) | (reach_mask >> 48))
            .trailing_zeros();
        return Some(action as u8);
    }
}

pub fn mate_expand(board: &Board) -> (bool, Vec<(u8, Board)>) {
    let mut board_vec = Vec::new();
    for action in board.valid_actions() {
        let def_board = board.next(action);
        let (att, def) = def_board.get_att_def();
        let def_reach_mask = get_reach_mask(att, def);
        if def_reach_mask != 0 {
            continue;
        }
        let reach_mask = get_reach_mask(def, att);
        if reach_mask == 0 {
            continue;
        }
        let action_mask = reach_mask | (reach_mask >> 16) | (reach_mask >> 32) | (reach_mask >> 48);
        let def_action = action_mask.trailing_zeros();
        // reach_maskの場所を把握して返す
        let num = reach_mask.count_ones();
        if num > 1 {
            return (true, vec![(action, board.clone())]);
        }
        let mut next_board = def_board.next(def_action as u8);
        let (att, def) = next_board.get_att_def();
        let att_reach_mask = get_reach_mask(att, def);

        if att_reach_mask != 0 {
            return (true, vec![(action, next_board)]);
        }

        let mut reach_mask = get_reach_mask(def, att);
        if reach_mask != 0 {
            loop {
                if reach_mask.count_ones() > 1 {
                    break;
                }
                let att_action =
                    (reach_mask | (reach_mask >> 16) | (reach_mask >> 32) | (reach_mask >> 48))
                        .trailing_zeros();
                let def_board = next_board.next(att_action as u8);
                let (att, def) = def_board.get_att_def();
                let att_reach_mask = get_reach_mask(def, att);
                if att_reach_mask == 0 {
                    break;
                }
                let def_action = (att_reach_mask
                    | (att_reach_mask >> 16)
                    | (att_reach_mask >> 32)
                    | (att_reach_mask >> 48))
                    .trailing_zeros();

                next_board = def_board.next(def_action as u8);
                let (att, def) = next_board.get_att_def();
                reach_mask = get_reach_mask(def, att);
                if reach_mask == 0 {
                    board_vec.push((action, next_board));
                    break;
                }
            }
        } else {
            board_vec.push((action, next_board));
        }
    }
    return (false, board_vec);
}

pub fn mate_check(board: &Board) -> Option<u8> {
    // 既に四があるかチェック
    let four = search_four(board);
    if four.is_some() {
        return four;
    }
    // 相手側の四があるかチェック
    let (att, def) = board.get_att_def();
    let reach_mask = get_reach_mask(def, att);
    if reach_mask != 0 {
        let action = (reach_mask | (reach_mask >> 16) | (reach_mask >> 32) | (reach_mask >> 48))
            .trailing_zeros();
        return Some(action as u8);
    }

    let mut hash = HashSet::new();
    // 攻撃側のハッシュだけで十分か？
    let mut expands;
    let end_flag;
    (end_flag, expands) = mate_expand(board);
    if end_flag {
        return Some(expands[0].0);
    }
    let mut count = 1;

    loop {
        if expands.len() == 0 {
            break;
        }
        let (action, tar_board) = expands.pop().unwrap();
        if hash.get(&tar_board).is_some() {
            continue;
        }

        let (end_flag, new_nodes) = mate_expand(&tar_board);
        count += 1;
        if end_flag {
            return Some(action);
        }
        hash.insert(tar_board);
        for (_, next_board) in new_nodes {
            expands.push((action, next_board));
        }
    }

    return None;
}

/// 最短手数でのmateを知りたいときはこっち
pub fn mate_check_horizontal(board: &Board) -> Option<(bool, u8)> {
    use std::collections::VecDeque;
    // 既に四があるかチェック
    let four = search_four(board);

    if let Some(action) = four {
        return Some((true, action));
    }
    // 相手側の四があるかチェック
    let (att, def) = board.get_att_def();
    let reach_mask = get_reach_mask(def, att);
    if reach_mask != 0 {
        let action = (reach_mask | (reach_mask >> 16) | (reach_mask >> 32) | (reach_mask >> 48))
            .trailing_zeros();
        return Some((false, action as u8));
    }

    let mut hash = HashSet::new();
    // 攻撃側のハッシュだけで十分か？
    let mut expands: VecDeque<(u8, Board)>;
    let (end_flag, root_expands) = mate_expand(board);
    if end_flag {
        return Some((true, root_expands[0].0));
    }

    expands = root_expands.into_iter().collect();

    let mut count = 1;

    loop {
        if expands.len() == 0 {
            break;
        }
        let (action, tar_board) = expands.pop_front().unwrap();
        if hash.get(&tar_board).is_some() {
            continue;
        }

        let (end_flag, new_nodes) = mate_expand(&tar_board);
        count += 1;
        if end_flag {
            return Some((true, action));
        }
        hash.insert(tar_board);
        for (_, next_board) in new_nodes {
            expands.push_back((action, next_board));
        }
    }

    return None;
}

pub fn pprint_u64(bit: u64) {
    let mut s = String::new();
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                let idx = j * 16 + i * 4 + k;
                if (bit >> idx) & 1 == 1 {
                    s += "O";
                } else {
                    s += "-";
                }
            }
            s += " | ";
        }
        s += "\n"
    }
    print!("{}", s);
}

pub fn get_random(board: &Board) -> u8 {
    let mut rng = rand::thread_rng();
    let actions = board.valid_actions();
    return actions[rng.gen::<usize>() % actions.len()];
}

pub fn pprint_board(board: &Board) {
    let mut s = String::new();
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                let idx = j * 16 + i * 4 + k;
                if (board.black >> idx) & 1 == 1 {
                    s += "O";
                } else if (board.white >> idx) & 1 == 1 {
                    s += "X";
                } else {
                    s += "-";
                }
            }
            s += " | ";
        }
        s += "\n"
    }
    print!("{}", s);
}

fn playout(board: &Board) -> f32 {
    let mut b = board.clone();
    let mut coef = 1.0;
    loop {
        let action = get_random(&b);
        b = b.next(action);
        if b.is_win() {
            return coef;
        } else if b.is_draw() {
            return 0.0;
        }
        coef *= -1.0;
    }
}

#[derive(serde::Serialize)]
pub struct MateRow {
    pub depth: i32,
    pub action: i32,
}

fn _search_mate(b: &Board, depth_max: u8) -> MateRow {
    for i in 0..=((depth_max - 1) / 2) {
        let depth = i * 2 + 1;
        println!("depth: {} start", depth);
        let (flag, action) = b.has_mate(depth);
        if flag {
            return MateRow {
                depth: depth as i32,
                action: action as i32,
            };
        }
        println!("depth: {} end", depth);
    }
    return MateRow {
        depth: 0,
        action: -1,
    };
}

pub struct Record {
    moves: Vec<u8>,
    mcts: Option<RefCell<Node>>,
    cursor: usize,
    initial: bool,
}

impl Record {
    pub fn new() -> Self {
        return Record {
            moves: vec![],
            cursor: 0,
            mcts: None,
            initial: true,
        };
    }

    pub fn from(moves: Vec<u8>) -> Self {
        let cursor = moves.len() - 1;
        let initial = moves.len() == 0;
        return Record {
            moves: moves,
            cursor: cursor,
            mcts: None,
            initial: initial,
        };
    }

    pub fn get_last_board(&self) -> Board {
        let mut b = Board::new();
        if self.initial || self.moves.len() == 0 {
            return b;
        }
        for i in 0..=self.cursor {
            b = b.next(self.moves[i]);
        }
        return b;
    }

    pub fn initial_board(&mut self) -> Board {
        self.initial = true;
        self.cursor = 0;
        return self.get_last_board();
    }

    pub fn jump_last_board(&mut self) -> Board {
        if self.moves.len() == 0 {
            return self.get_last_board();
        } else {
            self.initial = false;
            self.cursor = self.moves.len() - 1;
            return self.get_last_board();
        }
    }

    pub fn next(&mut self) -> Board {
        if self.initial {
            self.initial = false;
        } else {
            if self.cursor + 1 < self.moves.len() {
                self.cursor += 1;
            }
        }
        // let action = self.moves[self.cursor as usize];
        return self.get_last_board();
    }

    pub fn back(&mut self) -> Board {
        if self.cursor == 0 {
            self.initial = true;
            return self.get_last_board();
        }
        if self.cursor > 0 {
            self.cursor -= 1;
        }
        return self.get_last_board();
    }

    pub fn push(&mut self, action: u8) {
        if self.cursor + 1 < self.moves.len() {
            for _ in 0..(self.moves.len() - 1 - self.cursor) {
                let _ = self.moves.pop();
            }
        }
        // self.moves = self.moves[0..(self.cursor as usize)];
        if self.initial {
            self.moves.pop();
        }
        self.moves.push(action);
        self.cursor = self.moves.len() - 1;
        self.initial = false;
    }

    pub fn push_and_board(&mut self, action: u8) -> Board {
        self.push(action);
        return self.get_last_board();
    }

    pub fn run_mcts_evaluate(&mut self, search_n: usize) -> Vec<Score> {
        let current_board = self.get_last_board();
        match &self.mcts {
            None => {
                let mut node = Node::new(current_board);
                let result = node.search(50, search_n);
                self.mcts = Some(RefCell::new(node));
                return result;
            }
            Some(node) => {
                if node.borrow().board == current_board {
                    println!("current board is same");
                    let result = node.borrow_mut().search(50, search_n);
                    return result;
                } else {
                    let mut node = Node::new(current_board);
                    let result = node.search(50, search_n);
                    self.mcts = Some(RefCell::new(node));
                    return result;
                }
            }
        }
    }
}
pub struct Node {
    board: Board,
    n: f32,
    w: f32,
    children: HashMap<u8, RefCell<Node>>,
}

#[derive(serde::Serialize, PartialEq, PartialOrd)]
pub struct Score {
    pub action: u8,
    pub score: f32,
    pub q: f32,
    pub na: f32,
    pub n: f32,
}

impl fmt::Debug for Score {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "action: {:>2}, score: {:>5.2}%({:>7.0}/{:>7.0}), Q: {:>5.3}",
            self.action,
            self.score * 100.0,
            self.na,
            self.n,
            self.q
        );
        Ok(())
    }
}

impl Node {
    pub fn new(board: Board) -> Self {
        return Node {
            board: board,
            children: HashMap::new(),
            n: 1f32,
            w: 0f32,
        };
    }

    pub fn search(&mut self, expand_n: usize, search_n: usize) -> Vec<Score> {
        if self.children.len() == 0 {
            self.expand();
            for (action, node) in self.children.iter() {
                // println!(
                //     "[expand] action:{}, board:{:x}, player:{:#?}, player':{:#?}",
                //     action,
                //     node.borrow().board.black,
                //     node.borrow().board.player,
                //     self.board.player,
                // )
            }
        }
        for _ in 0..search_n {
            self.evaluate(expand_n);
        }

        let mut scores = Vec::new();
        for (action, node) in self.children.iter() {
            scores.push(Score {
                action: *action,
                score: node.borrow().n / self.n,
                q: node.borrow().w / node.borrow().n,
                na: node.borrow().n,
                n: self.n,
            });
            // println!("{}/{}", node.borrow().n, self.n);
        }
        return scores;
    }

    fn evaluate(&mut self, expand_n: usize) -> f32 {
        if self.board.is_win() {
            self.w += 1.0;
            self.n += 1.0;
            return 1.0;
        } else if self.board.is_draw() {
            self.n += 1.0;
            return 0.0;
        } else if self.children.len() == 0 {
            let value = -playout(&self.board);
            self.w += value;
            self.n += 1.0;
            if self.n == expand_n as f32 {
                self.expand();
            }
            return value;
        } else {
            let next_node_action = {
                let children = &self.children;
                let mut best_action = 0;
                // (best_action, best_node) = &children[&0];
                let mut max_score = -2.0;
                for (action, node) in children.iter() {
                    let ucb = node.borrow().get_uct(self.n);
                    if ucb > max_score {
                        max_score = ucb;
                        best_action = *action;
                    }
                }
                best_action
            };
            let value = -self
                .children
                .get(&next_node_action)
                .unwrap()
                .borrow_mut()
                .evaluate(expand_n);
            self.w += value;
            self.n += 1.0;
            return value;
        }
    }

    fn expand(&mut self) {
        let mut nodes = HashMap::new();
        let mut set: HashSet<u128> = HashSet::new();
        for action in self.board.valid_actions() {
            let next_board = self.board.next(action);
            assert_eq!(next_board.player, next_board.clone().player);
            // println!("{:#?}, {:#?}", self.board.player, next_board.clone().player);
            if !set.contains(&next_board.hash()) {
                nodes.insert(action, RefCell::new(Node::new(next_board.clone())));
                set.insert(next_board.hash());
            }
        }
        self.children = nodes
    }

    fn get_uct(&self, N: f32) -> f32 {
        return self.w / self.n + (2.0 * N.ln() / self.n).sqrt();
    }
}

pub fn mcts_action(board: &Board, n: usize, ex_n: usize) -> u8 {
    let mut node = Node::new(board.clone());
    let scores = node.search(ex_n, n);
    // let mut max_action = 0;
    let mut max_actions = Vec::new();
    let mut max_score = -2.0;
    for score in scores {
        if score.score > max_score {
            max_score = score.score;
            // max_action = score.action;
            max_actions = vec![score.action];
        } else if score.score == max_score {
            max_actions.push(score.action);
        }
    }
    let mut rng = rand::thread_rng();
    return max_actions[rng.gen::<usize>() % max_actions.len()];
    // return max_action;
}

pub trait GetAction {
    fn get_action(&self, b: &Board) -> u8;
}
pub trait GetActionMut {
    fn get_action_mut(&mut self, b: &Board) -> u8;
}

impl<T: GetAction> GetActionMut for T {
    fn get_action_mut(&mut self, b: &Board) -> u8 {
        return self.get_action(b);
    }
}

pub enum Agent {
    Human,
    Random,
    Minimax(u8),
    Mcts(usize, usize),
    Custom(String, Box<dyn Fn(&Board) -> u8>),
    Struct(String, Box<dyn GetAction>),
}

impl Agent {
    pub fn name(&self) -> String {
        match self {
            Agent::Human => String::from("Human"),
            Agent::Random => String::from("Random"),
            Agent::Minimax(depth) => format!("Minimax:{}", depth),
            Agent::Mcts(ex, se) => format!("Mcts:{}/{}", se, ex),
            Agent::Custom(name, func) => format!("{}", name),
            Agent::Struct(name, ga) => format!("{}", name),
        }
    }
}

impl GetAction for Agent {
    fn get_action(&self, board: &Board) -> u8 {
        use std::time::Instant;
        match self {
            Agent::Human => {
                // let (att, def) = board.get_att_def();
                // let reach = get_reach_mask(att, def);
                let start = Instant::now();
                let mate_action = mate_check(board);
                let end = start.elapsed();
                let a_milis = end.as_nanos();
                let start = Instant::now();
                let mate_action_h = mate_check_horizontal(board);
                let end = start.elapsed();
                let b_milis = end.as_nanos();
                if let Some(action) = mate_action {
                    let (is_mate, h_action) = mate_action_h.unwrap();
                    println!("mate:{is_mate}, v[{action}], h[{h_action}]");
                }
                input! {
                    action: u8
                }
                action
            }
            Agent::Minimax(depth) => board.minimax_action(*depth),
            Agent::Mcts(expand_n, search_n) => mcts_action(board, *search_n, *expand_n),
            Agent::Random => get_random(board),
            Agent::Custom(name, func) => func(board),
            Agent::Struct(name, ga) => ga.get_action(board),
        }
    }
}

pub fn play(a1: &Agent, a2: &Agent) -> (f32, f32) {
    let mut b = Board::new();
    loop {
        pprint_board(&b);
        if b.is_black() {
            let action = a1.get_action(&b);
            b = b.next(action);
            if b.is_win() {
                return (1.0, 0.0);
            } else if b.is_draw() {
                return (0.5, 0.5);
            }
        } else {
            let action = a2.get_action(&b);
            b = b.next(action);
            if b.is_win() {
                return (0.0, 1.0);
            } else if b.is_draw() {
                return (0.5, 0.5);
            }
        }
    }
}

pub fn play_actor(a1: &impl GetAction, a2: &impl GetAction, render: bool) -> (f32, f32) {
    let mut b = Board::new();
    // b = b.next(get_random(&b));
    // b = b.next(get_random(&b));
    // b = b.next(15);
    // b = b.next(12);

    loop {
        if render {
            pprint_board(&b);
        }
        if b.is_black() {
            let action = a1.get_action(&b);
            if render {
                println!("action:{action}");
            }
            b = b.next(action);
            if b.is_win() {
                return (1.0, 0.0);
            } else if b.is_draw() {
                return (0.5, 0.5);
            }
        } else {
            let action = a2.get_action(&b);
            if render {
                println!("action:{action}");
            }
            b = b.next(action);
            if b.is_win() {
                return (0.0, 1.0);
            } else if b.is_draw() {
                return (0.5, 0.5);
            }
        }
    }
}

pub fn play_and_result(a1: &Agent, a2: &Agent) -> (Board, f32, f32) {
    let mut b = Board::new();
    loop {
        pprint_board(&b);
        if b.is_black() {
            let action = a1.get_action(&b);
            b = b.next(action);
            if b.is_win() {
                return (b, 1.0, 0.0);
            } else if b.is_draw() {
                return (b, 0.5, 0.5);
            }
        } else {
            let action = a2.get_action(&b);
            b = b.next(action);
            if b.is_win() {
                return (b, 0.0, 1.0);
            } else if b.is_draw() {
                return (b, 0.5, 0.5);
            }
        }
    }
}

pub fn eval(a1: &Agent, a2: &Agent, n: usize) -> (f32, f32) {
    let mut score1 = 0.0;
    let mut score2 = 0.0;

    for i in 0..n {
        let (s1, s2) = play(a1, a2);
        score1 += s1;
        score2 += s2;
        // println!("game black: {}, s1:{}, s2:{}", i, s1, s2);
        let (s2, s1) = play(a2, a1);
        score1 += s1;
        score2 += s2;
        // println!("game white: {}, s1:{}, s2:{}", i, s1, s2);
    }
    return (score1 / (2 * n) as f32, score2 / (2 * n) as f32);
}

pub fn compare_agent(a1: &Agent, a2: &Agent, n: usize, th: f64, render: bool) -> (f32, f32, bool) {
    use super::utills::half_imcomplete_beta_func;
    let mut score1 = 0.0;
    let mut score2 = 0.0;

    let pb = ProgressBar::new((n * 2) as u64);
    pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) \n {msg}")
            .unwrap()
            .progress_chars("#>-"));

    let mut p = 0.5;
    for _ in 0..n {
        let (s1, s2) = play_actor(a1, a2, render);
        score1 += s1;
        score2 += s2;
        pb.inc(1);
        pb.set_message(format!(
            "acum:[{score1}, {score2}], p:[{p:.6}:{:.6}][{s1}, {s2}]",
            1.0 - p
        ));
        // println!("game black: {}, s1:{}, s2:{}", i, s1, s2);
        let (s2, s1) = play_actor(a2, a1, render);
        score1 += s1;
        score2 += s2;
        pb.inc(1);
        p = half_imcomplete_beta_func(score1.floor() as f64, score2.floor() as f64);
        pb.set_message(format!(
            "acum:[{score1}, {score2}], p:[{p:.6}:{:.6}][{s1}, {s2}]",
            1.0 - p
        ));

        // println!("game white: {}, s1:{}, s2:{}", i, s1, s2);

        if p < th || p > (1.0 - th) {
            pb.finish();
            return (score1, score2, true);
            break;
        }
    }
    pb.finish();
    return (score1, score2, false);
}

pub fn eval_actor(a1: &impl GetAction, a2: &impl GetAction, n: usize, render: bool) -> (f32, f32) {
    use std::{thread, time::Duration};
    let mut score1 = 0.0;
    let mut score2 = 0.0;

    let pb = ProgressBar::new((n * 2) as u64);
    pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) \n {msg}")
            .unwrap()
            .progress_chars("#>-"));

    for i in 0..n {
        // thread::sleep(Duration::from_millis(200));
        let (s1, s2) = play_actor(a1, a2, render);
        // println!("[{}/{}]black: {}, {}", i, n, s1, s2);
        score1 += s1;
        score2 += s2;
        pb.inc(1);
        pb.set_message(format!("[{score1}, {score2}]"));
        // println!("game black: {}, s1:{}, s2:{}", i, s1, s2);
        let (s2, s1) = play_actor(a2, a1, render);
        // println!("[{}/{}]white: {}, {}", i, n, s1, s2);
        score1 += s1;
        score2 += s2;
        pb.inc(1);
        pb.set_message(format!("[{score1}, {score2}]"));
        // println!("game white: {}, s1:{}, s2:{}", i, s1, s2);
    }
    return (score1 / (2 * n) as f32, score2 / (2 * n) as f32);
}
