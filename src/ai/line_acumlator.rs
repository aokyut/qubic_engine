use std::f64;
use std::hash::Hash;
use std::sync::LazyLock;
use std::cell::UnsafeCell;

use super::*;
use crate::board::{
    Action, HalfBoard, Player, UBoard, pprint_u64
};
use crate::ai::{Fail, EvaluatorF};
use crate::dfpn::pprint_uboard;
use rand::{Rng, thread_rng};

// Actionから伸びるLineの情報のみを更新する
// Actionについてはそれぞれ自分から伸びるLineのインデックスの配列を定数として保持しておく
// 0~64の立体PosActionを受け取り、（ラインインデックス、マスク）の配列を受け取る
// 相手のLineInfoに移り
// 盤面の情報は（自分のライン情報（１～，２～，３～）、相手のライン情報（１～、２～、３～））の３つ
// まず自分の１～を更新し、２～＝２～｜１～＆ラインマスク、３～＝３～｜２～＆ラインマスクとする。
// 有効なラインを調べる時には相手の１～を反転させたものが有効ラインとなる
type MonoLineInfo = [u64;7];

type HalfLineInfo = (MonoLineInfo, MonoLineInfo, MonoLineInfo);

fn count_mono_lineinfo(info: MonoLineInfo, mask: u64) -> usize{
    info.iter().map(|b| (b & mask).count_ones()).sum::<u32>() as usize
}

pub struct LSBGenerator{
    action_mask: u64,
    first: Option<(u64, usize)>,
    second: Option<(u64, usize)>,
}

pub fn action_mask2vec(action_mask: u64) -> Vec<(u64, usize)>{
    let mut action_mask = action_mask;
    let mut v = Vec::with_capacity(16);
    loop {
        if action_mask == 0{
            return v;
        }
        let action = action_mask & (!action_mask + 1);
        let idx = action.trailing_zeros() as usize;
        v.push((action, idx));
        action_mask &= action_mask - 1;
    }
}

impl LSBGenerator{
    pub fn new(action_mask: u64) -> Self{
        let first = action_mask & (!action_mask + 1);
        if first == 0{
            return Self{ action_mask:0, first:None, second:None};
        }
        let action_mask = action_mask ^ first;;
        let second = action_mask & (!action_mask + 1);
        if second == 0{
            return Self{ action_mask:0, first:Some((first, first.trailing_zeros() as usize)), second:None};
        }
        return Self { action_mask:action_mask ^ second, first:Some((first, first.trailing_zeros() as usize)), second:Some((second, second.trailing_zeros() as usize)) };
    }
    pub fn from_uboard(board: UBoard) -> Self{
        let stone = board.0 | board.1;
        let action_mask = !stone & ((stone << 16) | 0xffff);
        return Self::new(action_mask);
    }
}

impl Iterator for LSBGenerator{
    type Item = (Action, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let next_node;
        if self.action_mask == 0{
            let lsb = self.action_mask & (!self.action_mask + 1);
            let idx = lsb.trailing_zeros() as usize;
            self.action_mask &= self.action_mask - 1;
            next_node = Some((lsb, idx));
        }else{
            next_node = None;
        }
        let ans = self.first;
        self.first = self.second;
        self.second = next_node;
        return ans;
     }
}

//[x, y, z, x+y, x+z, y+z, x+y+z]
const ActionLineTable: [[u64; 7]; 64] = [
    [0x000000000000000f,0x0000000000001111,0x0001000100010001,0x0000000000008421,0x0008000400020001,0x1000010000100001,0x8000040000200001,],
    [0x000000000000000f,0x0000000000002222,0x0002000200020002,0x0000000000000000,0x0000000000000000,0x2000020000200002,0x0000000000000000,],
    [0x000000000000000f,0x0000000000004444,0x0004000400040004,0x0000000000000000,0x0000000000000000,0x4000040000400004,0x0000000000000000,],
    [0x000000000000000f,0x0000000000008888,0x0008000800080008,0x0000000000001248,0x0001000200040008,0x8000080000800008,0x1000020000400008,],
    [0x00000000000000f0,0x0000000000001111,0x0010001000100010,0x0000000000000000,0x0080004000200010,0x0000000000000000,0x0000000000000000,],
    [0x00000000000000f0,0x0000000000002222,0x0020002000200020,0x0000000000008421,0x0000000000000000,0x0000000000000000,0x0000000000000000,],
    [0x00000000000000f0,0x0000000000004444,0x0040004000400040,0x0000000000001248,0x0000000000000000,0x0000000000000000,0x0000000000000000,],
    [0x00000000000000f0,0x0000000000008888,0x0080008000800080,0x0000000000000000,0x0010002000400080,0x0000000000000000,0x0000000000000000,],
    [0x0000000000000f00,0x0000000000001111,0x0100010001000100,0x0000000000000000,0x0800040002000100,0x0000000000000000,0x0000000000000000,],
    [0x0000000000000f00,0x0000000000002222,0x0200020002000200,0x0000000000001248,0x0000000000000000,0x0000000000000000,0x0000000000000000,],
    [0x0000000000000f00,0x0000000000004444,0x0400040004000400,0x0000000000008421,0x0000000000000000,0x0000000000000000,0x0000000000000000,],
    [0x0000000000000f00,0x0000000000008888,0x0800080008000800,0x0000000000000000,0x0100020004000800,0x0000000000000000,0x0000000000000000,],
    [0x000000000000f000,0x0000000000001111,0x1000100010001000,0x0000000000001248,0x8000400020001000,0x0001001001001000,0x0008004002001000,],
    [0x000000000000f000,0x0000000000002222,0x2000200020002000,0x0000000000000000,0x0000000000000000,0x0002002002002000,0x0000000000000000,],
    [0x000000000000f000,0x0000000000004444,0x4000400040004000,0x0000000000000000,0x0000000000000000,0x0004004004004000,0x0000000000000000,],
    [0x000000000000f000,0x0000000000008888,0x8000800080008000,0x0000000000008421,0x1000200040008000,0x0008008008008000,0x0001002004008000,],
    [0x00000000000f0000,0x0000000011110000,0x0001000100010001,0x0000000084210000,0x0000000000000000,0x0000000000000000,0x0000000000000000,],
    [0x00000000000f0000,0x0000000022220000,0x0002000200020002,0x0000000000000000,0x0008000400020001,0x0000000000000000,0x0000000000000000,],
    [0x00000000000f0000,0x0000000044440000,0x0004000400040004,0x0000000000000000,0x0001000200040008,0x0000000000000000,0x0000000000000000,],
    [0x00000000000f0000,0x0000000088880000,0x0008000800080008,0x0000000012480000,0x0000000000000000,0x0000000000000000,0x0000000000000000,],
    [0x0000000000f00000,0x0000000011110000,0x0010001000100010,0x0000000000000000,0x0000000000000000,0x1000010000100001,0x0000000000000000,],
    [0x0000000000f00000,0x0000000022220000,0x0020002000200020,0x0000000084210000,0x0080004000200010,0x2000020000200002,0x8000040000200001,],
    [0x0000000000f00000,0x0000000044440000,0x0040004000400040,0x0000000012480000,0x0010002000400080,0x4000040000400004,0x1000020000400008,],
    [0x0000000000f00000,0x0000000088880000,0x0080008000800080,0x0000000000000000,0x0000000000000000,0x8000080000800008,0x0000000000000000,],
    [0x000000000f000000,0x0000000011110000,0x0100010001000100,0x0000000000000000,0x0000000000000000,0x0001001001001000,0x0000000000000000,],
    [0x000000000f000000,0x0000000022220000,0x0200020002000200,0x0000000012480000,0x0800040002000100,0x0002002002002000,0x0008004002001000,],
    [0x000000000f000000,0x0000000044440000,0x0400040004000400,0x0000000084210000,0x0100020004000800,0x0004004004004000,0x0001002004008000,],
    [0x000000000f000000,0x0000000088880000,0x0800080008000800,0x0000000000000000,0x0000000000000000,0x0008008008008000,0x0000000000000000,],
    [0x00000000f0000000,0x0000000011110000,0x1000100010001000,0x0000000012480000,0x0000000000000000,0x0000000000000000,0x0000000000000000,],
    [0x00000000f0000000,0x0000000022220000,0x2000200020002000,0x0000000000000000,0x8000400020001000,0x0000000000000000,0x0000000000000000,],
    [0x00000000f0000000,0x0000000044440000,0x4000400040004000,0x0000000000000000,0x1000200040008000,0x0000000000000000,0x0000000000000000,],
    [0x00000000f0000000,0x0000000088880000,0x8000800080008000,0x0000000084210000,0x0000000000000000,0x0000000000000000,0x0000000000000000,],
    [0x0000000f00000000,0x0000111100000000,0x0001000100010001,0x0000842100000000,0x0000000000000000,0x0000000000000000,0x0000000000000000,],
    [0x0000000f00000000,0x0000222200000000,0x0002000200020002,0x0000000000000000,0x0001000200040008,0x0000000000000000,0x0000000000000000,],
    [0x0000000f00000000,0x0000444400000000,0x0004000400040004,0x0000000000000000,0x0008000400020001,0x0000000000000000,0x0000000000000000,],
    [0x0000000f00000000,0x0000888800000000,0x0008000800080008,0x0000124800000000,0x0000000000000000,0x0000000000000000,0x0000000000000000,],
    [0x000000f000000000,0x0000111100000000,0x0010001000100010,0x0000000000000000,0x0000000000000000,0x0001001001001000,0x0000000000000000,],
    [0x000000f000000000,0x0000222200000000,0x0020002000200020,0x0000842100000000,0x0010002000400080,0x0002002002002000,0x0001002004008000,],
    [0x000000f000000000,0x0000444400000000,0x0040004000400040,0x0000124800000000,0x0080004000200010,0x0004004004004000,0x0008004002001000,],
    [0x000000f000000000,0x0000888800000000,0x0080008000800080,0x0000000000000000,0x0000000000000000,0x0008008008008000,0x0000000000000000,],
    [0x00000f0000000000,0x0000111100000000,0x0100010001000100,0x0000000000000000,0x0000000000000000,0x1000010000100001,0x0000000000000000,],
    [0x00000f0000000000,0x0000222200000000,0x0200020002000200,0x0000124800000000,0x0100020004000800,0x2000020000200002,0x1000020000400008,],
    [0x00000f0000000000,0x0000444400000000,0x0400040004000400,0x0000842100000000,0x0800040002000100,0x4000040000400004,0x8000040000200001,],
    [0x00000f0000000000,0x0000888800000000,0x0800080008000800,0x0000000000000000,0x0000000000000000,0x8000080000800008,0x0000000000000000,],
    [0x0000f00000000000,0x0000111100000000,0x1000100010001000,0x0000124800000000,0x0000000000000000,0x0000000000000000,0x0000000000000000,],
    [0x0000f00000000000,0x0000222200000000,0x2000200020002000,0x0000000000000000,0x1000200040008000,0x0000000000000000,0x0000000000000000,],
    [0x0000f00000000000,0x0000444400000000,0x4000400040004000,0x0000000000000000,0x8000400020001000,0x0000000000000000,0x0000000000000000,],
    [0x0000f00000000000,0x0000888800000000,0x8000800080008000,0x0000842100000000,0x0000000000000000,0x0000000000000000,0x0000000000000000,],
    [0x000f000000000000,0x1111000000000000,0x0001000100010001,0x8421000000000000,0x0001000200040008,0x0001001001001000,0x0001002004008000,],
    [0x000f000000000000,0x2222000000000000,0x0002000200020002,0x0000000000000000,0x0000000000000000,0x0002002002002000,0x0000000000000000,],
    [0x000f000000000000,0x4444000000000000,0x0004000400040004,0x0000000000000000,0x0000000000000000,0x0004004004004000,0x0000000000000000,],
    [0x000f000000000000,0x8888000000000000,0x0008000800080008,0x1248000000000000,0x0008000400020001,0x0008008008008000,0x0008004002001000,],
    [0x00f0000000000000,0x1111000000000000,0x0010001000100010,0x0000000000000000,0x0010002000400080,0x0000000000000000,0x0000000000000000,],
    [0x00f0000000000000,0x2222000000000000,0x0020002000200020,0x8421000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000,],
    [0x00f0000000000000,0x4444000000000000,0x0040004000400040,0x1248000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000,],
    [0x00f0000000000000,0x8888000000000000,0x0080008000800080,0x0000000000000000,0x0080004000200010,0x0000000000000000,0x0000000000000000,],
    [0x0f00000000000000,0x1111000000000000,0x0100010001000100,0x0000000000000000,0x0100020004000800,0x0000000000000000,0x0000000000000000,],
    [0x0f00000000000000,0x2222000000000000,0x0200020002000200,0x1248000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000,],
    [0x0f00000000000000,0x4444000000000000,0x0400040004000400,0x8421000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000,],
    [0x0f00000000000000,0x8888000000000000,0x0800080008000800,0x0000000000000000,0x0800040002000100,0x0000000000000000,0x0000000000000000,],
    [0xf000000000000000,0x1111000000000000,0x1000100010001000,0x1248000000000000,0x1000200040008000,0x1000010000100001,0x1000020000400008,],
    [0xf000000000000000,0x2222000000000000,0x2000200020002000,0x0000000000000000,0x0000000000000000,0x2000020000200002,0x0000000000000000,],
    [0xf000000000000000,0x4444000000000000,0x4000400040004000,0x0000000000000000,0x0000000000000000,0x4000040000400004,0x0000000000000000,],
    [0xf000000000000000,0x8888000000000000,0x8000800080008000,0x8421000000000000,0x8000400020001000,0x8000080000800008,0x8000040000200001,]
];

const REDUCTION_STRENGTH: f64 = 0.0;
const APPROX_LOG: [f64; 64] = [
    0.0,0.0,0.6931471805599453,1.0986122886681098,1.3862943611198906,1.6094379124341003,1.791759469228055,1.9459101490553132,2.0794415416798357,2.1972245773362196,2.302585092994046,2.3978952727983707,2.4849066497880004,2.5649493574615367,2.6390573296152584,2.70805020110221,2.772588722239781,2.833213344056216,2.8903717578961645,2.9444389791664403,2.995732273553991,3.044522437723423,3.091042453358316,3.1354942159291497,3.1780538303479458,3.2188758248682006,3.258096538021482,3.295836866004329,3.332204510175204,3.367295829986474,3.4011973816621555,3.4339872044851463,3.4657359027997265,3.4965075614664802,3.5263605246161616,3.5553480614894135,3.58351893845611,3.6109179126442243,3.6375861597263857,3.6635616461296463,3.6888794541139363,3.713572066704308,3.7376696182833684,3.7612001156935624,3.784189633918261,3.8066624897703196,3.828641396489095,3.8501476017100584,3.871201010907891,3.8918202981106265,3.912023005428146,3.9318256327243257,3.9512437185814275,3.970291913552122,3.9889840465642745,4.007333185232471,4.02535169073515,4.04305126783455,4.060443010546419,4.07753744390572,4.0943445622221,4.110873864173311,4.127134385045092,4.143134726391533,
];
pub const REDUCTION_TABLE: [[u8; 64]; 64] = {
    let mut table = [[0u8; 64]; 64];
    let mut depth = 1u8;
    while depth < 64 {
        let mut moves = 0u8;
        if depth <= 3{
            while moves < 64{
                table[depth as usize][moves as usize] = depth;
                moves += 1;
            }
            depth += 1;
            continue;
        }
        moves = 1u8;
        while moves < 64 {
            // LMR的な削減量の計算式
            let reduction = (APPROX_LOG[depth as usize] * APPROX_LOG[moves as usize] * REDUCTION_STRENGTH) as u8;
            table[depth as usize][moves as usize] = if reduction + 3 > depth { 3 } else { depth - reduction };
            moves += 1;
        }
        table[depth as usize][0] = depth;
        depth += 1;
    }
    table
};

#[derive(Clone, Debug)]
pub struct LineInfo{
    att: HalfLineInfo,
    def: HalfLineInfo,
}

impl LineInfo{
    pub fn from_board(b: &Board) -> Self{
        let (a, d) = b.get_att_def();
        let p = Self::ad_board_to_halfLineInfo(a, d);
        let e = Self::ad_board_to_halfLineInfo(d, a);
        return Self { att: p, def: e }
    }
    pub fn ad_board_to_halfLineInfo(a:u64, d:u64) -> HalfLineInfo{
        let mut half: HalfLineInfo = ([0; 7], [0; 7], [0; 7]);
        for i in 0..64{
            if a & (1 << i) != 0{
                for (idx, line) in ActionLineTable[i].iter().enumerate(){
                    half.2[idx] |= line & half.1[idx];
                    half.1[idx] |= line & half.0[idx];
                    half.0[idx] |= line;
                }
            }
        }

        return half;
    }

    pub fn next(&self, action: usize) -> Self{
        let action_line = ActionLineTable[action];
        let att_line_info = &self.att;
        let next_att_line_info = (
            [
                att_line_info.0[0] | action_line[0],
                att_line_info.0[1] | action_line[1],
                att_line_info.0[2] | action_line[2],
                att_line_info.0[3] | action_line[3],
                att_line_info.0[4] | action_line[4],
                att_line_info.0[5] | action_line[5],
                att_line_info.0[6] | action_line[6],
            ],
            [
                att_line_info.1[0] | (att_line_info.0[0] & action_line[0]),
                att_line_info.1[1] | (att_line_info.0[1] & action_line[1]),
                att_line_info.1[2] | (att_line_info.0[2] & action_line[2]),
                att_line_info.1[3] | (att_line_info.0[3] & action_line[3]),
                att_line_info.1[4] | (att_line_info.0[4] & action_line[4]),
                att_line_info.1[5] | (att_line_info.0[5] & action_line[5]),
                att_line_info.1[6] | (att_line_info.0[6] & action_line[6]),
            ],
            [
                att_line_info.2[0] | (att_line_info.1[0] & action_line[0]),
                att_line_info.2[1] | (att_line_info.1[1] & action_line[1]),
                att_line_info.2[2] | (att_line_info.1[2] & action_line[2]),
                att_line_info.2[3] | (att_line_info.1[3] & action_line[3]),
                att_line_info.2[4] | (att_line_info.1[4] & action_line[4]),
                att_line_info.2[5] | (att_line_info.1[5] & action_line[5]),
                att_line_info.2[6] | (att_line_info.1[6] & action_line[6]),
            ]
        );
        return Self { att: self.def.clone(), def: next_att_line_info }
    }
}

pub trait LineInfoEvaluator{
    fn evaluate_lineinfo(&self, board:&UBoard, line_info: &LineInfo) -> f32;
}

const POS_TO_LINE: [[u8; 7]; 64] = [[0, 16, 32, 48, 56, 64, 75], [0, 20, 33, 65, 76, 76, 76], [0, 24, 34, 66, 76, 76, 76], [0, 28, 35, 52, 60, 67, 74], [1, 16, 36, 57, 76, 76, 76], [1, 20, 37, 48, 76, 76, 76], [1, 24, 38, 52, 76, 76, 76], [1, 28, 39, 61, 76, 76, 76], [2, 16, 40, 58, 76, 76, 76], [2, 20, 41, 52, 76, 76, 76], [2, 24, 42, 48, 76, 76, 76], [2, 28, 43, 62, 76, 76, 76], [3, 16, 44, 52, 59, 68, 73], [3, 20, 45, 69, 76, 76, 76], [3, 24, 46, 70, 76, 76, 76], [3, 28, 47, 48, 63, 71, 72], [4, 17, 32, 49, 76, 76, 76], [4, 21, 33, 56, 76, 76, 76], [4, 25, 34, 60, 76, 76, 76], [4, 29, 35, 53, 76, 76, 76], [5, 17, 36, 64, 76, 76, 76], [5, 21, 37, 49, 57, 65, 75], [5, 25, 38, 53, 61, 66, 74], [5, 29, 39, 67, 76, 76, 76], [6, 17, 40, 68, 76, 76, 76], [6, 21, 41, 53, 58, 69, 73], [6, 25, 42, 49, 62, 70, 72], [6, 29, 43, 71, 76, 76, 76], [7, 17, 44, 53, 76, 76, 76], [7, 21, 45, 59, 76, 76, 76], [7, 25, 46, 63, 76, 76, 76], [7, 29, 47, 49, 76, 76, 76], [8, 18, 32, 50, 76, 76, 76], [8, 22, 33, 60, 76, 76, 76], [8, 26, 34, 56, 76, 76, 76], [8, 30, 35, 54, 76, 76, 76], [9, 18, 36, 68, 76, 76, 76], [9, 22, 37, 50, 61, 69, 72], [9, 26, 38, 54, 57, 70, 73], [9, 30, 39, 71, 76, 76, 76], [10, 18, 40, 64, 76, 76, 76], [10, 22, 41, 54, 62, 65, 74], [10, 26, 42, 50, 58, 66, 75], [10, 30, 43, 67, 76, 76, 76], [11, 18, 44, 54, 76, 76, 76], [11, 22, 45, 63, 76, 76, 76], [11, 26, 46, 59, 76, 76, 76], [11, 30, 47, 50, 76, 76, 76], [12, 19, 32, 51, 60, 68, 72], [12, 23, 33, 69, 76, 76, 76], [12, 27, 34, 70, 76, 76, 76], [12, 31, 35, 55, 56, 71, 73], [13, 19, 36, 61, 76, 76, 76], [13, 23, 37, 51, 76, 76, 76], [13, 27, 38, 55, 76, 76, 76], [13, 31, 39, 57, 76, 76, 76], [14, 19, 40, 62, 76, 76, 76], [14, 23, 41, 55, 76, 76, 76], [14, 27, 42, 51, 76, 76, 76], [14, 31, 43, 58, 76, 76, 76], [15, 19, 44, 55, 63, 64, 74], [15, 23, 45, 65, 76, 76, 76], [15, 27, 46, 66, 76, 76, 76], [15, 31, 47, 51, 59, 67, 75]];

const SCORE_LUT: [(u64, u64); 512] = [(0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x3), (0x0, 0x1000002), (0x0, 0x2000001), (0x0, 0x3000000), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x200), (0x0, 0x100000100), (0x0, 0x200000000), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x10000), (0x0, 0x10000000000), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x1000000000000), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x3, 0x0), (0x1000002, 0x0), (0x2000001, 0x0), (0x3000000, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x200, 0x0), (0x100000100, 0x0), (0x200000000, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x10000, 0x0), (0x10000000000, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x1000000000000, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), ];
const TRACKER_AG: u16 = 0b000_111_111;
const TRACKER_DG: u16 = 0b000_000_111;
const PLACE_STONE_DELTA: [u16; 2] = [
    0b000_111_111,
    0b000_000_111,
];
const STATE_INDEX_MASK: u16 = 0b111_111_111;
const STATE_INIT: [u16; 77] = [
    0x4, 0x4, 0x4, 0x4, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x4, 0x0, 0x0, 0x0, 0x4, 0x0, 0x0, 0x0, 0x4, 0x0, 0x0, 0x0, 0x4, 0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x4, 0x0, 0x0, 0x0, 0x4, 0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x0
];

#[derive(Clone, Debug)]
pub struct LineStateTracker{
    line_state: [u16; 77],
    packed_counts: (u64, u64) // 2 x 8 x 8bit
}

impl LineStateTracker {
    pub fn new() -> Self{
        return LineStateTracker { line_state: STATE_INIT.clone(), packed_counts: (0, 0) };
    }

    pub fn from_uboard((att, def): &UBoard, is_black: u8) -> Self{
        let (black, white) = if is_black == 0{
            (att, def)
        }else{
            (def, att)
        };
        let mut tracker = LineStateTracker::new();
        for i in 0..64{
            if (black >> i) & 1 == 1{
                tracker = tracker.next(i, 0);
            }else if (white >> i) & 1 == 1{
                tracker = tracker.next(i, 1);
            }
        }
        return tracker;
    }

    pub fn get_counts(&self, is_black: u8) -> (usize, usize, usize, usize, usize, usize, usize, usize, usize, usize, usize, usize){
        let null_state = self.line_state[76];
        let null_counts = SCORE_LUT[null_state as usize];
        let b = (self.packed_counts.0 - null_counts.0).to_le_bytes();
        let w = (self.packed_counts.1 - null_counts.1).to_le_bytes();
        
        if is_black == 0 {
            (
                b[0] as usize, b[1] as usize, b[2] as usize, b[3] as usize, b[4] as usize, b[5] as usize,
                w[0] as usize, w[1] as usize, w[2] as usize, w[3] as usize, w[4] as usize, w[5] as usize,
            )
        } else {
            (
                w[0] as usize, w[1] as usize, w[2] as usize, w[3] as usize, w[4] as usize, w[5] as usize,
                b[0] as usize, b[1] as usize, b[2] as usize, b[3] as usize, b[4] as usize, b[5] as usize,
            )
        }
    }

    pub fn next(&self, action_idx: usize, is_black: u8) -> Self{
        let mut new_tracker = self.clone();
        let (mut black_counts, mut white_counts) = new_tracker.packed_counts;
        let lines = &POS_TO_LINE[action_idx];
        let delta = PLACE_STONE_DELTA[is_black as usize];
        // change ground -> att
        for i in 0..7{
            let line_idx = lines[i];
            // ライン状態を取得
            let old_state = new_tracker.line_state[line_idx as usize];
            let old_counts = SCORE_LUT[old_state as usize];
            let new_state = (old_state + delta) & STATE_INDEX_MASK;
            let new_counts = SCORE_LUT[new_state as usize];
            println!("[{line_idx}]state:{old_state:0x}->{new_state:0x}, old_counts:({:0x}, {:0x}), new_counts:({:0x}, {:0x})", old_counts.0, old_counts.1, new_counts.0, new_counts.1);
            black_counts = black_counts - old_counts.0 + new_counts.0;
            white_counts = white_counts - old_counts.1 + new_counts.1;
            new_tracker.line_state[line_idx as usize] = new_state;
        }
        // change float -> ground
        if action_idx < 48{
            let upper_lines = &POS_TO_LINE[action_idx + 16];
            for i in 0..7{
                let line_idx = upper_lines[i];
                let old_state = new_tracker.line_state[line_idx as usize];
                let old_counts = SCORE_LUT[old_state as usize];
                let new_state = (old_state + 1) & STATE_INDEX_MASK;
                let new_counts = SCORE_LUT[new_state as usize];
                black_counts = black_counts - old_counts.0 + new_counts.0;
                white_counts = white_counts - old_counts.1 + new_counts.1;
                new_tracker.line_state[line_idx as usize] = new_state;
            }
        }
        let nc = SCORE_LUT[(self.line_state[76] as usize)];
        println!("now_counts:({:0x}, {:0x}), null_counts:({:0x}, {:0x})", self.packed_counts.0, self.packed_counts.1, nc.0, nc.1);
        let nc = SCORE_LUT[(new_tracker.line_state[76]) as usize];
        println!("new_counts:({:0x}, {:0x}), null_counts:({:0x}, {:0x})", black_counts, white_counts, nc.0, nc.1);
        new_tracker.packed_counts = (black_counts, white_counts);
        return new_tracker;
    }
}

pub trait LineStateTrackEvaluator{
    fn evaluate_line_state_track(b: &UBoard, tracker: &LineStateTracker) -> f32;
}

#[derive(Default, Debug)]
pub struct SearchStats {
    // === 基本ノード情報 ===
    nodes: u64,           // 総探索ノード数
    pub pv_nodes: u64,        // PVノード数
    pub time: u64,
    
    // === TTの効果 ===
    tt_hits: u64,         // TTヒット数
    tt_beta_cutoffs: u64, // TTによるベータカット
    tt_alpha_cutoffs: u64,// TTによるアルファカット
    tt_exact_hits: u64,   // TTで値を手に入れた回数
    tt_move_used: u64,    // TTのbest_moveを順序付けに使った数
    
    // === PVの変化 ===
    pv_changes: u64,               // alphaを更新した回数
    pv_changes_move_idx: Vec<u64>,

    pv_changes_move_idx_low: Vec<u64>, // move_idxがFail::Lowでpvが変わった回数
    pv_changes_move_idx_high: Vec<u64>, // move_idxがFail::Highでpvが変わった回数
    pv_changes_move_idx_ex: Vec<u64>, // move_idxに関係なくpvが変わった回数

    pv_additionals: u64,  // max_valが被った回数
    pv_count: Vec<u64>,   // max_valのカウント（純粋にmax_actionだけを出力するか決定する）
    pub pv_max_idx_frac: Vec<Vec<u64>>,

    // === LMR ===
    lmr_applied: u64,     // LMRを適用した回数
    lmr_researched: u64,  // LMRから再探索が発生した回数
    
    // === カット情報 ===
    cut_nodes: u64,            // betaカットが起きたノード数
    cut_on_first_move: u64,    // 最初の手でカットした回数
    cut_on_later_move: u64,    // 2手目以降でカットした回数

    history_moves: Vec<u64>,
    continuous_history_counts: Vec<u64>,
    continuous_history_moves: Vec<Vec<u64>>,
}

impl SearchStats{
    pub fn new() -> Self{
        let mut stats = Self::default();
        stats.pv_max_idx_frac = vec![vec![0;20];20];
        stats.pv_changes_move_idx = vec![0;20];
        stats.pv_changes_move_idx_low = vec![0;20];
        stats.pv_changes_move_idx_high = vec![0;20];
        stats.pv_changes_move_idx_ex = vec![0;20];
        stats.pv_count = vec![0;20];
        stats.history_moves = vec![0;128];
        stats.continuous_history_counts = vec![0;128];
        stats.continuous_history_moves = vec![vec![0;64];128];
        return stats;
    }
}

// Zobrist Hash table
pub const ZOBRIST_TABLE: [u64; 128] = [0x79239f74454c1d17,0x52bc96a9a779cf4c,0x44bf1e2b664f1188,0x2582231ffa1d7966,0x524c35bebda7ccf5,0x2191d77409871982,0x09bad9bbbab3552d,0x189c7e3173960fcb,0x41e2cb4b84c92f59,0x0392c18520b1e397,0x5e217ebd3c1b9e12,0x38f3789b2c9fc892,0x5552d3b4a2c1d06f,0x09a004c4649b577c,0x279e8bb2a2c1635c,0x716f874cba20764c,0x6480d4385086444f,0x2c5bb1bcaa32efc3,0x75d0e94107d4f7d7,0x5b44ab4ba3b599d5,0x7c9b8d28194fd0a3,0x6d8b40f61543c96c,0x174801760854362c,0x487c2ce0464dd5b7,0x344a5aec673b41b6,0x4f5926d54af6c216,0x05a2b0bbd90e9e55,0x54c0eab385897866,0x6fc85022be342b90,0x302bac6328b1351d,0x6632cd7af185abc1,0x00f9d5251308f39d,0x72b39f2a1efd1945,0x63d69892e3e7657b,0x3a384b7e8d6fd12b,0x129ffc1bbcae9396,0x5640be73e5ebddb1,0x68a441b86a5b14d6,0x2f685465eb251fb6,0x408f6f49cbbad8e9,0x2df0c408da71edbc,0x758ff3602e1ca432,0x6ad1b6c8d9e810ea,0x225e9a0481c7330b,0x4321abf5c5b7bc72,0x3b87f31e19c6e731,0x5af44f6e048ccfdf,0x15a2b542d2c43206,0x55d2087661d1e616,0x7354c7dabd9949b9,0x1b4cc0123d6ed4e7,0x2d584951f5a14a01,0x2277e0878cfd739d,0x62614afe80a3af62,0x199e1d73b821dc03,0x1673c61ac6b2463e,0x372b9290f35eb080,0x1edc9c2c8e4ca9fd,0x1f48bbde44e2a901,0x7aef182fbd40a2b0,0x47940dd371ed931a,0x2503a65e42e51138,0x3179e31a9913b8d3,0x4f99f7de125026e1,0x26bfe3681d4c000f,0x03506999f81b39a1,0x1c15c3888d9a1e62,0x18d7cc79c29e9c7f,0x1e9727f80cceae4c,0x5b48a6a1c3055e2d,0x359335db6fdbac0d,0x2393c02c009652f8,0x1f59f573e9d5f47f,0x0eee5347c2c9c84c,0x2c38ea53efcc9f1d,0x3d26d8bf8023b36d,0x6a3581f6dafb723f,0x0c66aaba76223307,0x5962cde70ee9b6e9,0x60b8faebb119aeea,0x241bbb3055f95a78,0x0329933e7bb3c64f,0x25f0a0181abf1bb0,0x7f80e57b50a6e88e,0x1701186e5eca3069,0x23bfd2bca4b66ad6,0x024118c2665d987b,0x6103993677a89e03,0x59d2517bdc1b5539,0x7bd0aa94335cc412,0x4be0eae5f76f2f4a,0x7186c973bcfa8bd9,0x6b7f48e3bebad30c,0x1507ed90f5eabc86,0x1dff29fa77b374fb,0x5c8fdfa206d8ca4d,0x582504d1c889cfba,0x699e940ce6afc6f8,0x73d38012d326c82c,0x096e6b6041e0476e,0x7fa507df103d3424,0x5f60c68c257f1c18,0x25af03e4c00dca30,0x081435ee1d22fc20,0x426244fb2a3b100b,0x38f48cb5d173a261,0x64a92affb990d724,0x78e1dfdd700a7e4a,0x0253592c791e52df,0x7bffd0702d530f98,0x41ef8b8c9d12f532,0x29d5ebe569f2ef23,0x6140eb22c43f70a8,0x3a9464fc59b2ba1b,0x625a8aa8982c2339,0x63796a0917458daf,0x7ec23e879c7e2175,0x67f4c9ab752ab89f,0x63889481b296fec2,0x5a410fdc355f11d9,0x64d39d6db9e6c3d3,0x71c037ae29995f74,0x005525c2364126d1,0x735bbe68fedaf3e8,0x471b231266b3c44c,0x2867d7b540370b79,0x44b5532033850e34,0x765d176fe6c179e3];
pub const ZOBRIST_INIT_HASH: u64 = 1 << 63;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct TTEntry{
    hash_hi: u32,
    fail: Fail,
    depth: u8,
    best_move: Option<u8>,
}

pub struct TT{
    v: Vec<TTEntry>,
    mask: u64,
}

impl TT{
    fn new(size: usize, mask: u64) -> Self{
        return Self { v: vec![TTEntry { hash_hi: 0, fail: Fail::Ex(0.0), depth: 0, best_move: None }; size], mask }
    }
    fn get(&self, hash: u64) -> Option<&TTEntry>{
        let idx = (hash & self.mask) as usize;
        let entry = self.v[idx];
        if entry.hash_hi as u64 != (hash >> 32){
            return None;
        }
        Some(&self.v[(hash & self.mask) as usize])
    }
    // 強制的に上書き
    fn insert_force(&mut self, hash: u64, entry: TTEntry){
        let idx = (hash & self.mask) as usize;
        self.v[idx] = entry;
    }
}

pub fn call_negscout_hash_lineinfo(b:&Board, depth: u8, e: &Box<dyn LineInfoEvaluator>, tt_size: u64, limit: u64) -> (Action, f32, SearchStats){
    let mut tt = TT::new(1 << tt_size, (1 << tt_size) - 1);
    let (att, def) = b.get_att_def();
    let lineinfo = LineInfo::from_board(b);
    let mut rng = thread_rng();
    
    let (mut action, mut val) = (0, Fail::Ex(0.0));
    let t = Instant::now();
    let limit = limit; // 1秒
    let depth = depth.min((64 - (att.count_ones() + def.count_ones())) as u8);
    let mut search_stats=SearchStats::new();
    for d in (3..=depth).step_by(2){
        // println!("depth:{d}");
        search_stats = SearchStats::new();
        let root_entry = tt.get(ZOBRIST_INIT_HASH);
        let best;
        if let Some(&TTEntry { hash_hi: _, fail: old_val, depth: old_depth, best_move }) = root_entry{
            best = best_move;
        }else{
            best = None;
        }
        (action, val) = negscout_hash_lineinfo(
            (att, def), &lineinfo, 
            ZOBRIST_INIT_HASH, 0, d, d, -2.0, 2.0, None, best, &mut tt, &mut search_stats, &mut rng, e);
        println!("lmr hit rate:{}/{}[{}%]", search_stats.lmr_researched, search_stats.lmr_applied, search_stats.lmr_researched * 100 / (1 + search_stats.lmr_applied));
        // println!("pv nodes:{}", search_stats.pv_nodes);
        let time = t.elapsed().as_micros();
        println!("[negscout_hash_lineinfo, depth:{d}]action:{}, val:{:#?}, time:{time}μs", action.trailing_zeros() % 16, val);
        // println!("time:{time}μs");
        println!("nps: {}", search_stats.pv_nodes * 1000_000 / (1 + time as u64));
        if time > limit as u128{
            // println!("pv changes: {:#?}", search_stats.pv_changes_move_idx);
            // println!("high: {:#?}", search_stats.pv_changes_move_idx_high);
            // println!("low: {:#?}", search_stats.pv_changes_move_idx_low);
            // println!("ex: {:#?}", search_stats.pv_changes_move_idx_ex);
            // println!("pv max_idx: {:#?}", search_stats.pv_max_idx_frac);
            search_stats.time = time as u64;
            // println!("{}", search_stats.history_moves.iter().sum::<u64>());
            break;
        }
        if d != depth{
            continue;
        }
    }
    let time = t.elapsed().as_micros();
    // println!("time:{time}μs");

    return (action, val.get_exval().unwrap(), search_stats);
}

const TURN_DECAY: f32 = 1.0;

// ハッシュ管理
// Late Move Reduction
pub fn negscout_hash_lineinfo(
    (att, def): UBoard, 
    line_info: &LineInfo, 
    now_hash: u64,
    is_att_flag: usize,  // 0の時att
    depth: u8, 
    max_depth: u8,
    alpha: f32,
    beta: f32,
    pre_move: Option<usize>,
    best_move: Option<u8>,
    tt: &mut TT,
    stats: &mut SearchStats,
    rng: &mut impl Rng,
    e: &Box<dyn LineInfoEvaluator>
) -> (Action, Fail){
    use Fail::*;
    stats.pv_nodes += 1;

    let four_mask = get_reach_mask(att, def);
    if four_mask != 0 {
        let action = four_mask & (!four_mask + 1);
        return (action, Ex(1.0))
    }
    let stone = att | def;
    // あと一箇所しか置ける場所がなければ引き分けが確定している。
    if stone.count_ones() == 63 {
        return (!stone, Ex(0.0));
    }

    let mut max_val = -2.0;
    let mut max_move_idx = 0;
    let mut max_action = 0;
    let mut max_action_count = 0;
    let mut alpha = alpha;

    if depth <= 1{
        let mut valid_action_mask = {
            let stone = att | def;
            !stone & ((stone << 16) | 0xffff)
        };
        let mut generator = action_mask2vec(valid_action_mask);
        for (action, idx) in generator{
            let next_line_info = line_info.next(idx);
            let val = -e.evaluate_lineinfo(&(def, att | action), &next_line_info);
            if val > max_val{
                max_val = val;
                max_action = action;
                max_action_count = 1;
                if max_val > alpha {
                    alpha = max_val;
                    if alpha >= beta{
                        return (max_action, High(max_val))
                    }
                }
            }else if val == max_val{
                max_action_count += 1;
                if rng.r#gen::<u32>() % max_action_count == 0{
                    max_action = action;
                }
            }
        }
        return (max_action, Ex(max_val));
    }else{
        let mut action_nb_vals: Vec<(Action, u8, UBoard, LineInfo, f32, u64, (Option<Fail>, u8), Option<u8>)> = Vec::new();

        let mut valid_action_mask = {
            let stone = att | def;
            !stone & ((stone << 16) | 0xffff)
        };
        // PVNode
        'outer : {
            let val;
            if let Some(best_move) = best_move {
                let action = 1u64 << best_move;
                assert_eq!(valid_action_mask & action, action);
                valid_action_mask = valid_action_mask ^ action;
                let next_board = (def, att | action);
                let hash = ZOBRIST_TABLE[best_move as usize + is_att_flag * 64] ^ now_hash;
                let tt_entry = tt.get(hash);
                // reductionはPVNodeでは行わない
                if let Some(&TTEntry { hash_hi: _, fail: old_val, depth: old_depth, best_move: next_best_move }) = tt_entry
                {
                    if old_depth >= depth - 1{
                        stats.tt_hits += 1;
                        match old_val{
                            High(x) => {
                                if alpha >= -x{
                                    stats.tt_alpha_cutoffs += 1;
                                    break 'outer;
                                }
                            },
                            Low(x) => {
                                if beta <= -x{
                                    stats.tt_beta_cutoffs += 1;
                                    stats.pv_max_idx_frac[1 + valid_action_mask.count_ones() as usize + 1][0] += 1;
                                    return (action, High(-x));
                                }
                            },
                            Ex(x) => {
                                stats.tt_exact_hits += 1;
                                if beta <= -x{
                                    stats.tt_beta_cutoffs += 1;
                                    stats.pv_max_idx_frac[1 + valid_action_mask.count_ones() as usize + 1][0] += 1;
                                    return (action, High(-x));
                                }else if alpha >= -x{
                                    stats.tt_alpha_cutoffs += 1;
                                    break 'outer;
                                }
                                // alpha < -x < beta
                                max_action = action;
                                max_move_idx = 0;
                                max_val = -x;
    
                                stats.pv_changes += 1;
                                stats.pv_changes_move_idx[0] += 1;
    
                                alpha = max_val;
                                break 'outer;
                            }
                        }
                    }else{
                        let next_line_info = line_info.next(best_move as usize);
                        let (next_action, next_fail) = negscout_hash_lineinfo(
                            next_board, 
                            &next_line_info, 
                            hash,
                            1 - is_att_flag, 
                            depth - 1, 
                            max_depth, 
                            -beta, 
                            -alpha, 
                            Some(action.trailing_zeros() as usize),
                            next_best_move,
                            tt,
                            stats,
                            rng,
                            e
                        );

                        let new_best_move = if next_action == 0{
                            None
                        }else{
                            Some(next_action.trailing_zeros() as u8)
                        };

                        tt.insert_force(hash, TTEntry { hash_hi: (hash >> 32) as u32, fail: next_fail, depth: depth - 1, best_move:new_best_move });

                        match next_fail{
                            High(v) => {
                                break 'outer;
                            },
                            Low(v) => {
                                stats.pv_max_idx_frac[1 + action_nb_vals.len()][max_move_idx] += 1;
                                return (action, High(-v));
                            },
                            Ex(v) => {
                                val = -v;
                            }
                        }

                        max_val = val;
                        max_move_idx = 0;
                        stats.pv_changes += 1;
                        stats.pv_changes_move_idx[0] += 1;
                        alpha = max_val;
                    }
                }else{
                    let next_line_info = line_info.next(best_move as usize);
                    let (next_action, next_fail) = negscout_hash_lineinfo(
                        next_board, 
                        &next_line_info, 
                        hash,
                        1 - is_att_flag, 
                        depth - 1, 
                        max_depth, 
                        -beta, 
                        -alpha, 
                        Some(action.trailing_zeros() as usize),
                        None,
                        tt,
                        stats,
                        rng,
                        e
                    );

                    let new_best_move = if next_action == 0{
                        None
                    }else{
                        Some(next_action.trailing_zeros() as u8)
                    };

                    tt.insert_force(hash, TTEntry { hash_hi: (hash >> 32) as u32, fail: next_fail, depth: depth - 1, best_move: new_best_move });

                    match next_fail{
                        High(v) => {
                            break 'outer;
                        },
                        Low(v) => {
                            stats.pv_max_idx_frac[1 + action_nb_vals.len()][max_move_idx] += 1;
                            return (action, High(-v));
                        },
                        Ex(v) => {
                            val = -TURN_DECAY * v;
                        }
                    }

                    max_val = val;
                    max_move_idx = 0;
                    stats.pv_changes += 1;
                    stats.pv_changes_move_idx[0] += 1;
                    alpha = max_val;
                }
            }
        }

        let mut generator: Vec<(u64, usize)> = action_mask2vec(valid_action_mask);
        for (idx, &(action, action_idx)) in generator.iter().enumerate(){
            if let Some(&(second_action, second_idx)) =  generator.get(idx + 1){
                let second_hash = ZOBRIST_TABLE[second_idx + is_att_flag * 64] ^ now_hash;
                unsafe {
                    let prefetch_ptr = tt.v.as_ptr().add((second_hash & tt.mask) as usize);
                    // T0ヒント：L1キャッシュまで持ってくる
                    std::arch::x86_64::_mm_prefetch(prefetch_ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
                }
            }

            let next_line_info = line_info.next(action_idx);
            let hash = ZOBRIST_TABLE[action_idx + is_att_flag * 64] ^ now_hash;
            let tt_entry = tt.get(hash);
            let continuous_history_val;


            if let Some(pre_move_idx) = pre_move{
                continuous_history_val = 4.0 * stats.continuous_history_moves[pre_move_idx + is_att_flag * 64][action_idx] as f32 / stats.continuous_history_counts[pre_move_idx].max(1) as f32;
            }else{
                continuous_history_val = 0.0;
            }

            match tt_entry{
                Some(&TTEntry { hash_hi: _, fail: old_val, depth: old_depth, best_move }) => {
                    let fail_val = match old_val{
                        Ex(v) => -v,
                        Low(v) => -v + 0.1,
                        High(v) => -v -1.0,
                    };
                    action_nb_vals.push((
                        action,
                        action_idx as u8,
                        (def, att | action),
                        next_line_info,
                        fail_val + continuous_history_val,
                        hash,
                        (Some(
                            match old_val{
                                Ex(v) => Ex(-v),
                                Low(v) => High(-v),
                                High(v) => Low(-v),
                            }
                        ), old_depth),
                        best_move
                    ))
                }
                None => {
                    let next_board = (def, att | action);
                    let v = -e.evaluate_lineinfo(&next_board, &next_line_info);
                    action_nb_vals.push((action, action_idx as u8, next_board, next_line_info, v + continuous_history_val, hash, (None, 0), None))
                }
            }
        }

        action_nb_vals.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());

        for (move_idx, &(action, action_idx, next_board, ref next_line_info, sort_val, hash, (hit, old_depth), next_best_move)) in action_nb_vals.iter().enumerate(){
            let mut val=-2.0;
            let mut is_set = false;

            // hitありー＞Exのとき値を使う
            // reduced search のスキップ
            //   a < a'(high)のとき
            let reduction_depth = REDUCTION_TABLE[(depth - 1) as usize][move_idx + 1];
            let reduction_flag = !(reduction_depth == depth - 1);

            if let Some(fail_val) = hit.clone()
            && old_depth >= depth - 1
            {
                stats.tt_hits += 1;
                match fail_val{
                    High(x) => {
                        if beta < x{
                            stats.tt_beta_cutoffs += 1;
                            stats.pv_max_idx_frac[action_nb_vals.len()][move_idx] += 1;
                            return (action, High(x));
                        }
                    },
                    Low(x) => {
                        if alpha >= x{
                            stats.tt_alpha_cutoffs += 1;
                            continue;
                        }
                    },
                    Ex(x) => {
                        stats.tt_exact_hits += 1;
                        val = TURN_DECAY * x;
                        is_set = true;
                    }
                }
            }

            let (mut next_action, mut next_fail);

            if !is_set{
                if reduction_flag{
                    // null window search
                    stats.lmr_applied += 1;
                    (next_action, next_fail) = negscout_hash_lineinfo(
                        next_board, 
                        next_line_info, 
                        hash,
                        1 - is_att_flag, 
                        reduction_depth, 
                        max_depth, 
                        -(alpha + 0.00001), 
                        -alpha, 
                        Some(action.trailing_zeros() as usize),
                        next_best_move,
                        tt,
                        stats,
                        rng,
                        e
                    );
                }else{
                    (next_action, next_fail) = negscout_hash_lineinfo(
                        next_board, 
                        next_line_info, 
                        hash,
                        1 - is_att_flag, 
                        depth - 1, 
                        max_depth, 
                        -beta, 
                        -alpha, 
                        Some(action.trailing_zeros() as usize),
                        next_best_move,
                        tt,
                        stats,
                        rng,
                        e
                    );
                }

                let new_best_move = if next_action == 0{
                    None
                }else{
                    Some(next_action.trailing_zeros() as u8)
                };

                // research
                if reduction_flag{
                    if !next_fail.is_fail_high(){            
                        // println!("next_fail:{:#?}, alpha:{}, beta:{}", next_fail, -(alpha + 0.00001), -alpha);
                        stats.lmr_researched += 1;
                        (next_action, next_fail) = negscout_hash_lineinfo(
                            next_board, next_line_info, hash, 1 - is_att_flag, depth - 1, max_depth, -beta, -alpha, Some(action.trailing_zeros() as usize), next_best_move, tt, stats, rng, e
                        );
                        if old_depth < depth - 1{
                            tt.insert_force(hash, TTEntry { hash_hi: (hash >> 32) as u32, fail: next_fail, depth: depth - 1, best_move: new_best_move });
                        }
                    }else if old_depth < reduction_depth{
                        tt.insert_force(hash, TTEntry{ hash_hi: (hash >> 32) as u32, fail: next_fail, depth: reduction_depth, best_move: new_best_move });
                    }
                }else if old_depth < depth - 1{
                    tt.insert_force(hash, TTEntry{ hash_hi: (hash >> 32) as u32, fail: next_fail, depth: depth - 1, best_move: new_best_move });
                }

                match next_fail{
                    High(v) => {
                        continue;
                    },
                    Low(v) => {
                        stats.pv_max_idx_frac[action_nb_vals.len()][max_move_idx] += 1;
                        return (action, High(-v));
                    },
                    Ex(v) => {
                        val = - TURN_DECAY * v;
                    }
                }
            }

            // println!("depth:{depth}, action: {}, val:{val}, sort_val:{sort_val}, alpha:{alpha}, beta:{beta}, max_val:{max_val}", action.trailing_zeros() % 16);
            // if depth == max_depth{
            //     println!("depth:{depth}, action: {}, val:{val}, sort_val:{sort_val}", action.trailing_zeros());
            // }

            if max_val < val{
                max_val = val;
                max_action_count += 1;
                max_move_idx = move_idx;

                max_action = action;
                stats.pv_changes += 1;
                stats.pv_changes_move_idx[move_idx] += 1;
                if let Some(fail_val) = hit{
                    if fail_val.is_fail_low(){
                        stats.pv_changes_move_idx_low[move_idx] += 1;
                    }else if fail_val.is_fail_high(){
                        stats.pv_changes_move_idx_high[move_idx] += 1;
                    }else{
                        stats.pv_changes_move_idx_ex[move_idx] += 1;
                    }
                }

                if alpha < max_val{
                    alpha = max_val;
                    if alpha > beta{
                        stats.cut_nodes += 1;
                        if let Some(pre_move) = pre_move{
                            stats.history_moves[pre_move] += 1;
                            if move_idx == 0{
                                stats.continuous_history_counts[pre_move] += 1;
                                stats.continuous_history_moves[pre_move][action.trailing_zeros() as usize] += 1;
                            }
                        }
                        if move_idx == 0{
                            stats.cut_on_first_move += 1;
                        }else{
                            stats.cut_on_later_move += 1;
                        }
                        // println!("beta cut");
                        return (max_action, High(alpha));
                    }
                }
            }else if max_val == val{
                max_action_count += 1;
                stats.pv_additionals += 1;
                if rng.r#gen::<u32>() % max_action_count == 0{
                    max_action = action;
                }
            }
        }

        stats.pv_count[max_action_count as usize] += 1;

        if max_val < alpha{
            return (max_action, Low(alpha));
        }

        stats.pv_max_idx_frac[action_nb_vals.len()][max_move_idx] += 1;

        return (max_action, Ex(max_val));
    }
}

// impl<E: EvaluatorF> LineInfoEvaluator for E{
//     fn evaluate(&self, board:&UBoard, line_info: &LineInfo) -> f32 {
//         let b = Board::from(board.0, board.1, Player::Black);
//         let f = self.eval_func_f32(&b) * 2.0 - 1.0;
//         return f
//     }
// }

pub struct TestLineAcumModel{
    l: Box<dyn LineInfoEvaluator>,
    pub search_stats: UnsafeCell<SearchStats>,
    pub limit: u64,
    pub max_depth: u64,
}

impl TestLineAcumModel{
    pub fn new(l: SimpleLineInfoEvaluator) -> Self{
        let mut l_ = SimplLineEvaluator::new();
        return Self { l: Box::new(l), search_stats: UnsafeCell::new(SearchStats::new()), limit:1, max_depth:29};
    }
}

impl GetAction for TestLineAcumModel{
    fn get_action(&self, b: &Board) -> u8 {
        let (action, val, stats) = call_negscout_hash_lineinfo(b, self.max_depth as u8, &self.l, 22, self.limit);
        unsafe {
            let k = &mut *self.search_stats.get();
            k.time += stats.time;
            k.pv_nodes += stats.pv_nodes;
            for i in 0..20{
                for j in 0..20{
                    k.pv_max_idx_frac[i][j] += stats.pv_max_idx_frac[i][j];
                }
            }
        }
        return (action.trailing_zeros() % 16) as u8;
    }
}

use crate::ai::line::*;
pub const WT3_SIZE: usize = 16;

#[derive(Clone, Serialize, Deserialize)]
pub struct SimpleLineInfoEvaluator {
    pub wfl3: Vec<f32>,
    pub wgl3: Vec<f32>,
    pub wfl2: Vec<f32>,
    pub wgl2: Vec<f32>,
    pub wfl1: Vec<f32>,
    pub wgl1: Vec<f32>,
    // pub wt3: Vec<f32>,
    pub wt3nw: Vec<f32>,
    pub wt3nb: Vec<f32>,
    pub bias: f32,
    l: SimplLineEvaluator,
}

impl SimpleLineInfoEvaluator {
    pub fn print_table(&self){
        println!("weight float line");
        let v = self.wfl2.clone();
        let length = WL2_WIDTH;
        let base = v[0];
        for i in 0..length{
            for j in 0..length{
                let idx = i * length + j;
                if v[idx] == 0.0{
                    print!("XXXXXXX, ");
                }else{
                    print!("{:>7.3}, ", v[idx] - base);
                }
            }
            println!("");
        }
        println!("weight ground line");
        let v = self.wgl2.clone();
        let length = WL2_WIDTH;
        let base = v[0];
        for i in 0..length{
            for j in 0..length{
                let idx = i * length + j;
                if v[idx] == 0.0{
                    print!("XXXXXXX, ");
                }else{
                    print!("{:>7.3}, ", v[idx] - base)
                }
            }
            println!("");
        }

    }
    pub fn new() -> Self {
        let mut l = SimplLineEvaluator::new();
        l.load(String::from("simple.json"));
        return SimpleLineInfoEvaluator {
            wfl3: vec![0.0; WFL3_WIDTH * WFL3_WIDTH],
            wgl3: vec![0.0; WGL3_WIDTH * WGL3_WIDTH],
            wfl2: vec![0.0; 64 * 64],
            wgl2: vec![0.0; 64 * 64],
            wfl1: vec![0.0; WFL1_WIDTH * WFL1_WIDTH],
            wgl1: vec![0.0; WGL1_WIDTH * WGL1_WIDTH],
            // wt3: vec![0.0; 3 * WT3_SIZE * WT3_SIZE],
            wt3nb: vec![0.0; 12],
            wt3nw: vec![0.0; 12],
            bias: 0.0,
            l: l
        };
    }

    pub fn from_sle(s: &SimplLineEvaluator) -> Self{
        return SimpleLineInfoEvaluator { 
            wfl3: s.wfl3.clone(), 
            wgl3: s.wgl3.clone(), 
            wfl2: s.wfl2.clone(), 
            wgl2: s.wgl2.clone(), 
            wfl1: s.wfl1.clone(), 
            wgl1: s.wgl1.clone(), 
            // wt3: vec![0.0; 3 * WT3_SIZE * WT3_SIZE], 
            wt3nb: s.wt3nb.clone(),
            wt3nw: s.wt3nw.clone(),
            bias: s.bias,
            l: s.clone()
        };
    }

    pub fn get_counts(
        (att, def): &UBoard, line_info: &LineInfo
    ) -> (
        usize, usize, usize, usize, usize, usize, usize, usize, usize, usize, usize, usize, usize, 
    ) {
        let stone = att | def;
        let ground = !stone & (stone << 16 | 0xffff);
        let float = !stone ^ ground;
        let is_black = stone.count_ones() % 2 == 0;

        let mut a1g = 0u32; let mut a1f = 0u32;
        let mut a2g = 0u32; let mut a2f = 0u32;
        let mut a3g = 0u32; let mut a3f = 0u32;
        let mut d1g = 0u32; let mut d1f = 0u32;
        let mut d2g = 0u32; let mut d2f = 0u32;
        let mut d3g = 0u32; let mut d3f = 0u32;
        
        let mut a3 = 0;
        let mut d3 = 0;

        let mut a3_or = 0u64;
        let mut d3_or = 0u64;
        let mut a3_or_l24 = 0u64;
        let mut d3_or_l24 = 0u64;
        // let mut tb_or = 0u64;
        // let mut tw_or = 0u64;
        let mut l3 = 0u64;

        let layer24_mask = 0xffff_0000_ffff_0000u64 & float;
        let layer3_mask  = 0x0000_ffff_0000_0000u64 & float;

        for i in 0..7 {
            let a0 = line_info.att.0[i];
            let a1 = line_info.att.1[i];
            let a2 = line_info.att.2[i];
            let d0 = line_info.def.0[i];
            let d1 = line_info.def.1[i];
            let d2 = line_info.def.2[i];

            let va1 = (a0 ^ a1) & !d0;
            let va2 = (a1 ^ a2) & !d0;
            let va3 = a2 & !d0;
            let vd1 = (d0 ^ d1) & !a0;
            let vd2 = (d1 ^ d2) & !a0;
            let vd3 = d2 & !a0;

            a1g += (va1 & ground).count_ones();
            a1f += (va1 & float).count_ones();
            a2g += (va2 & ground).count_ones();
            a2f += (va2 & float).count_ones();

            // a3g += (va3 & ground).count_ones();
            // a3f += (va3 & float).count_ones();
            a3 |= va3;
            
            d1g += (vd1 & ground).count_ones();
            d1f += (vd1 & float).count_ones();
            d2g += (vd2 & ground).count_ones();
            d2f += (vd2 & float).count_ones();

            // d3g += (vd3 & ground).count_ones();
            // d3f += (vd3 & float).count_ones();
            d3 |= vd3;

            // a3_or_l24 |= va3 & layer24_mask;
            // d3_or_l24 |= vd3 & layer24_mask;
            // tb_or |= (va3 & !vd3) & layer3_mask;
            // tw_or |= (!va3 & vd3) & layer3_mask;
            l3 |= (a2 | d2) & !stone;
        }

        // if tb_or == 0xffff00000000 || tw_or == 0xffff00000000{
        //     pprint_uboard((*att, *def));
        //     pprint_u64(*att);
        //     println!("-----");
        //     pprint_u64(*def);
        // }

        // let (gg, tb, tw) = if is_black{
        //     let mut gg = d3_or_l24.count_ones() as usize;
        //     if gg != 0{
        //         gg = (gg % 2) + 1;
        //     }
        //     let tb = tb_or.count_ones() as usize;
        //     let tw = tw_or.count_ones() as usize;
        //     (gg, tb, tw)
        // }else{
        //     let mut gg = a3_or_l24.count_ones() as usize;
        //     if gg != 0{
        //         gg = (gg % 2) + 1;
        //     }
        //     let tb = tw_or.count_ones() as usize;
        //     let tw = tb_or.count_ones() as usize;
        //     (gg, tb, tw)
        // };
        let a3g = (a3 & ground).count_ones();
        let a3f = (a3 & float).count_ones();
        let d3g = (d3 & ground).count_ones();
        let d3f = (d3 & float).count_ones();

        let trap_3_num = (l3 & (!l3 << 16) & 0x0000_ffff_0000_0000).count_ones() as usize;

        // let ans = (
        //     a1f as usize, a2f as usize, a3f as usize, a1g as usize, a2g as usize, a3g as usize, d1f as usize, d2f as usize, d3f as usize, d1g as usize, d2g as usize, d3g as usize, gg as usize, tb as usize, tw as usize
        // );
        let ans = (
            a1f as usize, a2f as usize, a3f as usize, a1g as usize, a2g as usize, a3g as usize, d1f as usize, d2f as usize, d3f as usize, d1g as usize, d2g as usize, d3g as usize, trap_3_num
        );
        return ans;
    }

    pub fn get_eval_from_counts(
        // &self, af1: usize, af2: usize, af3: usize, ag1: usize, ag2: usize, ag3: usize, df1: usize, df2: usize, df3: usize, dg1: usize, dg2: usize, dg3:usize, gg: usize, tb: usize, tw: usize
        &self, af1: usize, af2: usize, af3: usize, ag1: usize, ag2: usize, ag3: usize, df1: usize, df2: usize, df3: usize, dg1: usize, dg2: usize, dg3:usize, t3n: usize, is_black: bool
    ) -> f32{
        let mut val = 0.0;

        // if gg * WT3_SIZE * WT3_SIZE + tb * WT3_SIZE + tw >= self.wt3.len(){
        //     println!("gg:{gg}, tb:{tb}, tw:{tw}");
        // }
        if is_black {
            val += self.wt3nb[t3n];
        }else{
            val += self.wt3nw[t3n];
        }


        val += self.wfl1[af1 * WFL1_WIDTH + df1]
            + self.wfl2[af2 * WL2_WIDTH + df2]
            + self.wfl3[af3 * WFL3_WIDTH + df3]
            + self.wgl1[ag1 * WGL1_WIDTH + dg1]
            + self.wgl2[ag2 * WL2_WIDTH + dg2]
            + self.wgl3[ag3 * WGL3_WIDTH + dg3]
            // + self.wt3[gg * WT3_SIZE * WT3_SIZE + tb * WT3_SIZE + tw]
            + self.bias;
        return 1.0 / (1.0 + (-val).exp());
    }

    pub fn evaluate_board(&self, b: &UBoard, info: &LineInfo) -> f32 {
        let is_black = (b.0 | b.1).count_ones() % 2 == 0;
        // let (a1, a2, a3, a4, a5, a6, d1, d2, d3, d4, d5, d6, gg, tb, tw) = Self::get_counts(b, info);
        let (a1, a2, a3, a4, a5, a6, d1, d2, d3, d4, d5, d6, t3n) = Self::get_counts(b, info);
        // let v = self.get_eval_from_counts(a1, a2, a3, a4, a5, a6, d1, d2, d3, d4, d5, d6, gg, tb, tw);
        let v = self.get_eval_from_counts(a1, a2, a3, a4, a5, a6, d1, d2, d3, d4, d5, d6, t3n, is_black);
        return v;
    }

    pub fn save(&self, name: String) -> Result<()> {
        use anyhow::Context;
        use std::fs::File;
        use std::io::{BufWriter, Write};

        let data_str = serde_json::to_string(self)?;

        let file = File::create(name)?;
        let mut buff_writer: BufWriter<File> = BufWriter::new(file);

        buff_writer
            .write(data_str.as_bytes())
            .context("write error")?;
        buff_writer.flush().context("flush error")?;

        Ok(())
    }
    pub fn load(&mut self, name: String) -> Result<()> {
        use anyhow::Context;
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(name)?;
        let buff_reader: BufReader<File> = BufReader::new(file);

        let mut lines = Vec::new();

        for line in buff_reader.lines() {
            // if process here, can save memory
            lines.push(line.context("read error")?);
        }
        let data_str = lines.join("\n");
        let mut src: SimpleLineInfoEvaluator = serde_json::from_str(&data_str)?;

        std::mem::swap(self, &mut src);

        Ok(())
    }
}

impl LineInfoEvaluator for SimpleLineInfoEvaluator {
    fn evaluate_lineinfo(&self, b: &UBoard, info: &LineInfo) -> f32 {
        // let e = self.l.evaluate_board(&Board::from(b.0, b.1, Player::Black)) * 2.0 - 1.0;
        let self_e = self.evaluate_board(b, info) * 2.0 - 1.0;
        // assert!((e-self_e).abs() < 0.01, "e:{e}, self_e:{self_e}");
        return self_e;
    }
}

#[derive(Clone)]
pub struct TrainableSLIE {
    pub main: SimpleLineInfoEvaluator,
    v: SimpleLineInfoEvaluator,
    m: SimpleLineInfoEvaluator,
    lr: f32,
}

impl TrainableSLIE {
    pub fn new(lr: f32) -> Self {
        TrainableSLIE {
            main: SimpleLineInfoEvaluator::new(),
            v: SimpleLineInfoEvaluator::new(),
            m: SimpleLineInfoEvaluator::new(),
            lr: lr,
        }
    }

    pub fn from(e: SimpleLineInfoEvaluator, lr: f32) -> Self {
        TrainableSLIE {
            main: e,
            v: SimpleLineInfoEvaluator::new(),
            m: SimpleLineInfoEvaluator::new(),
            lr: lr,
        }
    }
}

impl Trainable for TrainableSLIE {
    fn update(&mut self, b: &Board, delta: f32) {
        let (att, def) = b.get_att_def();
        let line_info = LineInfo::from_board(b);
        // let (a1, a2, a3, a1_, a2_, a3_, d1, d2, d3, d1_, d2_, d3_, gg, tb, tw) =
        //     SimpleLineInfoEvaluator::get_counts(&(att, def), &line_info);
        let (a1, a2, a3, a1_, a2_, a3_, d1, d2, d3, d1_, d2_, d3_, tn3) =
            SimpleLineInfoEvaluator::get_counts(&(att, def), &line_info);
        // println!("{:#?}", (gg, tb, tw));
        // とりあえずsgd
        let val = self.main.evaluate_board(&(att, def), &line_info);
        let dv = val * (1.0 - val);
        let delta = self.lr * delta * dv;
        self.main.wfl1[a1 * WFL1_WIDTH + d1] += delta;
        // self.main.wfl1[d1 * WFL1_WIDTH + a1] -= delta;

        self.main.wfl2[a2 * WL2_WIDTH + d2] += delta;
        // self.main.wfl2[d2 * WL2_WIDTH + a2] -= delta;
        
        self.main.wfl3[a3 * WFL3_WIDTH + d3] += delta;
        // self.main.wfl3[d3 * WFL3_WIDTH + a3] -= delta;

        self.main.wgl1[a1_ * WGL1_WIDTH + d1_] += delta;
        // self.main.wgl1[d1_ * WGL1_WIDTH + a1_] -= delta;

        self.main.wgl2[a2_ * WL2_WIDTH + d2_] += delta;
        // self.main.wgl2[d2_ * WL2_WIDTH + a2_] -= delta;
        
        self.main.wgl3[a3_ * WGL3_WIDTH + d3_] += delta;
        // self.main.wgl3[d3_ * WGL3_WIDTH + a3_] -= delta;
        
        if (att | def).count_ones() % 2 == 0{
            self.main.wt3nb[tn3] += delta;
        }else{
            self.main.wt3nw[tn3] += delta;
        }
        // self.main.wt3[gg * WT3_SIZE * WT3_SIZE + tb * WT3_SIZE + tw] += delta;
        self.main.bias += delta;
    }

    fn get_val(&self, b: &Board) -> f32 {
        self.main.evaluate_board(&(b.get_att_def()), &LineInfo::from_board(b))
    }

    fn save(&self, file: String) -> Result<()> {
        self.main.save(file)
    }
    fn load(&mut self, file: String) -> Result<()> {
        self.main.load(file)
    }
    fn eval(&mut self) {}
    fn train(&mut self) {}
}

impl EvaluatorF for TrainableSLIE {
    fn eval_func_f32(&self, b: &Board) -> f32 {
        return self.main.evaluate_board(&(b.get_att_def()), &LineInfo::from_board(b)).clamp(0.0, 1.0);
    }
}