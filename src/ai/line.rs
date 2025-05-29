use crate::board::Board;

use super::{
    acum_mask_bundle, acum_or, apply_mask_bundle, EvaluatorF, LineEvaluator, LineMaskBundle,
    Trainable,
};
use anyhow::{Ok, Result};
use serde::{Deserialize, Serialize};

const WFL3_WIDTH: usize = 32;
const WGL3_WIDTH: usize = 16;
const WL2_WIDTH: usize = 64;
const WGL1_WIDTH: usize = 96;
const WFL1_WIDTH: usize = 106;

#[derive(Clone, Serialize, Deserialize)]
pub struct SimplLineEvaluator {
    pub wfl3: Vec<f32>,
    pub wgl3: Vec<f32>,
    pub wfl2: Vec<f32>,
    pub wgl2: Vec<f32>,
    pub wfl1: Vec<f32>,
    pub wgl1: Vec<f32>,
    pub wt3nw: Vec<f32>,
    pub wt3nb: Vec<f32>,
    pub bias: f32,
}

impl SimplLineEvaluator {
    pub fn analyze_board(a: u64, d: u64) -> (LineMaskBundle, LineMaskBundle, LineMaskBundle) {
        let stone = a | d;
        let b = !stone;
        let (
            a1,
            a2,
            a3,
            a4,
            a5,
            a6,
            a8,
            a9,
            a10,
            a11,
            a12,
            a13,
            a15,
            a16,
            a17,
            a19,
            a20,
            a21,
            a22,
            a24,
            a26,
            a30,
            a32,
            a33,
            a34,
            a36,
            a38,
            a39,
            a40,
            a42,
            a45,
            a48,
            a51,
            a57,
            a60,
            a63,
        ) = (
            a >> 1,
            a >> 2,
            a >> 3,
            a >> 4,
            a >> 5,
            a >> 6,
            a >> 8,
            a >> 9,
            a >> 10,
            a >> 11,
            a >> 12,
            a >> 13,
            a >> 15,
            a >> 16,
            a >> 17,
            a >> 19,
            a >> 20,
            a >> 21,
            a >> 22,
            a >> 24,
            a >> 26,
            a >> 30,
            a >> 32,
            a >> 33,
            a >> 34,
            a >> 36,
            a >> 38,
            a >> 39,
            a >> 40,
            a >> 42,
            a >> 45,
            a >> 48,
            a >> 51,
            a >> 57,
            a >> 60,
            a >> 63,
        );
        let (
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b8,
            b9,
            b10,
            b11,
            b12,
            b13,
            b15,
            b16,
            b17,
            b19,
            b20,
            b21,
            b22,
            b24,
            b26,
            b30,
            b32,
            b33,
            b34,
            b36,
            b38,
            b39,
            b40,
            b42,
            b45,
            b48,
            b51,
            b57,
            b60,
            b63,
        ) = (
            b >> 1,
            b >> 2,
            b >> 3,
            b >> 4,
            b >> 5,
            b >> 6,
            b >> 8,
            b >> 9,
            b >> 10,
            b >> 11,
            b >> 12,
            b >> 13,
            b >> 15,
            b >> 16,
            b >> 17,
            b >> 19,
            b >> 20,
            b >> 21,
            b >> 22,
            b >> 24,
            b >> 26,
            b >> 30,
            b >> 32,
            b >> 33,
            b >> 34,
            b >> 36,
            b >> 38,
            b >> 39,
            b >> 40,
            b >> 42,
            b >> 45,
            b >> 48,
            b >> 51,
            b >> 57,
            b >> 60,
            b >> 63,
        );
        let (x1, x2, x3) =
            LineEvaluator::analyze_line(a, a1, a2, a3, b, b1, b2, b3, 0x1111_1111_1111_1111, 0xf);
        let (x1, x2, x3) = (x1 & b, x2 & b, x3 & b);

        let (y1, y2, y3) = LineEvaluator::analyze_line(
            a,
            a4,
            a8,
            a12,
            b,
            b4,
            b8,
            b12,
            0x000f_000f_000f_000f,
            0x1111,
        );
        let (y1, y2, y3) = (y1 & b, y2 & b, y3 & b);

        let (z1, z2, z3) = LineEvaluator::analyze_line(
            a,
            a16,
            a32,
            a48,
            b,
            b16,
            b32,
            b48,
            0xffff,
            0x0001_0001_0001_0001,
        );
        let (xy1, xy2, xy3) = LineEvaluator::analyze_line(
            a,
            a5,
            a10,
            a15,
            b,
            b5,
            b10,
            b15,
            0x0001_0001_0001_0001,
            0x8421,
        );
        let (xy1_, xy2_, xy3_) =
            LineEvaluator::analyze_line(a, a3, a6, a9, b, b3, b6, b9, 0x0008_0008_0008_0008, 0x249);
        let (xz1, xz2, xz3) = LineEvaluator::analyze_line(
            a,
            a17,
            a34,
            a51,
            b,
            b17,
            b34,
            b51,
            0x1111,
            0x0008_0004_0002_0001,
        );
        let (xz1_, xz2_, xz3_) = LineEvaluator::analyze_line(
            a,
            a15,
            a30,
            a45,
            b,
            b15,
            b30,
            b45,
            0x8888,
            0x2000_4000_8001,
        );
        let (yz1, yz2, yz3) = LineEvaluator::analyze_line(
            a,
            a20,
            a40,
            a60,
            b,
            b20,
            b40,
            b60,
            0x000f,
            0x1000_0100_0010_0001,
        );
        let (yz1_, yz2_, yz3_) = LineEvaluator::analyze_line(
            a,
            a12,
            a24,
            a36,
            b,
            b12,
            b24,
            b36,
            0xf000,
            0x0000_0010_0100_1001,
        );

        let xyz1d = 0x8000_0400_0020_0001 & d;
        let (xyz11, xyz12, xyz13) = if xyz1d == 0 {
            let xyz1_count = (0x8000_0400_0020_0001 & a).count_ones();
            if xyz1_count > 1 {
                if xyz1_count == 3 {
                    (0, 0, 0x8000_0400_0020_0001 & b)
                } else {
                    (0, 0x8000_0400_0020_0001 & b, 0)
                }
            } else if xyz1_count == 1 {
                (0x8000_0400_0020_0001 & b, 0, 0)
            } else {
                (0, 0, 0)
            }
        } else {
            (0, 0, 0)
        };
        let xyz2d = 0x1000_0200_0040_0008 & d;
        let (xyz21, xyz22, xyz23) = if xyz2d == 0 {
            let xyz2_count = (0x1000_0200_0040_0008 & a).count_ones();
            if xyz2_count > 1 {
                if xyz2_count == 3 {
                    (0, 0, 0x1000_0200_0040_0008 & b)
                } else {
                    (0, 0x1000_0200_0040_0008 & b, 0)
                }
            } else if xyz2_count == 1 {
                (0x1000_0200_0040_0008 & b, 0, 0)
            } else {
                (0, 0, 0)
            }
        } else {
            (0, 0, 0)
        };
        let xyz3d = 0x0008_0040_0200_1000 & d;
        let (xyz31, xyz32, xyz33) = if xyz3d == 0 {
            let xyz3_count = (0x0008_0040_0200_1000 & a).count_ones();
            if xyz3_count > 1 {
                if xyz3_count == 3 {
                    (0, 0, 0x0008_0040_0200_1000 & b)
                } else {
                    (0, 0x0008_0040_0200_1000 & b, 0)
                }
            } else if xyz3_count == 1 {
                (0x0008_0040_0200_1000 & b, 0, 0)
            } else {
                (0, 0, 0)
            }
        } else {
            (0, 0, 0)
        };
        let xyz4d = 0x0001_0020_0400_8000 & d;
        let (xyz41, xyz42, xyz43) = if xyz4d == 0 {
            let xyz4_count = (0x0001_0020_0400_8000 & a).count_ones();
            if xyz4_count > 1 {
                if xyz4_count == 3 {
                    (0, 0, 0x0001_0020_0400_8000 & b)
                } else {
                    (0, 0x0001_0020_0400_8000 & b, 0)
                }
            } else if xyz4_count == 1 {
                (0x0001_0020_0400_8000 & b, 0, 0)
            } else {
                (0, 0, 0)
            }
        } else {
            (0, 0, 0)
        };

        let (z1, z2, z3) = (z1 & b, z2 & b, z3 & b);
        let (xy1, xy2, xy3) = (xy1 & b, xy2 & b, xy3 & b);
        let (xy1_, xy2_, xy3_) = (xy1_ & b, xy2_ & b, xy3_ & b);
        let (yz1, yz2, yz3) = (yz1 & b, yz2 & b, yz3 & b);
        let (yz1_, yz2_, yz3_) = (yz1_ & b, yz2_ & b, yz3_ & b);
        let (xz1, xz2, xz3) = (xz1 & b, xz2 & b, xz3 & b);
        let (xz1_, xz2_, xz3_) = (xz1_ & b, xz2_ & b, xz3_ & b);
        let (xyz11, xyz12, xyz13) = (xyz11 & b, xyz12 & b, xyz13 & b);
        let (xyz21, xyz22, xyz23) = (xyz21 & b, xyz22 & b, xyz23 & b);
        let (xyz31, xyz32, xyz33) = (xyz31 & b, xyz32 & b, xyz33 & b);
        let (xyz41, xyz42, xyz43) = (xyz41 & b, xyz42 & b, xyz43 & b);

        return (
            (
                x1, y1, z1, xy1, xy1_, yz1, yz1_, xz1, xz1_, xyz11, xyz21, xyz31, xyz41,
            ),
            (
                x2, y2, z2, xy2, xy2_, yz2, yz2_, xz2, xz2_, xyz12, xyz22, xyz32, xyz42,
            ),
            (
                x3, y3, z3, xy3, xy3_, yz3, yz3_, xz3, xz3_, xyz13, xyz23, xyz33, xyz43,
            ),
        );
    }
    pub fn new() -> Self {
        return SimplLineEvaluator {
            wfl3: vec![0.0; WFL3_WIDTH * WFL3_WIDTH],
            wgl3: vec![0.0; WGL3_WIDTH * WGL3_WIDTH],
            wfl2: vec![0.0; 64 * 64],
            wgl2: vec![0.0; 64 * 64],
            wfl1: vec![0.0; WFL1_WIDTH * WFL1_WIDTH],
            wgl1: vec![0.0; WGL1_WIDTH * WGL1_WIDTH],
            wt3nw: vec![0.0; 12],
            wt3nb: vec![0.0; 12],
            bias: 0.0,
        };
    }

    pub fn get_counts(
        b: &Board,
    ) -> (
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    ) {
        let (att, def) = b.get_att_def();
        let (a1, a2, a3, _, _, _) = LineEvaluator::analyze_board(att, def);
        let (d1, d2, d3, _, _, _) = LineEvaluator::analyze_board(def, att);
        let stone = att | def;
        let ground = !stone & (stone << 16 | 0xffff);
        let float = !stone ^ ground;
        let a1_float = acum_mask_bundle(apply_mask_bundle(a1, float)) as usize;
        let a1_ground = acum_mask_bundle(apply_mask_bundle(a1, ground)) as usize;
        let a2_float = acum_mask_bundle(apply_mask_bundle(a2, float)) as usize;
        let a2_ground = acum_mask_bundle(apply_mask_bundle(a2, ground)) as usize;
        let a3_float = (acum_or(a3) & float).count_ones() as usize;
        let a3_ground = (acum_or(a3) & ground).count_ones() as usize;
        let d1_float = acum_mask_bundle(apply_mask_bundle(d1, float)) as usize;
        let d1_ground = acum_mask_bundle(apply_mask_bundle(d1, ground)) as usize;
        let d2_float = acum_mask_bundle(apply_mask_bundle(d2, float)) as usize;
        let d2_ground = acum_mask_bundle(apply_mask_bundle(d2, ground)) as usize;
        let d3_float = (acum_or(d3) & float).count_ones() as usize;
        let d3_ground = (acum_or(d3) & ground).count_ones() as usize;

        let l3_mask = acum_or(d3) | acum_or(a3);
        let trap_3_num = (l3_mask & (!l3_mask << 16) & 0x0000_ffff_0000_0000).count_ones() as usize;

        return (
            a1_float, a2_float, a3_float, a1_ground, a2_ground, a3_ground, d1_float, d2_float,
            d3_float, d1_ground, d2_ground, d3_ground, trap_3_num,
        );
    }

    pub fn evaluate_board(&self, b: &Board) -> f32 {
        let (af1, af2, af3, ag1, ag2, ag3, df1, df2, df3, dg1, dg2, dg3, tn3) =
            SimplLineEvaluator::get_counts(&b);
        let (att, def) = b.get_att_def();
        let is_black = (att.count_ones() + def.count_ones()) % 2 == 0;
        let mut val = 0.0;

        if is_black {
            val += self.wt3nb[tn3];
        } else {
            val += self.wt3nw[tn3];
        }
        val += self.wfl1[af1 * WFL1_WIDTH + df1]
            + self.wfl2[af2 * WL2_WIDTH + df2]
            + self.wfl3[af3 * WFL3_WIDTH + df3]
            + self.wgl1[ag1 * WGL1_WIDTH + dg1]
            + self.wgl2[ag2 * WL2_WIDTH + dg2]
            + self.wgl3[ag3 * WGL3_WIDTH + dg3]
            // + self.wdfl1[96 - df1 + af1]
            // + self.wdfl2[64 - df2 + af2]
            // + self.wdfl3[32 - df3 + af3]
            // + self.wdgl1[96 + ag1 - dg1]
            // + self.wdgl2[64 + ag2 - dg2]
            // + self.wdgl3[32 + ag3 - dg3]
            + self.bias;
        return 1.0 / (1.0 + (-val).exp());
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
        let mut src: SimplLineEvaluator = serde_json::from_str(&data_str)?;

        std::mem::swap(self, &mut src);

        Ok(())
    }
}

impl EvaluatorF for SimplLineEvaluator {
    fn eval_func_f32(&self, b: &Board) -> f32 {
        return self.evaluate_board(b);
    }
}

#[derive(Clone)]
pub struct TrainableSLE {
    main: SimplLineEvaluator,
    v: SimplLineEvaluator,
    m: SimplLineEvaluator,
    lr: f32,
}

impl TrainableSLE {
    pub fn new(lr: f32) -> Self {
        TrainableSLE {
            main: SimplLineEvaluator::new(),
            v: SimplLineEvaluator::new(),
            m: SimplLineEvaluator::new(),
            lr: lr,
        }
    }

    pub fn from(e: SimplLineEvaluator, lr: f32) -> Self {
        TrainableSLE {
            main: e,
            v: SimplLineEvaluator::new(),
            m: SimplLineEvaluator::new(),
            lr: lr,
        }
    }
}

impl Trainable for TrainableSLE {
    fn update(&mut self, b: &Board, delta: f32) {
        let (a1, a2, a3, a1_, a2_, a3_, d1, d2, d3, d1_, d2_, d3_, trap_3_num) =
            SimplLineEvaluator::get_counts(b);
        let is_black = (b.black.count_ones() + b.white.count_ones()) % 2 == 0;
        // とりあえずsgd
        let val = self.main.evaluate_board(b);
        let dv = val * (1.0 - val);
        let delta = self.lr * delta * dv;
        self.main.wfl1[a1 * WFL1_WIDTH + d1] += delta;
        self.main.wfl2[a2 * WL2_WIDTH + d2] += delta;
        self.main.wfl3[a3 * WFL3_WIDTH + d3] += delta;
        self.main.wgl1[a1_ * WGL1_WIDTH + d1_] += delta;
        self.main.wgl2[a2_ * WL2_WIDTH + d2_] += delta;
        self.main.wgl3[a3_ * WGL3_WIDTH + d3_] += delta;
        // self.main.wdfl1[96 + a1 - d1] += delta;
        // self.main.wdfl2[64 + a2 - d2] += delta;
        // self.main.wdfl3[32 + a3 - d3] += delta;
        // self.main.wdgl1[96 + a1_ - d1_] += delta;
        // self.main.wdgl2[64 + a2_ - d2_] += delta;
        // self.main.wdgl3[32 + a3_ - d3_] += delta;
        if is_black {
            self.main.wt3nb[trap_3_num] += delta;
        } else {
            self.main.wt3nw[trap_3_num] += delta;
        }
        self.main.bias += delta;
    }

    fn get_val(&self, b: &Board) -> f32 {
        self.main.evaluate_board(b)
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

impl EvaluatorF for TrainableSLE {
    fn eval_func_f32(&self, b: &Board) -> f32 {
        return self.main.evaluate_board(b).clamp(0.0, 1.0);
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SimplePatternEvaluator {
    // big_h: Vec<f32>,
    // piller_3x4: Vec<f32>,
    // bottom_corner: Vec<f32>,
    p4x4: Vec<f32>,
}

impl SimplePatternEvaluator {
    const BIGH_SIZE: usize = 1 << 16;
    const BIGH_MAGIC: u64 = 0x8008101000200040;

    const PILLER3X4_SIZE: usize = 1 << 16;
    const PILLER3X4_MAGIC: u64 = 0x1011000200040000;

    const BOTTOM_CORNER_SIZE: usize = 1 << 16;
    const BOTTOM_CORNER_MAGIC: u64 = 0x100800100020000;

    const fn _get_piller4x4_idx(a: u64, d: u64) -> usize {
        // a = ......0000_1111
        //           ^^^^ ^^^^
        //          black white

        let a = (a + (d << 1));
        let a = (a & 0x0000_ffff_0000_ffff) + ((a >> 16) & 0x0000_ffff_0000_ffff) * 31;
        let a = (a & 0xffff_ffff) + ((a >> 32) & 0xffff_ffff) * 31 * 31;

        return a as usize;
    }

    const fn get_piller4x4_idx(a: u64, d: u64) -> [usize; 10] {
        let a = Board::u64_xzflip(a);
        let d = Board::u64_xzflip(d);
        let mut idxs: [usize; 10] = [0; 10];
        idxs[0] = Self::_get_piller4x4_idx(a & 0x000f_000f_000f_000f, d & 0x000f_000f_000f_000f);
        idxs[1] = Self::_get_piller4x4_idx(
            (a >> 4) & 0x000f_000f_000f_000f,
            (d >> 4) & 0x000f_000f_000f_000f,
        );
        idxs[2] = Self::_get_piller4x4_idx(
            (a >> 8) & 0x000f_000f_000f_000f,
            (d >> 8) & 0x000f_000f_000f_000f,
        );
        idxs[3] = Self::_get_piller4x4_idx(
            (a >> 12) & 0x000f_000f_000f_000f,
            (d >> 12) & 0x000f_000f_000f_000f,
        );

        let (a, d) = (Board::u64_yzflip(a), Board::u64_yzflip(d));
        idxs[4] = Self::_get_piller4x4_idx(a & 0x000f_000f_000f_000f, d & 0x000f_000f_000f_000f);
        idxs[5] = Self::_get_piller4x4_idx(
            (a >> 4) & 0x000f_000f_000f_000f,
            (d >> 4) & 0x000f_000f_000f_000f,
        );
        idxs[6] = Self::_get_piller4x4_idx(
            (a >> 8) & 0x000f_000f_000f_000f,
            (d >> 8) & 0x000f_000f_000f_000f,
        );
        idxs[7] = Self::_get_piller4x4_idx(
            (a >> 12) & 0x000f_000f_000f_000f,
            (d >> 12) & 0x000f_000f_000f_000f,
        );
        idxs[8] = Self::_get_piller4x4_idx(
            a & 0x000f
                | (a >> 4) & 0xf_0000
                | (a >> 8) & 0xf_0000_0000
                | (a >> 12) & 0xf_0000_0000_0000,
            d & 0x000f
                | (d >> 4) & 0xf_0000
                | (d >> 8) & 0xf_0000_0000
                | (d >> 12) & 0xf_0000_0000_0000,
        );
        idxs[9] = Self::_get_piller4x4_idx(
            (a >> 12) & 0x000f
                | (a >> 8) & 0xf_0000
                | (a >> 4) & 0xf_0000_0000
                | a & 0xf_0000_0000_0000,
            (d >> 12) & 0x000f
                | (d >> 8) & 0xf_0000
                | (d >> 4) & 0xf_0000_0000
                | d & 0xf_0000_0000_0000,
        );
        return idxs;
    }

    fn get_bigh_idx(a: u64, d: u64) -> [usize; 10] {
        let a_ = Board::u64_dflip(a);
        let d_ = Board::u64_dflip(d);

        let a = (a >> 8) & 0x00f0_00f0_00f0_00f0
            | (a << 8) & 0xf000_f000_f000_f000
            | a & 0x0f0f_0f0f_0f0f_0f0f;
        let d = (d >> 8) & 0x00f0_00f0_00f0_00f0
            | (d << 8) & 0xf000_f000_f000_f000
            | d & 0x0f0f_0f0f_0f0f_0f0f;

        let bboard = a & 0x0011_1111_1111_0011 | ((d & 0x0011_1111_1111_0011) << 1);
        let mut idxs: [usize; 10] = [0; 10];
        for i in 0..4 {
            let bboard =
                (a >> i) & 0x0011_1111_1111_0011 | (((d >> i) & 0x0011_1111_1111_0011) << 1);
            idxs[i] = ((bboard * Self::BIGH_MAGIC) >> 48) as usize;
        }
        for i in 0..4 {
            let bboard =
                (a_ >> i) & 0x0011_1111_1111_0011 | (((d_ >> i) & 0x0011_1111_1111_0011) << 1);
            idxs[i + 4] = ((bboard * Self::BIGH_MAGIC) >> 48) as usize;
        }
        let bboard = (a & 0x0080_0080_0080_0080) >> 3
            | (a & 0x0400_0400_0400_0400) >> 2
            | (a & 0x2000_2000_2000_2000) >> 1
            | (a & 0x0001_0001_0001_0001)
            | (d & 0x0080_0080_0080_0080) >> 2
            | (d & 0x0400_0400_0400_0400) >> 1
            | (d & 0x2000_2000_2000_2000)
            | (d & 0x0001_0001_0001_0001) << 1;
        idxs[8] = ((bboard * Self::BIGH_MAGIC) >> 48) as usize;
        let (a, d) = (Board::u64_hflip(a), Board::u64_hflip(d));
        let bboard = (a & 0x0080_0080_0080_0080) >> 3
            | (a & 0x0400_0400_0400_0400) >> 2
            | (a & 0x2000_2000_2000_2000) >> 1
            | (a & 0x0001_0001_0001_0001)
            | (d & 0x0080_0080_0080_0080) >> 2
            | (d & 0x0400_0400_0400_0400) >> 1
            | (d & 0x2000_2000_2000_2000)
            | (d & 0x0001_0001_0001_0001) << 1;
        idxs[9] = ((bboard * Self::BIGH_MAGIC) >> 48) as usize;

        return idxs;
    }

    fn get_piller3x4_idx(a: u64, d: u64) -> [usize; 10] {
        let mut ans = [0; 10];
        for i in 0..4 {
            let bboard;
            if i == 0 {
                bboard = a & 0x1111_1111_1111_1111 | ((d & 0x1111_1111_1111_1111) << 1);
            } else {
                bboard =
                    (a >> i) & 0x1111_1111_1111_1111 | ((d >> i) & 0x1111_1111_1111_1111) >> (i - 1)
            }
            ans[i] = ((bboard * Self::PILLER3X4_MAGIC) >> 48) as usize;
        }
        let (a, d) = (Board::u64_dflip(a), Board::u64_dflip(a));
        for i in 0..4 {
            let bboard: u64;
            if i == 0 {
                bboard = a & 0x1111_1111_1111_1111 | ((d & 0x1111_1111_1111_1111) << 1);
            } else {
                bboard =
                    (a >> i) & 0x1111_1111_1111_1111 | ((d >> i) & 0x1111_1111_1111_1111) >> (i - 1)
            }
            ans[i + 4] = ((bboard * Self::PILLER3X4_MAGIC) >> 48) as usize;
        }
        let bboard: u64 = (a & 0x0080_0080_0080_0080) >> 3
            | (a & 0x0400_0400_0400_0400) >> 2
            | (a & 0x2000_2000_2000_2000) >> 1
            | (a & 0x0001_0001_0001_0001)
            | (d & 0x0080_0080_0080_0080) >> 2
            | (d & 0x0400_0400_0400_0400) >> 1
            | (d & 0x2000_2000_2000_2000)
            | (d & 0x0001_0001_0001_0001) << 1;
        ans[8] = ((bboard * Self::PILLER3X4_MAGIC) >> 48) as usize;
        let (a, d) = (Board::u64_hflip(a), Board::u64_hflip(d));
        let bboard = (a & 0x0080_0080_0080_0080) >> 3
            | (a & 0x0400_0400_0400_0400) >> 2
            | (a & 0x2000_2000_2000_2000) >> 1
            | (a & 0x0001_0001_0001_0001)
            | (d & 0x0080_0080_0080_0080) >> 2
            | (d & 0x0400_0400_0400_0400) >> 1
            | (d & 0x2000_2000_2000_2000)
            | (d & 0x0001_0001_0001_0001) << 1;
        ans[9] = ((bboard * Self::PILLER3X4_MAGIC) >> 48) as usize;

        return ans;
    }

    fn xflip(a: u64) -> u64 {
        return (a >> 3) & 0x1111_1111_1111_1111
            | (a >> 1) & 0x2222_2222_2222_2222
            | (a << 1) & 0x4444_4444_4444_4444
            | (a << 3) & 0x8888_8888_8888_8888;
    }

    fn yflip(a: u64) -> u64 {
        return (a >> 12) & 0x000f_000f_000f_000f
            | (a >> 4) & 0x00f0_00f0_00f0_00f0
            | (a << 4) & 0x0f00_0f00_0f00_0f00
            | (a << 12) & 0xf000_f000_f000_f000;
    }

    fn get_bottom_corner_idx(a: u64, d: u64) -> [usize; 4] {
        let mut idxs = [0; 4];
        idxs[0] = Self::_get_bottom_corner_idx(a, d);
        let (a, d) = (Self::xflip(a), Self::xflip(d));
        idxs[1] = Self::_get_bottom_corner_idx(a, d);
        let (a, d) = (Self::yflip(a), Self::yflip(d));
        idxs[2] = Self::_get_bottom_corner_idx(a, d);
        let (a, d) = (Self::xflip(a), Self::xflip(d));
        idxs[3] = Self::_get_bottom_corner_idx(a, d);
        return idxs;
    }
    fn _get_bottom_corner_idx(a: u64, b: u64) -> usize {
        let mut k = 0;
        k = 3 * k + (a & 1) + (b & 1) << 1;
        k = 3 * k + (a >> 1) & 1 + ((b >> 1) & 1) << 1;
        k = 3 * k + (a >> 2) & 1 + ((b >> 2) & 1) << 1;
        k = 3 * k + (a >> 3) & 1 + ((b >> 3) & 1) << 1;
        k = 3 * k + (a >> 4) & 1 + ((b >> 4) & 1) << 1;
        k = 3 * k + (a >> 5) & 1 + ((b >> 5) & 1) << 1;
        k = 3 * k + (a >> 8) & 1 + ((b >> 8) & 1) << 1;
        k = 3 * k + (a >> 10) & 1 + ((b >> 10) & 1) << 1;
        k = 3 * k + (a >> 12) & 1 + ((b >> 12) & 1) << 1;
        k = 3 * k + (a >> 15) & 1 + ((b >> 15) & 1) << 1;
        return k as usize;
    }

    pub fn new() -> Self {
        // let big_h = vec![0.0; Self::BIGH_SIZE];
        // let piller = vec![0.0; Self::PILLER3X4_SIZE];
        // let bottom_corner = vec![0.0; Self::BOTTOM_CORNER_SIZE];
        let p4x4 = vec![0.0; 31 * 31 * 31 * 31];
        return SimplePatternEvaluator {
            // big_h: big_h,
            // piller_3x4: piller,
            // bottom_corner: bottom_corner,
            p4x4: p4x4,
        };
    }

    pub fn evaluate_board(&self, b: &Board) -> f32 {
        let mut val = 0.0;
        let (att, def) = b.get_att_def();
        // let idxs = Self::get_bigh_idx(att, def);
        // for idx in idxs {
        //     val += self.big_h[idx];
        // }
        // let idxs = Self::get_piller3x4_idx(att, def);
        // for idx in idxs {
        //     val += self.piller_3x4[idx];
        // }
        // let idxs = Self::get_bottom_corner_idx(att, def);
        // for idx in idxs {
        //     val += self.bottom_corner[idx];
        // }
        let idxs = Self::get_piller4x4_idx(att, def);
        for idx in idxs {
            val += self.p4x4[idx];
        }

        return 1.0 / (1.0 + (-val).exp());
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
        let mut src: SimplePatternEvaluator = serde_json::from_str(&data_str)?;

        std::mem::swap(self, &mut src);

        Ok(())
    }
}

#[derive(Clone)]
pub struct TrainableSPE {
    main: SimplePatternEvaluator,
    v: SimplePatternEvaluator,
    m: SimplePatternEvaluator,
    lr: f32,
}

impl TrainableSPE {
    pub fn new(lr: f32) -> Self {
        TrainableSPE {
            main: SimplePatternEvaluator::new(),
            v: SimplePatternEvaluator::new(),
            m: SimplePatternEvaluator::new(),
            lr: lr,
        }
    }

    pub fn from(e: SimplePatternEvaluator, lr: f32) -> Self {
        TrainableSPE {
            main: e,
            v: SimplePatternEvaluator::new(),
            m: SimplePatternEvaluator::new(),
            lr: lr,
        }
    }
}

impl Trainable for TrainableSPE {
    fn update(&mut self, b: &Board, delta: f32) {
        let (att, def) = b.get_att_def();
        let bigh_idxs = SimplePatternEvaluator::get_bigh_idx(att, def);
        let pi3x4_idxs = SimplePatternEvaluator::get_piller3x4_idx(att, def);
        let bcorner_idxs = SimplePatternEvaluator::get_bottom_corner_idx(att, def);
        let p4x4_idxs = SimplePatternEvaluator::get_piller4x4_idx(att, def);
        let val = self.main.evaluate_board(b);

        let dv = val * (1.0 - val);
        let delta = self.lr * delta * dv;
        // for i in bigh_idxs {
        //     self.main.big_h[i] += delta;
        // }
        // for i in pi3x4_idxs {
        //     self.main.piller_3x4[i] += delta;
        // }
        // for i in bcorner_idxs {
        //     self.main.bottom_corner[i] += delta;
        // }
        for i in p4x4_idxs {
            self.main.p4x4[i] += delta;
        }
    }

    fn get_val(&self, b: &Board) -> f32 {
        self.main.evaluate_board(b)
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

impl EvaluatorF for TrainableSPE {
    fn eval_func_f32(&self, b: &Board) -> f32 {
        return self.main.evaluate_board(b).clamp(0.0, 1.0);
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct BucketLineEvaluator {
    evals: Vec<SimplLineEvaluator>,
}

impl BucketLineEvaluator {
    pub fn new() -> Self {
        let mut v = Vec::new();
        for _ in 0..64 {
            v.push(SimplLineEvaluator::new());
        }
        return BucketLineEvaluator { evals: v };
    }

    pub fn from(simpl: SimplLineEvaluator) -> Self {
        return BucketLineEvaluator {
            evals: vec![simpl; 64],
        };
    }

    pub fn get_counts(
        &self,
        b: &Board,
    ) -> (
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    ) {
        let (att, def) = b.get_att_def();
        let (a1, a2, a3, _, _, _) = LineEvaluator::analyze_board(att, def);
        let (d1, d2, d3, _, _, _) = LineEvaluator::analyze_board(def, att);
        let stone = att | def;
        let ground = !stone & (stone << 16 | 0xffff);
        let float = !stone ^ ground;
        let a1_float = acum_mask_bundle(apply_mask_bundle(a1, float)) as usize;
        let a1_ground = acum_mask_bundle(apply_mask_bundle(a1, ground)) as usize;
        let a2_float = acum_mask_bundle(apply_mask_bundle(a2, float)) as usize;
        let a2_ground = acum_mask_bundle(apply_mask_bundle(a2, ground)) as usize;
        let a3_float = (acum_or(a3) & float).count_ones() as usize;
        let a3_ground = (acum_or(a3) & ground).count_ones() as usize;
        let d1_float = acum_mask_bundle(apply_mask_bundle(d1, float)) as usize;
        let d1_ground = acum_mask_bundle(apply_mask_bundle(d1, ground)) as usize;
        let d2_float = acum_mask_bundle(apply_mask_bundle(d2, float)) as usize;
        let d2_ground = acum_mask_bundle(apply_mask_bundle(d2, ground)) as usize;
        let d3_float = (acum_or(d3) & float).count_ones() as usize;
        let d3_ground = (acum_or(d3) & ground).count_ones() as usize;

        let l3_mask = acum_or(d3) | acum_or(a3);
        let trap_3_num = (l3_mask & (!l3_mask << 16) & 0x0000_ffff_0000_0000).count_ones() as usize;

        return (
            a1_float, a2_float, a3_float, a1_ground, a2_ground, a3_ground, d1_float, d2_float,
            d3_float, d1_ground, d2_ground, d3_ground, trap_3_num,
        );
    }

    pub fn evaluate_board(&self, b: &Board) -> f32 {
        let (af1, af2, af3, ag1, ag2, ag3, df1, df2, df3, dg1, dg2, dg3, tn3) = self.get_counts(&b);
        let (att, def) = b.get_att_def();
        let num = att.count_ones() + def.count_ones();
        let is_black = num % 2 == 0;
        let mut val = 0.0;

        let eval = &self.evals[num as usize];

        if is_black {
            val += eval.wt3nb[tn3];
        } else {
            val += eval.wt3nw[tn3];
        }
        val += eval.wfl1[af1 * WFL1_WIDTH + df1]
            + eval.wfl2[af2 * WL2_WIDTH + df2]
            + eval.wfl3[af3 * WFL3_WIDTH + df3]
            + eval.wgl1[ag1 * WGL1_WIDTH + dg1]
            + eval.wgl2[ag2 * WL2_WIDTH + dg2]
            + eval.wgl3[ag3 * WGL3_WIDTH + dg3]
            + eval.bias;
        return 1.0 / (1.0 + (-val).exp());
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
        let mut src: BucketLineEvaluator = serde_json::from_str(&data_str)?;

        std::mem::swap(self, &mut src);

        Ok(())
    }
}

impl EvaluatorF for BucketLineEvaluator {
    fn eval_func_f32(&self, b: &Board) -> f32 {
        return self.evaluate_board(b);
    }
}

#[derive(Clone)]
pub struct TrainableBLE {
    main: BucketLineEvaluator,
    v: BucketLineEvaluator,
    m: BucketLineEvaluator,
    lr: f32,
}

impl TrainableBLE {
    pub fn new(lr: f32) -> Self {
        TrainableBLE {
            main: BucketLineEvaluator::new(),
            v: BucketLineEvaluator::new(),
            m: BucketLineEvaluator::new(),
            lr: lr,
        }
    }

    pub fn from(e: BucketLineEvaluator, lr: f32) -> Self {
        TrainableBLE {
            main: e,
            v: BucketLineEvaluator::new(),
            m: BucketLineEvaluator::new(),
            lr: lr,
        }
    }
}

impl Trainable for TrainableBLE {
    fn update(&mut self, b: &Board, delta: f32) {
        let (a1, a2, a3, a1_, a2_, a3_, d1, d2, d3, d1_, d2_, d3_, trap_3_num) =
            self.main.get_counts(b);
        let num = (b.black.count_ones() + b.white.count_ones()) as usize;
        let is_black = num % 2 == 0;
        // とりあえずsgd
        let val = self.main.evaluate_board(b);
        let dv = val * (1.0 - val);
        let delta = self.lr * delta * dv;
        self.main.evals[num].wfl1[a1 * WFL1_WIDTH + d1] += delta;
        self.main.evals[num].wfl2[a2 * WL2_WIDTH + d2] += delta;
        self.main.evals[num].wfl3[a3 * WFL3_WIDTH + d3] += delta;
        self.main.evals[num].wgl1[a1_ * WGL1_WIDTH + d1_] += delta;
        self.main.evals[num].wgl2[a2_ * WL2_WIDTH + d2_] += delta;
        self.main.evals[num].wgl3[a3_ * WGL3_WIDTH + d3_] += delta;
        if is_black {
            self.main.evals[num].wt3nb[trap_3_num] += delta;
        } else {
            self.main.evals[num].wt3nw[trap_3_num] += delta;
        }
        self.main.evals[num].bias += delta;
    }

    fn get_val(&self, b: &Board) -> f32 {
        self.main.evaluate_board(b)
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

impl EvaluatorF for TrainableBLE {
    fn eval_func_f32(&self, b: &Board) -> f32 {
        return self.main.evaluate_board(b).clamp(0.0, 1.0);
    }
}
