#[cfg(target_arch = "x86")]
use std::arch::x86::*;
use std::arch::x86_64::__m256i;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::board::{pprint_board, Board};

use super::{
    acum_mask_bundle, acum_or, apply_mask_bundle, EvaluatorF, LineEvaluator, LineMaskBundle,
    Trainable,
};
use anyhow::{Ok, Result};
use serde::{Deserialize, Serialize};

const V0MASK: u64 = 0x9009;
const E0MASK: u64 = 0x6996;
const S0MASK: u64 = 0x0660;
const V1MASK: u64 = V0MASK << 16;
const E1MASK: u64 = E0MASK << 16;
const S1MASK: u64 = S0MASK << 16;
const V2MASK: u64 = V1MASK << 16;
const E2MASK: u64 = E1MASK << 16;
const S2MASK: u64 = S1MASK << 16;
const V3MASK: u64 = V2MASK << 16;
const E3MASK: u64 = E2MASK << 16;
const S3MASK: u64 = S2MASK << 16;
const MASKS: [u64; 16] = [
    V0MASK,
    V1MASK,
    V2MASK,
    V3MASK,
    S0MASK,
    S1MASK,
    S2MASK,
    S3MASK,
    E0MASK,
    E1MASK,
    E2MASK,
    E3MASK,
    0xffff,
    0xffff << 16,
    0xffff << 32,
    0xffff << 48,
];
const VMASK: u64 = V0MASK | V1MASK | V2MASK | V3MASK;
const SMASK: u64 = S0MASK | S1MASK | S2MASK | S3MASK;
const EMASK: u64 = E0MASK | E1MASK | E2MASK | E3MASK;
const VMASKI: i64 = (V0MASK | V1MASK | V2MASK | V3MASK) as i64;
const SMASKI: i64 = (S0MASK | S1MASK | S2MASK | S3MASK) as i64;
const EMASKI: i64 = (E0MASK | E1MASK | E2MASK | E3MASK) as i64;

const POPCOUNT_MASK0_U64: [u64; 4] = [
    0x5555555555555555,
    0x5555555555555555,
    0x5555555555555555,
    0x5555555555555555,
];
const POPCOUNT_MASK0: __m256i = unsafe { std::mem::transmute(POPCOUNT_MASK0_U64) };
const POPCOUNT_MASK1_U64: [u64; 4] = [
    0x3333333333333333,
    0x3333333333333333,
    0x3333333333333333,
    0x3333333333333333,
];
const POPCOUNT_MASK1: __m256i = unsafe { std::mem::transmute(POPCOUNT_MASK1_U64) };
const POPCOUNT_MASK2_U64: [u64; 4] = [
    0x0f0f0f0f0f0f0f0f,
    0x0f0f0f0f0f0f0f0f,
    0x0f0f0f0f0f0f0f0f,
    0x0f0f0f0f0f0f0f0f,
];
const POPCOUNT_MASK2: __m256i = unsafe { std::mem::transmute(POPCOUNT_MASK2_U64) };
const POPCOUNT_MASK3_U64: [u64; 4] = [
    0x00ff00ff00ff00ff,
    0x00ff00ff00ff00ff,
    0x00ff00ff00ff00ff,
    0x00ff00ff00ff00ff,
];
const POPCOUNT_MASK3: __m256i = unsafe { std::mem::transmute(POPCOUNT_MASK3_U64) };
const MASK_ALL_U64: [i64; 4] = [-1, -1, -1, -1];
const MASK_ALL: __m256i = unsafe { std::mem::transmute(MASK_ALL_U64) };

fn pprint_weight(v: &Vec<f32>, width: usize, float_size: usize, f: &mut std::fmt::Formatter<'_>) {
    for i in 0..=width {
        for j in 0..(width - i) {
            let idx = j + (i + j) * (i + j + 1) / 2;
            let front = float_size + 3;
            let _ = write!(f, "{:>front$.float_size$}, ", v[idx]);
        }
        let _ = writeln!(f, "");
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PositionMaskEvaluator {
    pub wv0: Vec<f32>,
    pub we0: Vec<f32>,
    pub ws0: Vec<f32>,
    pub wv1: Vec<f32>,
    pub we1: Vec<f32>,
    pub ws1: Vec<f32>,
    pub wv2: Vec<f32>,
    pub we2: Vec<f32>,
    pub ws2: Vec<f32>,
    pub wv3: Vec<f32>,
    pub we3: Vec<f32>,
    pub ws3: Vec<f32>,
    pub w0: Vec<f32>,
    pub w1: Vec<f32>,
    pub w2: Vec<f32>,
    pub w3: Vec<f32>,
    pub bias: f32,
}

impl PositionMaskEvaluator {
    pub fn new() -> Self {
        return PositionMaskEvaluator {
            wv0: vec![0.0; 45],
            we0: vec![0.0; 45],
            ws0: vec![0.0; 45],
            wv1: vec![0.0; 45],
            we1: vec![0.0; 45],
            ws1: vec![0.0; 45],
            wv2: vec![0.0; 45],
            we2: vec![0.0; 45],
            ws2: vec![0.0; 45],
            wv3: vec![0.0; 45],
            we3: vec![0.0; 45],
            ws3: vec![0.0; 45],
            w0: vec![0.0; 153],
            w1: vec![0.0; 153],
            w2: vec![0.0; 153],
            w3: vec![0.0; 153],
            bias: 0.0,
        };
    }

    const fn pop_count_16(bit: u64) -> (usize, usize, usize, usize) {
        let bit = bit - ((bit >> 1) & 0x5555555555555555);
        let bit = (bit & 0x3333333333333333) + ((bit >> 2) & 0x3333333333333333);
        let bit = (bit + (bit >> 4));
        let bit = bit + (bit >> 8);
        return (
            (bit & 0xf) as usize,
            ((bit >> 16) & 0xf) as usize,
            ((bit >> 32) & 0xf) as usize,
            ((bit >> 48) & 0xf) as usize,
        );
    }

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    ))]
    #[inline]
    fn popcount_16_si256(a: __m256i) -> __m256i {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;
        let a = unsafe {
            let a = _mm256_sub_epi16(
                a,
                _mm256_and_si256(_mm256_srli_epi16::<1>(a), POPCOUNT_MASK0),
            );
            let a = _mm256_add_epi16(
                _mm256_and_si256(a, POPCOUNT_MASK1),
                _mm256_and_si256(_mm256_srli_epi16::<2>(a), POPCOUNT_MASK1),
            );
            let a = _mm256_and_si256(
                _mm256_add_epi16(a, _mm256_srli_epi16::<4>(a)),
                POPCOUNT_MASK2,
            );
            _mm256_and_si256(
                _mm256_add_epi16(a, _mm256_srli_epi16::<8>(a)),
                POPCOUNT_MASK3,
            )
        };
        return a;
    }

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    ))]
    pub fn get_counts(b: &Board) -> [usize; 16] {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let (att, def) = b.get_att_def();
        let stone = (att | def) as i64;
        let idxs = unsafe {
            // let ssss = _mm256_set_epi64x(stone, stone, stone, stone);
            let ssss = std::mem::transmute::<[i64; 4], __m256i>([stone, stone, stone, stone]);
            // let dddd = _mm256_set_epi64x(def as i64, def as i64, def as i64, def as i64);
            let dddd = std::mem::transmute::<[u64; 4], __m256i>([def, def, def, def]);
            let vsen_mask = _mm256_set_epi64x(-1, EMASKI, SMASKI, VMASKI);

            let maskall = _mm256_set_epi64x(-1, -1, -1, -1);

            let s = _mm256_and_si256(ssss, vsen_mask);
            let s = PositionMaskEvaluator::popcount_16_si256(s);

            let s = _mm256_srli_epi16::<1>(_mm256_add_epi16(_mm256_mullo_epi16(s, s), s));

            let d = _mm256_and_si256(dddd, vsen_mask);
            let d = PositionMaskEvaluator::popcount_16_si256(d);

            let sd = _mm256_add_epi16(s, d);
            let sd_half_low = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<0>(sd));
            let sd_half_high = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(sd));

            let mut vs: [i32; 16] = [0; 16];
            let mut ptr: *mut i32 = &mut vs as *mut i32;
            _mm256_maskstore_epi32(ptr, MASK_ALL, sd_half_low);
            _mm256_maskstore_epi32(ptr.add(8), MASK_ALL, sd_half_high);

            vs
        };
        let mut ans: [usize; 16] = [0; 16];
        for i in 0..16 {
            ans[i] = idxs[i] as usize;
        }

        return ans;
    }

    #[cfg(not(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    )))]
    pub fn get_counts(b: &Board) -> [usize; 16] {
        let (att, def) = b.get_att_def();
        let stone = att | def;
        let mut ans = [0; 16];

        for i in 0..16 {
            ans[i] = {
                let (ij, j) = (
                    (stone & MASKS[i]).count_ones(),
                    (def & MASKS[i]).count_ones(),
                );
                (j + ij * (ij + 1) / 2) as usize
            }
        }

        return ans;
    }

    pub fn evaluate_board(&self, b: &Board) -> f32 {
        let idxs = PositionMaskEvaluator::get_counts(&b);

        let val = self.bias
            + self.wv0[idxs[0]]
            + self.wv1[idxs[1]]
            + self.wv2[idxs[2]]
            + self.wv3[idxs[3]]
            + self.ws0[idxs[4]]
            + self.ws1[idxs[5]]
            + self.ws2[idxs[6]]
            + self.ws3[idxs[7]]
            + self.we0[idxs[8]]
            + self.we1[idxs[9]]
            + self.we2[idxs[10]]
            + self.we3[idxs[11]]
            + self.w0[idxs[12]]
            + self.w1[idxs[13]]
            + self.w2[idxs[14]]
            + self.w3[idxs[15]];
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
        let mut src: PositionMaskEvaluator = serde_json::from_str(&data_str)?;

        std::mem::swap(self, &mut src);

        Ok(())
    }

    pub fn move_average(&mut self) {
        let we0 = self.we0[0];
        self.bias += we0;
        for i in 0..self.we0.len() {
            self.we0[i] -= we0;
        }
        let we1 = self.we1[0];
        self.bias += we1;
        for i in 0..self.we1.len() {
            self.we1[i] -= we1;
        }
        let we2 = self.we2[0];
        self.bias += we2;
        for i in 0..self.we2.len() {
            self.we2[i] -= we2;
        }
        let we3 = self.we3[0];
        self.bias += we3;
        for i in 0..self.we3.len() {
            self.we3[i] -= we3;
        }

        let wv0 = self.wv0[0];
        self.bias += wv0;
        for i in 0..self.wv0.len() {
            self.wv0[i] -= wv0;
        }
        let wv1 = self.wv1[0];
        self.bias += wv1;
        for i in 0..self.wv1.len() {
            self.wv1[i] -= wv1;
        }
        let wv2 = self.wv2[0];
        self.bias += wv2;
        for i in 0..self.wv2.len() {
            self.wv2[i] -= wv2;
        }
        let wv3 = self.wv3[0];
        self.bias += wv3;
        for i in 0..self.wv3.len() {
            self.wv3[i] -= wv3;
        }

        let ws0 = self.ws0[0];
        self.bias += ws0;
        for i in 0..self.ws0.len() {
            self.ws0[i] -= ws0;
        }
        let ws1 = self.ws1[0];
        self.bias += ws1;
        for i in 0..self.ws1.len() {
            self.ws1[i] -= ws1;
        }
        let ws2 = self.ws2[0];
        self.bias += ws2;
        for i in 0..self.ws2.len() {
            self.ws2[i] -= ws2;
        }
        let ws3 = self.ws3[0];
        self.bias += ws3;
        for i in 0..self.ws3.len() {
            self.ws3[i] -= ws3;
        }

        let w0 = self.w0[0];
        self.bias += w0;
        for i in 0..self.w0.len() {
            self.w0[i] -= w0;
        }
        let w1 = self.w1[0];
        self.bias += w1;
        for i in 0..self.w1.len() {
            self.w1[i] -= w1;
        }
        let w2 = self.w2[0];
        self.bias += w2;
        for i in 0..self.w2.len() {
            self.w2[i] -= w2;
        }
        let w3 = self.w3[0];
        self.bias += w3;
        for i in 0..self.w3.len() {
            self.w3[i] -= w3;
        }
    }
}

impl EvaluatorF for PositionMaskEvaluator {
    fn eval_func_f32(&self, b: &Board) -> f32 {
        return self.evaluate_board(b);
    }
}

impl std::fmt::Debug for PositionMaskEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let float_size = 2;
        let _ = writeln!(f, "bias:\n{:.3}", self.bias);
        let _ = writeln!(f, "wv0:");
        pprint_weight(&self.wv0, 4, float_size, f);
        let _ = writeln!(f, "wv1:");
        pprint_weight(&self.wv1, 4, float_size, f);
        let _ = writeln!(f, "wv2:");
        pprint_weight(&self.wv2, 4, float_size, f);
        let _ = writeln!(f, "wv3:");
        pprint_weight(&self.wv3, 4, float_size, f);

        let _ = writeln!(f, "we0:");
        pprint_weight(&self.we0, 8, float_size, f);
        let _ = writeln!(f, "we1:");
        pprint_weight(&self.we1, 8, float_size, f);
        let _ = writeln!(f, "we2:");
        pprint_weight(&self.we2, 8, float_size, f);
        let _ = writeln!(f, "we3:");
        pprint_weight(&self.we3, 8, float_size, f);

        let _ = writeln!(f, "ws0:");
        pprint_weight(&self.ws0, 4, float_size, f);
        let _ = writeln!(f, "ws1:");
        pprint_weight(&self.ws1, 4, float_size, f);
        let _ = writeln!(f, "ws2:");
        pprint_weight(&self.ws2, 4, float_size, f);
        let _ = writeln!(f, "ws3:");
        pprint_weight(&self.ws3, 4, float_size, f);

        let _ = writeln!(f, "w0:");
        pprint_weight(&self.w0, 16, float_size, f);
        let _ = writeln!(f, "w1:");
        pprint_weight(&self.w1, 16, float_size, f);
        let _ = writeln!(f, "w2:");
        pprint_weight(&self.w2, 16, float_size, f);
        let _ = writeln!(f, "w3:");
        pprint_weight(&self.w3, 16, float_size, f);

        Result::Ok(())
    }
}

#[derive(Clone)]
pub struct TrainablePME {
    main: PositionMaskEvaluator,
    v: PositionMaskEvaluator,
    m: PositionMaskEvaluator,
    lr: f32,
}

impl TrainablePME {
    pub fn new(lr: f32) -> Self {
        TrainablePME {
            main: PositionMaskEvaluator::new(),
            v: PositionMaskEvaluator::new(),
            m: PositionMaskEvaluator::new(),
            lr: lr,
        }
    }

    pub fn from(e: PositionMaskEvaluator, lr: f32) -> Self {
        TrainablePME {
            main: e,
            v: PositionMaskEvaluator::new(),
            m: PositionMaskEvaluator::new(),
            lr: lr,
        }
    }
}

impl Trainable for TrainablePME {
    fn update(&mut self, b: &Board, delta: f32) {
        let idxs = PositionMaskEvaluator::get_counts(b);
        // とりあえずsgd
        let val = self.main.evaluate_board(b);
        let dv = val * (1.0 - val);
        let delta = self.lr * delta * dv;
        self.main.wv0[idxs[0]] += delta;
        self.main.wv1[idxs[1]] += delta;
        self.main.wv2[idxs[2]] += delta;
        self.main.wv3[idxs[3]] += delta;
        self.main.ws0[idxs[4]] += delta;
        self.main.ws1[idxs[5]] += delta;
        self.main.ws2[idxs[6]] += delta;
        self.main.ws3[idxs[7]] += delta;
        self.main.we0[idxs[8]] += delta;
        self.main.we1[idxs[9]] += delta;
        self.main.we2[idxs[10]] += delta;
        self.main.we3[idxs[11]] += delta;
        self.main.w0[idxs[12]] += delta;
        self.main.w1[idxs[13]] += delta;
        self.main.w2[idxs[14]] += delta;
        self.main.w3[idxs[15]] += delta;
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
    fn train(&mut self) {
        self.main.move_average();
    }
}

impl EvaluatorF for TrainablePME {
    fn eval_func_f32(&self, b: &Board) -> f32 {
        return self.main.evaluate_board(b).clamp(0.0, 1.0);
    }
}
