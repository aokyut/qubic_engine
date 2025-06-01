use crate::{
    board::Board,
    ml::{funcs::ReLU, params::Linear, xiver_vec, Graph, Node, Tensor},
};
use anyhow::Result;
use ndarray::{
    self as nd, array, concatenate, s, Array, Array0, Array2, ArrayViewMut1, Axis, Dim, Ix,
};
use rand_distr::Distribution;
use serde::{Deserialize, Serialize};

use crate::ml;

use super::{
    line::{self, SimplLineEvaluator},
    EvaluatorF, LineEvaluator, LineMaskBundle, Trainable,
};

const D: usize = 2;
const INPUT: usize = D * 7 + 1;
const SECOND: usize = 16;

const WFL3_WIDTH: usize = 32;
const WGL3_WIDTH: usize = 16;
const WFL2_WIDTH: usize = 64;
const WGL2_WIDTH: usize = 64;
const WGL1_WIDTH: usize = 96;
const WFL1_WIDTH: usize = 106;

#[derive(Clone)]
pub struct NNLineEvaluator {
    pub wfl3: Array2<f32>,
    pub wgl3: Array2<f32>,
    pub wfl2: Array2<f32>,
    pub wgl2: Array2<f32>,
    pub wfl1: Array2<f32>,
    pub wgl1: Array2<f32>,
    pub wt3nw: Array2<f32>,
    pub wt3nb: Array2<f32>,
    pub wmat1: Array2<f32>,
    pub wmat2: Array2<f32>,
    pub bias1: Array2<f32>,
    pub bias2: Array2<f32>,
    pub bias: Array2<f32>,
}

impl NNLineEvaluator {
    pub fn new() -> Self {
        let mut wmat1 = ml::xiver_vec(SECOND, INPUT * SECOND);
        let mut wmat2 = ml::xiver_vec(1, SECOND);

        return NNLineEvaluator {
            wfl3: unsafe {
                Array::from_shape_vec_unchecked([32 * 32, D], ml::xiver_vec(1, 32 * 32 * D))
            },
            wgl3: unsafe {
                Array::from_shape_vec_unchecked([16 * 16, D], ml::xiver_vec(1, 16 * 16 * D))
            },
            wfl2: unsafe {
                Array::from_shape_vec_unchecked([64 * 64, D], ml::xiver_vec(1, 64 * 64 * D))
            },
            wgl2: unsafe {
                Array::from_shape_vec_unchecked([64 * 64, D], ml::xiver_vec(1, 64 * 64 * D))
            },
            wfl1: unsafe {
                Array::from_shape_vec_unchecked([96 * 96, D], ml::xiver_vec(1, 96 * 96 * D))
            },
            wgl1: unsafe {
                Array::from_shape_vec_unchecked([106 * 106, D], ml::xiver_vec(1, 106 * 106 * D))
            },
            wt3nw: unsafe { Array::from_shape_vec_unchecked([12, D], ml::xiver_vec(1, 12 * D)) },
            wt3nb: unsafe { Array::from_shape_vec_unchecked([12, D], ml::xiver_vec(1, 12 * D)) },
            wmat1: unsafe {
                Array::from_shape_vec_unchecked(
                    [SECOND, INPUT],
                    ml::xiver_vec(INPUT, SECOND * INPUT),
                )
            },
            wmat2: unsafe {
                Array::from_shape_vec_unchecked([1, SECOND], ml::xiver_vec(SECOND, SECOND))
            },
            bias1: unsafe {
                Array::from_shape_vec_unchecked([SECOND, 1], ml::xiver_vec(1, SECOND))
            },
            bias2: array![[0.0]],
            bias: array![[0.0]],
        };
    }

    pub fn evaluate_board(&self, b: &Board) -> f32 {
        let (af1, af2, af3, ag1, ag2, ag3, df1, df2, df3, dg1, dg2, dg3, tn3) =
            line::SimplLineEvaluator::get_counts(&b);
        let (att, def) = b.get_att_def();
        let is_black = (att.count_ones() + def.count_ones()) % 2 == 0;

        let wfl1 = self.wfl1.slice(s![af1 * WFL1_WIDTH + df1, ..]);
        let wfl2 = self.wfl2.slice(s![af2 * WFL2_WIDTH + df2, ..]);
        let wfl3 = self.wfl3.slice(s![af3 * WFL3_WIDTH + df3, ..]);
        let wgl1 = self.wgl1.slice(s![ag1 * WGL1_WIDTH + dg1, ..]);
        let wgl2 = self.wgl2.slice(s![ag2 * WGL2_WIDTH + dg2, ..]);
        let wgl3 = self.wgl3.slice(s![ag3 * WGL3_WIDTH + dg3, ..]);

        let wt3;
        if is_black {
            wt3 = self.wt3nb.slice(s![tn3, ..]);
        } else {
            wt3 = self.wt3nw.slice(s![tn3, ..]);
        }

        // [1, input_size]
        let input = concatenate(
            Axis(0),
            &[
                wfl1,
                wfl2,
                wfl3,
                wgl1,
                wgl2,
                wgl3,
                wt3,
                self.bias.slice(s![0, ..]),
            ],
        )
        .unwrap();
        assert_eq!(input.shape(), &[INPUT]);

        let x = self.wmat1.dot(&input.view().t()) + self.bias1.view();
        let x = x.map(|t| t.max(0.0));
        let x = self.wmat2.dot(&x) + self.bias2.view();
        let val = x[[0, 0]];

        return 1.0 / (1.0 + (-val).exp());
    }

    pub fn save(&self, _: String) -> Result<()> {
        Ok(())
    }

    pub fn load(&mut self, _: String) -> Result<()> {
        Ok(())
    }
}

impl EvaluatorF for NNLineEvaluator {
    fn eval_func_f32(&self, b: &Board) -> f32 {
        return self.evaluate_board(b);
    }
}

#[derive(Clone)]
pub struct TrainableNLE {
    main: NNLineEvaluator,
    v: NNLineEvaluator,
    m: NNLineEvaluator,
    lr: f32,
}

impl TrainableNLE {
    pub fn new(lr: f32) -> Self {
        TrainableNLE {
            main: NNLineEvaluator::new(),
            v: NNLineEvaluator::new(),
            m: NNLineEvaluator::new(),
            lr: lr,
        }
    }

    pub fn from(e: NNLineEvaluator, lr: f32) -> Self {
        TrainableNLE {
            main: e,
            v: NNLineEvaluator::new(),
            m: NNLineEvaluator::new(),
            lr: lr,
        }
    }
}

impl Trainable for TrainableNLE {
    fn update(&mut self, b: &Board, delta: f32) {
        let (af1, af2, af3, ag1, ag2, ag3, df1, df2, df3, dg1, dg2, dg3, tn3) =
            SimplLineEvaluator::get_counts(b);
        let is_black = (b.black.count_ones() + b.white.count_ones()) % 2 == 0;
        // とりあえずsgd
        let val = self.main.evaluate_board(b);
        let dv = val * (1.0 - val);
        println!("val:{:#?}, delta:{}", val, delta);
        let delta = self.lr * delta * dv;
        // let delta = 0.0;

        let wfl1 = self.main.wfl1.slice(s![af1 * WFL1_WIDTH + df1, ..]);
        let wfl2 = self.main.wfl2.slice(s![af2 * WFL2_WIDTH + df2, ..]);
        let wfl3 = self.main.wfl3.slice(s![af3 * WFL3_WIDTH + df3, ..]);
        let wgl1 = self.main.wgl1.slice(s![ag1 * WGL1_WIDTH + dg1, ..]);
        let wgl2 = self.main.wgl2.slice(s![ag2 * WGL2_WIDTH + dg2, ..]);
        let wgl3 = self.main.wgl3.slice(s![ag3 * WGL3_WIDTH + dg3, ..]);

        let wt3;
        if is_black {
            wt3 = self.main.wt3nb.slice(s![tn3, ..]);
        } else {
            wt3 = self.main.wt3nw.slice(s![tn3, ..]);
        }

        // [1, input_size]
        let input = concatenate(
            Axis(0),
            &[
                wfl1,
                wfl2,
                wfl3,
                wgl1,
                wgl2,
                wgl3,
                wt3,
                self.main.bias.slice(s![0, ..]),
            ],
        )
        .unwrap();
        assert_eq!(input.shape(), &[INPUT]);

        let x1 = self.main.wmat1.dot(&input.view().t()); //+ self.main.bias1.view();
        let x2 = x1.map(|t| t.max(0.0));
        let x3 = self.main.wmat2.dot(&x2) + self.main.bias2.view();
        let val = x3[[0, 0]];

        println!("input:{:#?}", input.view());
        println!("x1:{:#?}", x1.view());
        println!("x2:{:#?}", x2.view());
        println!("x2_t:{:#?}", x2.view().insert_axis(Axis(0)));
        println!("x3:{:#?}", x3.view());
        println!("wmat2:{:#?}", self.main.wmat2.view());
        println!("wmat1:{:#?}", self.main.wmat1.view());
        println!("bias:{:#?}", self.main.bias2.view());
        self.main.bias2 = array![[self.main.bias2[[0, 0]] + delta]];
        println!("bias:{:#?}", self.main.bias2.view());
        // x3 = x3 * wm2
        // dx3 / dwm2 = out * in^T
        let dwm2 = delta * x3.dot(&x2.view().insert_axis(Axis(0)));
        // dx3 / dx2 = wm^T * out
        let dx2 = delta * self.main.wmat2.view().t().dot(&x3);
        self.main.wmat2 = self.main.wmat2.clone() + dwm2.view(); // apply
        println!("dwmat2:{:#?}", dwm2.view());
        let sign = x1.map(|t| t.signum() * 0.5 + 0.5).insert_axis(Axis(1));
        let dx1 = &dx2 * &sign;
        self.main.bias1 = self.main.bias1.clone() + dx1.view();
        println!("dx3:{}", delta);
        println!("dx2:{:#?}", dx2.view());
        println!("dx1:{:#?}", dx1.view());
        println!("dx1:{:#?}", dx1.view());
        let dwm1 = dx1.dot(&input.view().insert_axis(Axis(0)));
        println!("dwmat1:{:#?}", dwm1.view());
        let binding = self.main.wmat1.view().t().dot(&dx1);
        let di = binding.t();
        println!("di:{:#?}", di.view());
        self.main.wmat1 = self.main.wmat1.clone() + dwm1;

        let mut wfl1 = self.main.wfl1.slice_mut(s![af1 * WFL1_WIDTH + df1, ..]);
        wfl1.assign(&(&wfl1.view() + &di.slice(s![0, (0 * D)..(1 * D)]).view()));
        let mut wfl2 = self.main.wfl2.slice_mut(s![af2 * WFL2_WIDTH + df2, ..]);
        wfl2.assign(&(&wfl2.view() + &di.slice(s![0, (1 * D)..(2 * D)])));
        let mut wfl3 = self.main.wfl3.slice_mut(s![af3 * WFL3_WIDTH + df3, ..]);
        wfl3.assign(&(&wfl3.view() + &di.slice(s![0, (2 * D)..(3 * D)])));

        let mut wgl1 = self.main.wgl1.slice_mut(s![ag1 * WGL1_WIDTH + dg1, ..]);
        wgl1.assign(&(&wgl1.view() + &di.slice(s![0, (3 * D)..(4 * D)]).view()));
        let mut wgl2 = self.main.wgl2.slice_mut(s![ag2 * WGL2_WIDTH + dg2, ..]);
        wgl2.assign(&(&wgl2.view() + &di.slice(s![0, (4 * D)..(5 * D)])));
        let mut wgl3 = self.main.wgl3.slice_mut(s![ag3 * WGL3_WIDTH + dg3, ..]);
        wgl3.assign(&(&wgl3.view() + &di.slice(s![0, (5 * D)..(6 * D)])));

        if is_black {
            let mut wt3nb = self.main.wt3nb.slice_mut(s![tn3, ..]);
            wt3nb.assign(&(&wt3nb.view() + &di.slice(s![0, (6 * D)..(7 * D)])));
        } else {
            let mut wt3nw = self.main.wt3nw.slice_mut(s![tn3, ..]);
            wt3nw.assign(&(&wt3nw.view() + &di.slice(s![0, (6 * D)..(7 * D)])));
        }
        let mut bias = self.main.bias.slice_mut(s![0, ..]);
        bias.assign(&(&bias.view() + &di.slice(s![0, (7 * D)..(7 * D + 1)])));

        let input = concatenate(
            Axis(0),
            &[
                wfl1.view(),
                wfl2.view(),
                wfl3.view(),
                wgl1.view(),
                wgl2.view(),
                wgl3.view(),
                self.main.bias.slice(s![0, ..]),
            ],
        )
        .unwrap();

        println!("input_:{:#?}", input);
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

impl EvaluatorF for TrainableNLE {
    fn eval_func_f32(&self, b: &Board) -> f32 {
        return self.main.evaluate_board(b).clamp(0.0, 1.0);
    }
}

use std::arch::x86_64::*;
pub struct MMEvaluator {
    pub wfl3: Vec<__m256>,
    pub wfl2: Vec<__m256>,
    pub wfl1: Vec<__m256>,
    pub wgl3: Vec<__m256>,
    pub wgl2: Vec<__m256>,
    pub wgl1: Vec<__m256>,
    pub wt3nb: Vec<__m256>,
    pub wt3nw: Vec<__m256>,
    pub wcore: Vec<__m256>,
    pub wturn: Vec<__m256>,
    pub w_acum: __m256,
    pub bias: __m256,
    pub lbias: f32,
}

impl MMEvaluator {
    const D: usize = 8;
    const INPUT: usize = 7 * Self::D;
    const F3: usize = WFL3_WIDTH * WFL3_WIDTH;
    const F2: usize = WFL2_WIDTH * WFL2_WIDTH;
    const F1: usize = WFL1_WIDTH * WFL1_WIDTH;
    const G3: usize = WGL3_WIDTH * WGL3_WIDTH;
    const G2: usize = WGL2_WIDTH * WGL2_WIDTH;
    const G1: usize = WGL1_WIDTH * WGL1_WIDTH;
    const ZERO: __m256 = unsafe { std::mem::transmute([0.0f32; 8]) };
    const ALPHA: __m256 = unsafe { std::mem::transmute([0.01f32; 8]) };
    pub fn from(nn: NNLineEvaluator_) -> Self {
        let mut f1: Vec<__m256> = Vec::new();
        for v in nn.wfl1.iter() {
            f1.push(unsafe { _mm256_load_ps(v.as_ptr()) });
        }
        let mut f2 = Vec::new();
        for v in nn.wfl2.iter() {
            f2.push({ unsafe { _mm256_load_ps(v.as_ptr()) } });
        }
        let mut f3 = Vec::new();
        for v in nn.wfl3.iter() {
            f3.push({ unsafe { _mm256_load_ps(v.as_ptr()) } });
        }
        let mut g1: Vec<__m256> = Vec::new();
        for v in nn.wgl1.iter() {
            g1.push(unsafe { _mm256_load_ps(v.as_ptr()) });
        }
        let mut g2 = Vec::new();
        for v in nn.wgl2.iter() {
            g2.push({ unsafe { _mm256_load_ps(v.as_ptr()) } });
        }
        let mut g3 = Vec::new();
        for v in nn.wgl3.iter() {
            g3.push({ unsafe { _mm256_load_ps(v.as_ptr()) } });
        }

        let mut t3b = Vec::new();
        for v in nn.wt3nb.iter() {
            t3b.push({ unsafe { _mm256_load_ps(v.as_ptr()) } });
        }
        let mut t3w = Vec::new();
        for v in nn.wt3nw.iter() {
            t3w.push({ unsafe { _mm256_load_ps(v.as_ptr()) } });
        }
        let mut core = Vec::new();
        for v in nn.wcore.iter() {
            core.push(unsafe { _mm256_load_ps(v.as_ptr()) });
        }
        let mut turn = Vec::new();
        for v in nn.wturn.iter() {
            turn.push(unsafe { _mm256_load_ps(v.as_ptr()) });
        }

        let bias = unsafe { _mm256_load_ps(nn.bias.as_ptr()) };
        let acum = unsafe { _mm256_load_ps(nn.w_acum.as_ptr()) };
        let lbias = nn.lbias;

        return MMEvaluator {
            wfl3: f3,
            wfl2: f2,
            wfl1: f1,
            wgl3: g3,
            wgl2: g2,
            wgl1: g1,
            wt3nb: t3b,
            wt3nw: t3w,
            wcore: core,
            w_acum: acum,
            wturn: turn,
            bias: bias,
            lbias: lbias,
        };
    }
    pub fn evaluate_board(&self, b: &Board) -> f32 {
        let (af1, af2, af3, ag1, ag2, ag3, df1, df2, df3, dg1, dg2, dg3, tn3) =
            line::SimplLineEvaluator::get_counts(&b);
        let (att, def) = b.get_att_def();
        let n_stone = (att | def).count_ones() as usize;
        let is_black = n_stone % 2 == 0;

        let mut input = vec![0.0; Self::INPUT];

        let wt3;
        if is_black {
            wt3 = self.wt3nb[tn3].clone();
        } else {
            wt3 = self.wt3nw[tn3].clone();
        }
        let core = self.wcore[NNLineEvaluator_::get_core_idx(att, def)];
        // println!("core:{}", Self::get_core_idx(att, def));

        let mut v = self.wfl1[af1 * WFL1_WIDTH + df1];

        v = unsafe { _mm256_add_ps(v, self.wfl2[af2 * WFL2_WIDTH + df2]) };
        v = unsafe { _mm256_add_ps(v, self.wfl3[af3 * WFL3_WIDTH + df3]) };
        v = unsafe { _mm256_add_ps(v, self.wgl1[ag1 * WGL1_WIDTH + dg1]) };
        v = unsafe { _mm256_add_ps(v, self.wgl2[ag2 * WGL2_WIDTH + dg2]) };
        v = unsafe { _mm256_add_ps(v, self.wgl3[ag3 * WGL3_WIDTH + dg3]) };
        v = unsafe { _mm256_add_ps(v, core) };
        v = unsafe { _mm256_add_ps(v, self.wturn[n_stone]) };
        v = unsafe { _mm256_add_ps(v, self.bias) };
        v = unsafe { _mm256_max_ps(v, _mm256_mul_ps(v, Self::ALPHA)) };

        v = unsafe { _mm256_mul_ps(v, self.w_acum) };

        v = unsafe { _mm256_add_ps(v, _mm256_permute_ps::<0b01_00_11_10>(v)) };
        v = unsafe { _mm256_add_ps(v, _mm256_permute_ps::<0b10_11_00_01>(v)) };

        let mut out: [f32; 8] = [0.0; 8];
        unsafe { _mm256_store_ps(&mut out as *mut f32, v) };

        let val = out[0] + out[4] + self.lbias;

        return 1.0 / (1.0 + (-val).exp());
    }
}

impl EvaluatorF for MMEvaluator {
    fn eval_func_f32(&self, b: &Board) -> f32 {
        return self.evaluate_board(b);
    }
}

type BundleBitBoard = (u64, u64, u64, u64, u64, u64, u64);
type GroupBitBoard = (
    (BundleBitBoard, BundleBitBoard, BundleBitBoard),
    (BundleBitBoard, BundleBitBoard, BundleBitBoard),
);

pub fn acum_count_bbb(b: BundleBitBoard) -> usize {
    return (b.0.count_ones()
        + b.1.count_ones()
        + b.2.count_ones()
        + b.3.count_ones()
        + b.4.count_ones()
        + b.5.count_ones()
        + b.6.count_ones()) as usize;
}
pub fn acum_or_bbb(b: BundleBitBoard) -> u64 {
    return b.0 | b.1 | b.2 | b.3 | b.4 | b.5 | b.6;
}

#[derive(Clone, Serialize, Deserialize)]
pub struct NNLineEvaluator_ {
    pub wfl3: Vec<Vec<f32>>,
    pub wgl3: Vec<Vec<f32>>,
    pub wfl2: Vec<Vec<f32>>,
    pub wgl2: Vec<Vec<f32>>,
    pub wfl1: Vec<Vec<f32>>,
    pub wgl1: Vec<Vec<f32>>,
    pub wt3nw: Vec<Vec<f32>>,
    pub wt3nb: Vec<Vec<f32>>,
    pub wcore: Vec<Vec<f32>>,
    pub wturn: Vec<Vec<f32>>,
    pub wbboard: Vec<Vec<f32>>,
    pub w_acum: Vec<f32>,
    pub bias: Vec<f32>,
    pub lbias: f32,
}

impl NNLineEvaluator_ {
    const D: usize = 8;
    const INPUT: usize = 7 * Self::D;
    const F3: usize = WFL3_WIDTH * WFL3_WIDTH;
    const F2: usize = WFL2_WIDTH * WFL2_WIDTH;
    const F1: usize = WFL1_WIDTH * WFL1_WIDTH;
    const G3: usize = WGL3_WIDTH * WGL3_WIDTH;
    const G2: usize = WGL2_WIDTH * WGL2_WIDTH;
    const G1: usize = WGL1_WIDTH * WGL1_WIDTH;
    const CORE_SIZE: usize = 1 << 12;
    const CORE_MAGIC: u64 = 0x201060008020200;
    const B_SIZE: usize = 5376;

    pub fn get_core_idx(a: u64, d: u64) -> usize {
        let a = a & 0x0000_0660_0660_0000;
        let d = a | ((d & 0x0000_0660_0660_0000) << 8);
        return (((d * Self::CORE_MAGIC) >> 52) & 0xfff) as usize;
    }

    pub fn new() -> Self {
        use rand_distr::{Distribution, Normal};

        let mut f3 = vec![vec![0.0; Self::D]; Self::F3];
        let mut f2 = vec![vec![0.0; Self::D]; Self::F2];
        let mut f1 = vec![vec![0.0; Self::D]; Self::F1];
        let mut g3 = vec![vec![0.0; Self::D]; Self::G3];
        let mut g2 = vec![vec![0.0; Self::D]; Self::G2];
        let mut g1 = vec![vec![0.0; Self::D]; Self::G1];
        let mut tb = vec![vec![0.0; Self::D]; 12];
        let mut tw = vec![vec![0.0; Self::D]; 12];
        let mut tn = vec![vec![0.0; Self::D]; 65];
        let mut co = vec![vec![0.0; Self::D]; Self::CORE_SIZE];
        let mut bb = vec![vec![0.0; Self::D]; Self::B_SIZE];

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        for i in 0..Self::F3 {
            for j in 0..Self::D {
                f3[i][j] = normal.sample(&mut rng);
            }
        }
        for i in 0..Self::F2 {
            for j in 0..Self::D {
                f2[i][j] = normal.sample(&mut rng);
            }
        }
        for i in 0..Self::F1 {
            for j in 0..Self::D {
                f1[i][j] = normal.sample(&mut rng);
            }
        }

        for i in 0..Self::G3 {
            for j in 0..Self::D {
                g3[i][j] = normal.sample(&mut rng);
            }
        }
        for i in 0..Self::G2 {
            for j in 0..Self::D {
                g2[i][j] = normal.sample(&mut rng);
            }
        }
        for i in 0..Self::G1 {
            for j in 0..Self::D {
                g1[i][j] = normal.sample(&mut rng);
            }
        }
        for i in 0..12 {
            for j in 0..Self::D {
                tb[i][j] = normal.sample(&mut rng);
            }
        }
        for i in 0..12 {
            for j in 0..Self::D {
                tw[i][j] = normal.sample(&mut rng);
            }
        }
        for i in 0..65 {
            for j in 0..Self::D {
                tn[i][j] = normal.sample(&mut rng);
            }
        }
        for i in 0..Self::CORE_SIZE {
            for j in 0..Self::D {
                co[i][j] = normal.sample(&mut rng);
            }
        }
        for i in 0..Self::B_SIZE {
            for j in 0..Self::D {
                bb[i][j] = normal.sample(&mut rng);
            }
        }

        let mut bias = vec![0.0; Self::D];

        for j in 0..Self::D {
            bias[j] = normal.sample(&mut rng);
        }

        let sigma = (1.0 / D as f32).sqrt();
        let normal = Normal::new(0.0, sigma).unwrap();
        let mut acum = vec![0.0; Self::D];

        for j in 0..Self::D {
            acum[j] = normal.sample(&mut rng);
        }

        return NNLineEvaluator_ {
            wfl3: f3,
            wgl3: g3,
            wfl2: f2,
            wgl2: g2,
            wfl1: f1,
            wgl1: g1,
            wt3nw: tw,
            wt3nb: tb,
            wturn: tn,
            w_acum: acum,
            wcore: co,
            wbboard: bb,
            bias: bias,
            lbias: 0.0,
        };
    }

    pub fn zero() -> Self {
        return NNLineEvaluator_ {
            wfl3: vec![vec![0.0; Self::D]; Self::F3],
            wgl3: vec![vec![0.0; Self::D]; Self::G3],
            wfl2: vec![vec![0.0; Self::D]; Self::F2],
            wgl2: vec![vec![0.0; Self::D]; Self::G2],
            wfl1: vec![vec![0.0; Self::D]; Self::F1],
            wgl1: vec![vec![0.0; Self::D]; Self::G1],
            wt3nw: vec![vec![0.0; Self::D]; 12],
            wt3nb: vec![vec![0.0; Self::D]; 12],
            wturn: vec![vec![0.0; Self::D]; 65],
            wbboard: vec![vec![0.0; Self::D]; Self::B_SIZE],
            w_acum: vec![0.0; Self::D],
            bias: vec![0.0; Self::D],
            wcore: vec![vec![0.0; Self::D]; Self::CORE_SIZE],
            lbias: 0.0,
        };
    }

    pub fn analyze_board(a: u64, d: u64) -> GroupBitBoard {
        let stone = a | d;
        let b = !stone;
        let g = ((stone << 16) | 0xffff) & b;
        let f = b ^ g;

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
        let (yx1, yx2, yx3) =
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
        let (zx1, zx2, zx3) = LineEvaluator::analyze_line(
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
        let (zy1, zy2, zy3) = LineEvaluator::analyze_line(
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

        let (xy1, xy2, xy3) = (xy1 | yx1, xy2 | yx2, xy3 | yx3);
        let (yz1, yz2, yz3) = (yz1 | zy1, yz2 | zy2, yz3 | zy3);
        let (xz1, xz2, xz3) = (xz1 | zx1, xz2 | zx2, xz3 | zx3);
        let (xyz1, xyz2, xyz3) = (
            xyz11 | xyz21 | xyz31 | xyz41,
            xyz12 | xyz22 | xyz32 | xyz42,
            xyz13 | xyz23 | xyz33 | xyz43,
        );

        let (x1g, x2g, x3g) = (x1 & g, x2 & g, x3 & g);
        let (x1f, x2f, x3f) = (x1 & f, x2 & f, x3 & f);
        let (y1g, y2g, y3g) = (y1 & g, y2 & g, y3 & g);
        let (y1f, y2f, y3f) = (y1 & f, y2 & f, y3 & f);
        let (z1g, z2g, z3g) = (z1 & g, z2 & g, z3 & g);
        let (z1f, z2f, z3f) = (z1 & f, z2 & f, z3 & f);
        let (xy1g, xy2g, xy3g) = (xy1 & g, xy2 & g, xy3 & g);
        let (xy1f, xy2f, xy3f) = (xy1 & f, xy2 & f, xy3 & f);
        let (yz1g, yz2g, yz3g) = (yz1 & g, yz2 & g, yz3 & g);
        let (yz1f, yz2f, yz3f) = (yz1 & f, yz2 & f, yz3 & f);
        let (xz1g, xz2g, xz3g) = (xz1 & g, xz2 & g, xz3 & g);
        let (xz1f, xz2f, xz3f) = (xz1 & f, xz2 & f, xz3 & f);
        let (xyz1g, xyz2g, xyz3g) = (xyz1 & g, xyz2 & g, xyz3 & g);
        let (xyz1f, xyz2f, xyz3f) = (xyz1 & f, xyz2 & f, xyz3 & f);

        return (
            (
                (x1g, y1g, z1g, xy1g, yz1g, xz1g, xyz1g),
                (x2g, y2g, z2g, xy2g, yz2g, xz2g, xyz2g),
                (x3g, y3g, z3g, xy3g, yz3g, xz3g, xyz3g),
            ),
            (
                (x1f, y1f, z1f, xy1f, yz1f, xz1f, xyz1f),
                (x2f, y2f, z2f, xy2f, yz2f, xz2f, xyz2f),
                (x3f, y3f, z3f, xy3f, yz3f, xz3f, xyz3f),
            ),
        );
    }

    pub fn get_counts(
        ga: GroupBitBoard,
        gd: GroupBitBoard,
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
        let l3_mask = acum_or_bbb(ga.1 .2) | acum_or_bbb(gd.1 .2);
        let trap_3_num = (l3_mask & (!l3_mask << 16) & 0xffff_0000_0000).count_ones();

        return (
            acum_count_bbb(ga.1 .0),
            acum_count_bbb(ga.1 .1),
            acum_count_bbb(ga.1 .2),
            acum_count_bbb(ga.0 .0),
            acum_count_bbb(ga.0 .1),
            acum_count_bbb(ga.0 .2),
            acum_count_bbb(gd.1 .0),
            acum_count_bbb(gd.1 .1),
            acum_count_bbb(gd.1 .2),
            acum_count_bbb(gd.0 .0),
            acum_count_bbb(gd.0 .1),
            acum_count_bbb(gd.0 .2),
            trap_3_num as usize,
        );
    }

    pub fn get_idxs(a: GroupBitBoard, d: GroupBitBoard) -> Vec<usize> {
        let mut offset = 0;
        let input = [
            a.0 .0, a.0 .1, a.0 .2, a.1 .0, a.1 .1, a.1 .2, d.0 .0, d.0 .1, d.0 .2, d.1 .0, d.1 .1,
            d.1 .2,
        ];
        let mut idxs = Vec::new();
        for bits in input.iter() {
            for mut bit in [bits.0, bits.1, bits.2, bits.3, bits.4, bits.5, bits.6] {
                loop {
                    if bit == 0 {
                        break;
                    }
                    let onehot = bit & !(bit - 1);
                    bit ^= onehot;
                    idxs.push(bit.trailing_zeros() as usize);
                }
                offset += 64;
            }
        }
        return idxs;
    }

    pub fn evaluate_board(&self, b: &Board) -> f32 {
        let (att, def) = b.get_att_def();
        let ga = Self::analyze_board(att, def);
        let gd = Self::analyze_board(def, att);

        let (af1, af2, af3, ag1, ag2, ag3, df1, df2, df3, dg1, dg2, dg3, tn3) =
            Self::get_counts(ga, gd);
        let idxs = Self::get_idxs(ga, gd);

        let (att, def) = b.get_att_def();
        let n_stone = (att | def).count_ones() as usize;
        let is_black = n_stone % 2 == 0;

        let mut input = vec![0.0; Self::INPUT];

        let wt3;
        if is_black {
            wt3 = self.wt3nb[tn3].clone();
        } else {
            wt3 = self.wt3nw[tn3].clone();
        }
        let core = &self.wcore[Self::get_core_idx(att, def)];
        // println!("core:{}", Self::get_core_idx(att, def));

        let mut v = self.wfl1[af1 * WFL1_WIDTH + df1].clone();

        for i in 0..Self::D {
            v[i] += self.wfl2[af2 * WFL2_WIDTH + df2][i];
            v[i] += self.wfl3[af3 * WFL3_WIDTH + df3][i];
            v[i] += self.wgl1[ag1 * WGL1_WIDTH + dg1][i];
            v[i] += self.wgl2[ag2 * WGL2_WIDTH + dg2][i];
            v[i] += self.wgl3[ag3 * WGL3_WIDTH + dg3][i];
            v[i] += wt3[i];
            v[i] += core[i];
            v[i] += self.bias[i];
            v[i] += self.wturn[n_stone][i];
        }

        for idx in idxs {
            for i in 0..Self::D {
                v[i] += self.wbboard[idx][i];
            }
        }
        // let val = v[0];

        // return 1.0 / (1.0 + (-val).exp());

        let val = v
            .iter()
            .zip(self.w_acum.iter())
            .map(|(a, b)| (a.max(a * 0.01)) * b)
            // .map(|(a, b)| a * b)
            .sum::<f32>()
            + self.lbias;

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
        let mut src: Self = serde_json::from_str(&data_str)?;

        std::mem::swap(self, &mut src);

        Ok(())
    }
}

impl EvaluatorF for NNLineEvaluator_ {
    fn eval_func_f32(&self, b: &Board) -> f32 {
        return self.evaluate_board(b);
    }
}

#[derive(Clone)]
pub struct TrainableNLE_ {
    main: NNLineEvaluator_,
    v: NNLineEvaluator_,
    m: NNLineEvaluator_,
    lr: f32,
    batch_size: usize,
    batch_i: usize,
    beta: f32,
}

impl TrainableNLE_ {
    pub fn new(lr: f32) -> Self {
        TrainableNLE_ {
            main: NNLineEvaluator_::new(),
            v: NNLineEvaluator_::zero(),
            m: NNLineEvaluator_::zero(),
            lr: lr,
            batch_size: 16,
            batch_i: 0,
            beta: 0.9,
        }
    }

    pub fn from(e: NNLineEvaluator_, lr: f32) -> Self {
        TrainableNLE_ {
            main: e,
            v: NNLineEvaluator_::zero(),
            m: NNLineEvaluator_::zero(),
            lr: lr,
            batch_size: 16,
            batch_i: 0,
            beta: 0.9,
        }
    }
}

impl Trainable for TrainableNLE_ {
    fn update(&mut self, b: &Board, delta: f32) {
        // let (a1, a2, a3, a1_, a2_, a3_, d1, d2, d3, d1_, d2_, d3_, trap_3_num) =
        //     SimplLineEvaluator::get_counts(b);
        // let is_black = (b.black.count_ones() + b.white.count_ones()) % 2 == 0;
        // // とりあえずsgd
        // let val = self.main.evaluate_board(b);
        // let dv = val * (1.0 - val);
        // let delta = self.lr * delta * dv;
        // self.main.wfl1[a1 * WFL1_WIDTH + d1][0] += delta;
        // self.main.wfl2[a2 * WFL2_WIDTH + d2][0] += delta;
        // self.main.wfl3[a3 * WFL3_WIDTH + d3][0] += delta;
        // self.main.wgl1[a1_ * WGL1_WIDTH + d1_][0] += delta;
        // self.main.wgl2[a2_ * WGL2_WIDTH + d2_][0] += delta;
        // self.main.wgl3[a3_ * WGL3_WIDTH + d3_][0] += delta;

        // if is_black {
        //     self.main.wt3nb[trap_3_num][0] += delta;
        // } else {
        //     self.main.wt3nw[trap_3_num][0] += delta;
        // }
        // self.main.bias[0] += delta;

        // return;

        self.batch_i += 1;
        let (att, def) = b.get_att_def();
        let ga = NNLineEvaluator_::analyze_board(att, def);
        let gd = NNLineEvaluator_::analyze_board(def, att);

        let (af1, af2, af3, ag1, ag2, ag3, df1, df2, df3, dg1, dg2, dg3, tn3) =
            NNLineEvaluator_::get_counts(ga, gd);
        let idxs = NNLineEvaluator_::get_idxs(ga, gd);

        let n_stone = (att | def).count_ones() as usize;
        let is_black = n_stone % 2 == 0;

        let wt3;
        if is_black {
            wt3 = self.main.wt3nb[tn3].clone();
        } else {
            wt3 = self.main.wt3nw[tn3].clone();
        }

        let (f1, f2, f3, g1, g2, g3) = (
            af1 * WFL1_WIDTH + df1,
            af2 * WFL2_WIDTH + df2,
            af3 * WFL3_WIDTH + df3,
            ag1 * WGL1_WIDTH + dg1,
            ag2 * WGL2_WIDTH + dg2,
            ag3 * WGL3_WIDTH + dg3,
        );

        let core_idx = NNLineEvaluator_::get_core_idx(att, def);

        let mut v0 = self.main.wfl1[af1 * WFL1_WIDTH + df1].clone();
        for i in 0..NNLineEvaluator_::D {
            v0[i] += self.main.wfl2[f2][i];
            v0[i] += self.main.wfl3[f3][i];
            v0[i] += self.main.wgl1[g1][i];
            v0[i] += self.main.wgl2[g2][i];
            v0[i] += self.main.wgl3[g3][i];
            v0[i] += wt3[i];
            v0[i] += self.main.wcore[core_idx][i];
            v0[i] += self.main.bias[i];
            v0[i] += self.main.wturn[n_stone][i];
        }

        // println!("idx.size:{}", idxs.len());
        for &idx in idxs.iter() {
            for i in 0..NNLineEvaluator_::D {
                v0[i] += self.main.wbboard[idx][i];
            }
        }

        let v1 = v0.iter().map(|a| a.max(a * 0.01)).collect::<Vec<f32>>();

        let v2 = v1
            .iter()
            .zip(self.main.w_acum.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>()
            + self.main.lbias;
        let v3 = 1.0 / (1.0 + (-v2).exp());

        // println!("val:{v3}, bias:{}, delta:{delta}", self.main.lbias);

        let dv2 = v3 * (1.0 - v3) * delta * self.lr;
        self.main.lbias += dv2;

        // dR
        // let dw_w_acum: Vec<f32> = dv2.iter().zip(v1.iter()).map(|(a, b)| a * b).collect();
        let dw_w_acum: Vec<f32> = v1.iter().map(|a| a * dv2).collect();
        let dv1: Vec<f32> = self.main.w_acum.iter().map(|a| a * dv2).collect();
        let dv0: Vec<f32> = dv1
            .iter()
            .zip(v0.iter())
            .map(|(a, b)| if *b < 0.0 { 0.01 * a } else { *a })
            .collect();
        let di: Vec<f32> = dv0.clone();

        // println!("v0:{:#?}", v0);
        // println!("v1:{:#?}", v1);
        // println!("v2:{:#?}", v2);
        // println!("v3:{:#?}", v3);

        // println!("dv0:{:#?}", dv0);
        // println!("dv1:{:#?}", dv1);
        // println!("dv2:{:#?}", dv2);
        // println!("dv3:{:#?}", delta);
        self.v.w_acum = self
            .v
            .w_acum
            .iter()
            .zip(dw_w_acum.iter())
            .map(|(a, b)| a * self.beta + (1.0 - self.beta) * b)
            .collect();
        self.main.w_acum = self
            .main
            .w_acum
            .iter()
            .zip(self.v.w_acum.iter())
            .map(|(a, b)| a + b)
            .collect();
        // self.main.w_acum = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];

        // di

        for &idx in idxs.iter() {
            for i in 0..NNLineEvaluator_::D {
                self.v.wbboard[idx][i] =
                    self.beta * self.v.wbboard[idx][i] + (1.0 - self.beta) * di[i];
                self.main.wbboard[idx][i] += self.v.wbboard[idx][i];
            }
        }

        for i in 0..NNLineEvaluator_::D {
            // print!("{:#?},", self.v.wfl1[f1][i]);
            // print!("{:#?},", self.v.wfl2[f2][i]);
            // print!("{:#?},", self.v.wfl3[f3][i]);
            // print!("{:#?},", self.v.wgl1[g1][i]);
            // print!("{:#?},", self.v.wgl2[g2][i]);
            // print!("{:#?},", self.v.wgl3[g3][i]);
            // print!("{:#?},", self.v.bias[i]);
            // println!("");

            self.v.wfl1[f1][i] = self.beta * self.v.wfl1[f1][i] + (1.0 - self.beta) * di[i];
            self.v.wfl2[f2][i] = self.beta * self.v.wfl2[f2][i] + (1.0 - self.beta) * di[i];
            self.v.wfl3[f3][i] = self.beta * self.v.wfl3[f3][i] + (1.0 - self.beta) * di[i];
            self.v.wgl1[g1][i] = self.beta * self.v.wgl1[g1][i] + (1.0 - self.beta) * di[i];
            self.v.wgl2[g2][i] = self.beta * self.v.wgl2[g2][i] + (1.0 - self.beta) * di[i];
            self.v.wgl3[g3][i] = self.beta * self.v.wgl3[g3][i] + (1.0 - self.beta) * di[i];
            self.v.bias[i] = self.beta * self.v.bias[i] + (1.0 - self.beta) * di[i];
            if is_black {
                self.v.wt3nb[tn3][i] = self.beta * self.v.wt3nb[tn3][i] + (1.0 - self.beta) * di[i];
            } else {
                self.v.wt3nw[tn3][i] = self.beta * self.v.wt3nw[tn3][i] + (1.0 - self.beta) * di[i];
            }
            self.v.wcore[core_idx][i] =
                self.beta * self.v.wcore[core_idx][i] + (1.0 - self.beta) * di[i];
            self.v.wturn[n_stone][i] =
                self.beta * self.v.wturn[n_stone][i] + (1.0 - self.beta) * di[i];
            self.main.wfl1[f1][i] += self.v.wfl1[f1][i];
            self.main.wfl2[f2][i] += self.v.wfl2[f2][i];
            self.main.wfl3[f3][i] += self.v.wfl3[f3][i];
            self.main.wgl1[g1][i] += self.v.wgl1[g1][i];
            self.main.wgl2[g2][i] += self.v.wgl2[g2][i];
            self.main.wgl3[g3][i] += self.v.wgl3[g3][i];
            self.main.bias[i] += self.v.bias[i];
            if is_black {
                self.main.wt3nb[tn3][i] += self.v.wt3nb[tn3][i];
            } else {
                self.main.wt3nw[tn3][i] += self.v.wt3nw[tn3][i];
            }
            self.main.wcore[core_idx][i] += self.v.wcore[core_idx][i];
            self.main.wturn[n_stone][i] += self.v.wturn[n_stone][i];
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

impl EvaluatorF for TrainableNLE_ {
    fn eval_func_f32(&self, b: &Board) -> f32 {
        return self.main.evaluate_board(b).clamp(0.0, 1.0);
    }
}
