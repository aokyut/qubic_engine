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
    EvaluatorF, Trainable,
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
            w_acum: acum,
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
            w_acum: vec![0.0; Self::D],
            bias: vec![0.0; Self::D],
            lbias: 0.0,
        };
    }

    pub fn evaluate_board(&self, b: &Board) -> f32 {
        let (af1, af2, af3, ag1, ag2, ag3, df1, df2, df3, dg1, dg2, dg3, tn3) =
            line::SimplLineEvaluator::get_counts(&b);
        let (att, def) = b.get_att_def();
        let is_black = (att.count_ones() + def.count_ones()) % 2 == 0;

        let mut input = vec![0.0; Self::INPUT];

        let wt3;
        if is_black {
            wt3 = self.wt3nb[tn3].clone();
        } else {
            wt3 = self.wt3nw[tn3].clone();
        }

        let mut v = self.wfl1[af1 * WFL1_WIDTH + df1].clone();

        for i in 0..Self::D {
            v[i] += self.wfl2[af2 * WFL2_WIDTH + df2][i];
            v[i] += self.wfl3[af3 * WFL3_WIDTH + df3][i];
            v[i] += self.wgl1[ag1 * WGL1_WIDTH + dg1][i];
            v[i] += self.wgl2[ag2 * WGL2_WIDTH + dg2][i];
            v[i] += self.wgl3[ag3 * WGL3_WIDTH + dg3][i];
            v[i] += wt3[i];
            v[i] += self.bias[i];
        }
        // let val = v[0];

        // return 1.0 / (1.0 + (-val).exp());

        let val = v
            .iter()
            .zip(self.w_acum.iter())
            .map(|(a, b)| a.max(0.0) * b + (a * 0.01).min(0.0) * b)
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

    pub fn adjust(&mut self) {
        let mut mean = vec![0.0; Self::D];
        let mut square = vec![0.0; Self::D];
        let mut stds = vec![0.0; Self::D];
        let sigma = (Self::D as f32).sqrt();

        for j in 0..Self::D {
            for i in 0..self.wfl1.len() {
                mean[j] += self.wfl1[i][j];
                square[j] += self.wfl1[i][j].powi(2);
            }
            mean[j] /= self.wfl1.len() as f32;
            square[j] /= self.wfl1.len() as f32;
            stds[j] = (square[j] - mean[j].powi(2)).sqrt();
        }
        for i in 0..self.wfl1.len() {
            for j in 0..Self::D {
                self.wfl1[i][j] = (self.wfl1[i][j] - mean[j]) / (stds[j] * sigma);
            }
        }

        let mut mean = vec![0.0; Self::D];
        let mut square = vec![0.0; Self::D];
        let mut stds = vec![0.0; Self::D];

        for j in 0..Self::D {
            for i in 0..self.wfl2.len() {
                mean[j] += self.wfl2[i][j];
                square[j] += self.wfl2[i][j].powi(2);
            }
            mean[j] /= self.wfl2.len() as f32;
            square[j] /= self.wfl2.len() as f32;
            stds[j] = (square[j] - mean[j].powi(2)).sqrt();
        }
        for i in 0..self.wfl2.len() {
            for j in 0..Self::D {
                self.wfl2[i][j] = (self.wfl2[i][j] - mean[j]) / (stds[j] * sigma);
            }
        }
        let mut mean = vec![0.0; Self::D];
        let mut square = vec![0.0; Self::D];
        let mut stds = vec![0.0; Self::D];

        for j in 0..Self::D {
            for i in 0..self.wfl3.len() {
                mean[j] += self.wfl3[i][j];
                square[j] += self.wfl3[i][j].powi(2);
            }
            mean[j] /= self.wfl3.len() as f32;
            square[j] /= self.wfl3.len() as f32;
            stds[j] = (square[j] - mean[j].powi(2)).sqrt();
        }
        for i in 0..self.wfl3.len() {
            for j in 0..Self::D {
                self.wfl3[i][j] = (self.wfl3[i][j] - mean[j]) / (stds[j] * sigma);
            }
        }

        let mut mean = vec![0.0; Self::D];
        let mut square = vec![0.0; Self::D];
        let mut stds = vec![0.0; Self::D];

        for j in 0..Self::D {
            for i in 0..self.wgl1.len() {
                mean[j] += self.wgl1[i][j];
                square[j] += self.wgl1[i][j].powi(2);
            }
            mean[j] /= self.wgl1.len() as f32;
            square[j] /= self.wgl1.len() as f32;
            stds[j] = (square[j] - mean[j].powi(2)).sqrt();
        }
        for i in 0..self.wgl1.len() {
            for j in 0..Self::D {
                self.wgl1[i][j] = (self.wgl1[i][j] - mean[j]) / (stds[j] * sigma);
            }
        }

        let mut mean = vec![0.0; Self::D];
        let mut square = vec![0.0; Self::D];
        let mut stds = vec![0.0; Self::D];

        for j in 0..Self::D {
            for i in 0..self.wgl2.len() {
                mean[j] += self.wgl2[i][j];
                square[j] += self.wgl2[i][j].powi(2);
            }
            mean[j] /= self.wgl2.len() as f32;
            square[j] /= self.wgl2.len() as f32;
            stds[j] = (square[j] - mean[j].powi(2)).sqrt();
        }
        for i in 0..self.wgl2.len() {
            for j in 0..Self::D {
                self.wgl2[i][j] = (self.wgl2[i][j] - mean[j]) / (stds[j] * sigma);
            }
        }
        let mut mean = vec![0.0; Self::D];
        let mut square = vec![0.0; Self::D];
        let mut stds = vec![0.0; Self::D];

        for j in 0..Self::D {
            for i in 0..self.wgl3.len() {
                mean[j] += self.wgl3[i][j];
                square[j] += self.wgl3[i][j].powi(2);
            }
            mean[j] /= self.wgl3.len() as f32;
            square[j] /= self.wgl3.len() as f32;
            stds[j] = (square[j] - mean[j].powi(2)).sqrt();
        }
        for i in 0..self.wgl3.len() {
            for j in 0..Self::D {
                self.wgl3[i][j] = (self.wgl3[i][j] - mean[j]) / (stds[j] * sigma);
            }
        }
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
        let (af1, af2, af3, ag1, ag2, ag3, df1, df2, df3, dg1, dg2, dg3, tn3) =
            line::SimplLineEvaluator::get_counts(&b);
        let (att, def) = b.get_att_def();
        let is_black = (att.count_ones() + def.count_ones()) % 2 == 0;

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

        let mut v0 = self.main.wfl1[af1 * WFL1_WIDTH + df1].clone();
        for i in 0..NNLineEvaluator_::D {
            v0[i] += self.main.wfl2[af2 * WFL2_WIDTH + df2][i];
            v0[i] += self.main.wfl3[af3 * WFL3_WIDTH + df3][i];
            v0[i] += self.main.wgl1[ag1 * WGL1_WIDTH + dg1][i];
            v0[i] += self.main.wgl2[ag2 * WGL2_WIDTH + dg2][i];
            v0[i] += self.main.wgl3[ag3 * WGL3_WIDTH + dg3][i];
            v0[i] += wt3[i];
            v0[i] += self.main.bias[i];
        }

        let v1 = v0
            .iter()
            .map(|a| a.max(0.0) + (a * 0.01).min(0.0))
            .collect::<Vec<f32>>();
        // let v1: Vec<f32> = v0.iter().map(|a| *a).collect::<Vec<f32>>();

        let v2 = v1
            .iter()
            .zip(self.main.w_acum.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>()
            + self.main.lbias
            + v0[0];
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
            .map(|(a, b)| a * (b.signum() * 0.495 + 0.505))
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
        }

        // self.main.adjust();
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
