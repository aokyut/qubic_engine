use super::board::{eval_actor, Agent, Board, GetAction};
use super::{b2u128, u128_to_b, Analyzer, EvalAndAnalyze, EvaluatorF, NegAlphaF};
use crate::db::{random_rot, BoardDB};
use indicatif::{ProgressBar, ProgressStyle};
use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Normal};

use std::time::Duration;
use std::{thread, time};

const EPS: f32 = 0.0000001;

pub type Idxs = Vec<usize>;
pub type PatternMap = Vec<Idxs>;

pub type MapBundle = Vec<PatternMap>;

fn map_bundle_from_vec(map: &[usize]) -> MapBundle {
    return vec![vec![map.to_vec()]];
}

#[derive(Clone)]
pub struct Pattern {
    map: MapBundle,
    v: Vec<f32>,
}

impl Pattern {
    pub fn from(map: &[usize], v: &[f32]) -> Self {
        return Pattern {
            map: map_bundle_from_vec(map),
            v: v.to_vec(),
        };
    }

    pub fn zeros_like(pat: &Pattern) -> Self {
        let map = pat.map.clone();
        let mut v = Vec::new();
        for i in 0..pat.v.len() {
            v.push(0.0)
        }
        return Pattern { map: map, v: v };
    }

    pub fn new(map: Vec<Vec<Vec<usize>>>, scale: f32) -> Self {
        let s: usize = 3_usize.pow(map[0][0].len() as u32);
        let mut v = Vec::new();
        // let mut rng = rand::thread_rng();

        // let sigma = (1.0 / s as f32).sqrt();
        // let normal = Normal::new(scale / 2.0, sigma).unwrap();

        for _ in 0..s {
            // v.push(normal.sample(&mut rng));
            v.push(scale * 0.5);
        }
        return Pattern { map: map, v: v };
    }

    pub fn get_idxs(&self, b: &Board) -> Vec<usize> {
        let u7 = b2u128(b);
        let mut idxs = Vec::new();
        for pattern_map in self.map.iter() {
            let mut idx_min = None;
            for map in pattern_map.iter() {
                let mut idx = 0;
                for &map_id in map.iter() {
                    idx = (idx * 3) + (u7 >> map_id) & 1 + ((u7 >> (map_id + 64)) & 1) << 1;
                }
                match idx_min {
                    None => idx_min = Some(idx),
                    Some(idx_) => {
                        if idx < idx_ {
                            idx_min = Some(idx)
                        }
                    }
                }
            }
            idxs.push(idx_min.unwrap() as usize);
        }

        return idxs;
    }

    pub fn get_value(&self, b: &Board) -> Vec<f32> {
        let idxs = self.get_idxs(b);

        let mut vs = Vec::new();
        // println!("{}, {:#?}", self.v.len(), idxs);
        for i in idxs {
            vs.push(self.v[i]);
        }

        return vs;
    }

    pub fn update(&mut self, b: &Board, deltas: &[f32]) -> Vec<f32> {
        let idxs = self.get_idxs(b);
        let mut vs = Vec::new();

        for i in 0..idxs.len() {
            self.v[idxs[i]] += deltas[i];
            vs.push(self.v[idxs[i]]);
        }
        return vs;
    }
}

pub struct PatternEvaluator {
    patterns: Vec<Pattern>,
}

impl PatternEvaluator {
    pub fn new() -> Self {
        return PatternEvaluator {
            patterns: Vec::new(),
        };
    }
    pub fn push_pattern(&mut self, pattern: Pattern) {
        self.patterns.push(pattern);
    }

    pub fn get_value(&self, b: &Board) -> f32 {
        let mut val: f32 = 0.0;

        for pat in self.patterns.iter() {
            val += pat.get_value(b).iter().sum::<f32>();
        }

        return val;
    }
    pub fn update(&mut self, b: &Board, deltas: Vec<f32>) {
        for i in 0..self.patterns.len() {
            self.patterns[i].update(b, &deltas);
        }
    }
}

pub fn test_pattern_evaluator() -> PatternEvaluator {
    let mut pe = PatternEvaluator::new();
    let scale = 1.0 / 37.0;

    // 縦横高さ
    // pe.push_pattern(Pattern::new(&vec![0, 1, 2, 3, 4, 8, 12, 16, 32, 48], scale));
    // pe.push_pattern(Pattern::new(
    //     &vec![0, 1, 2, 3, 7, 11, 15, 19, 35, 51],
    //     scale,
    // ));
    // pe.push_pattern(Pattern::new(
    //     &vec![0, 4, 8, 12, 13, 14, 15, 28, 44, 60],
    //     scale,
    // ));
    // pe.push_pattern(Pattern::new(
    //     &vec![3, 7, 11, 12, 13, 14, 15, 31, 47, 63],
    //     scale,
    // ));

    // ブロック下段(4)
    pe.push_pattern(Pattern::new(
        vec![
            vec![vec![0, 1, 4, 5, 16, 17, 20, 21]],
            vec![vec![3, 7, 2, 6, 19, 23, 18, 22]],
            vec![vec![12, 8, 13, 9, 28, 24, 29, 25]],
            vec![vec![15, 14, 11, 10, 31, 30, 27, 26]],
        ],
        scale,
    ));
    // ブロック上段(4)
    pe.push_pattern(Pattern::new(
        vec![
            vec![vec![32, 33, 36, 37, 48, 49, 52, 53]],
            vec![vec![35, 39, 34, 38, 51, 55, 50, 54]],
            vec![vec![44, 40, 45, 41, 60, 56, 61, 57]],
            vec![vec![47, 46, 43, 42, 63, 62, 59, 58]],
        ],
        scale,
    ));

    // ブロック中央(1)
    pe.push_pattern(Pattern::new(
        vec![vec![vec![21, 22, 25, 26, 37, 38, 41, 42]]],
        scale,
    ));
    pe.push_pattern(Pattern::new(
        vec![vec![vec![5, 6, 9, 10, 21, 22, 25, 26]]],
        scale,
    ));

    // 下段二層 　外側(4)
    pe.push_pattern(Pattern::new(
        vec![
            vec![vec![0, 1, 2, 3, 16, 17, 18, 19]],
            vec![vec![12, 13, 14, 15, 28, 29, 30, 31]],
            vec![vec![0, 4, 8, 12, 16, 20, 24, 28]],
            vec![vec![3, 7, 11, 15, 19, 23, 27, 31]],
        ],
        scale,
    ));

    //　下段二層　内側(4)
    pe.push_pattern(Pattern::new(
        vec![
            vec![vec![4, 5, 6, 7, 20, 21, 22, 23]],
            vec![vec![8, 9, 10, 11, 24, 25, 26, 27]],
            vec![vec![1, 5, 9, 13, 17, 21, 25, 29]],
            vec![vec![2, 6, 10, 14, 18, 22, 26, 30]],
        ],
        scale,
    ));

    // 下段二層　クロス(2)
    pe.push_pattern(Pattern::new(
        vec![
            vec![vec![0, 5, 10, 15, 16, 21, 26, 31]],
            vec![vec![3, 6, 9, 12, 19, 22, 25, 28]],
        ],
        scale,
    ));

    // 逆ミッキー外側(4)
    pe.push_pattern(Pattern::new(
        vec![
            vec![vec![0, 3, 17, 18, 33, 34, 32, 35]],
            vec![vec![12, 15, 29, 30, 45, 46, 44, 47]],
            vec![vec![0, 12, 20, 24, 36, 40, 32, 44]],
            vec![vec![3, 15, 23, 27, 39, 43, 35, 47]],
        ],
        scale,
    ));
    // 逆ミッキー内側(4)
    pe.push_pattern(Pattern::new(
        vec![
            vec![vec![4, 7, 21, 22, 37, 38, 36, 39]],
            vec![vec![8, 11, 25, 26, 41, 42, 40, 43]],
            vec![vec![1, 13, 21, 25, 37, 41, 33, 45]],
            vec![vec![2, 14, 22, 26, 38, 42, 34, 46]],
        ],
        scale,
    ));

    // 逆ミッキー クロス(2)
    pe.push_pattern(Pattern::new(
        vec![
            vec![vec![0, 15, 21, 26, 37, 42, 32, 47]],
            vec![vec![3, 12, 22, 25, 38, 41, 35, 44]],
        ],
        scale,
    ));

    // under edge(4)
    pe.push_pattern(Pattern::new(
        vec![
            vec![vec![0, 1, 2, 4, 5, 8, 16, 17, 20, 32]],
            vec![vec![3, 7, 11, 2, 6, 1, 19, 23, 18, 35]],
            vec![vec![15, 14, 13, 11, 10, 7, 31, 30, 27, 47]],
            vec![vec![12, 8, 4, 13, 9, 14, 28, 24, 29, 44]],
        ],
        scale,
    ));

    // diag piller(4)
    pe.push_pattern(Pattern::new(
        vec![
            vec![vec![0, 5, 16, 21, 32, 37, 48, 53]],
            vec![vec![3, 6, 19, 22, 35, 38, 51, 54]],
            vec![vec![15, 10, 31, 26, 47, 42, 63, 58]],
            vec![vec![12, 9, 28, 25, 44, 41, 60, 57]],
        ],
        scale,
    ));

    // 外枠
    pe.push_pattern(Pattern::new(
        vec![
            vec![vec![3, 2, 1, 0, 16, 32, 48]],
            vec![vec![0, 1, 2, 3, 19, 35, 51]],
            vec![vec![15, 11, 7, 3, 19, 35, 51]],
            vec![vec![3, 7, 11, 15, 31, 47, 63]],
            vec![vec![12, 13, 14, 15, 31, 47, 63]],
            vec![vec![15, 14, 13, 12, 28, 44, 60]],
            vec![vec![0, 4, 8, 12, 28, 44, 60]],
            vec![vec![12, 8, 4, 0, 16, 32, 48]],
        ],
        scale,
    ));

    // pe.push_pattern(
    //     Pattern::new(
    //         vec![vec![vec![21, 22, 25, 26, 37, 38, 41, 42]]], scale
    //     )
    // );

    return pe;
}

pub struct TrainablePatternEvaluator {
    pats: Vec<Pattern>,
    pats_v: Vec<Pattern>,
    pats_m: Vec<Pattern>,
    beta1: f32,
    beta2: f32,
    alpha: f32,
}

impl TrainablePatternEvaluator {
    pub fn new(alpha: f32, beta1: f32, beta2: f32) -> Self {
        return TrainablePatternEvaluator {
            pats: Vec::new(),
            pats_v: Vec::new(),
            pats_m: Vec::new(),
            alpha: alpha,
            beta1: beta1,
            beta2: beta2,
        };
    }
    pub fn from_pattern_evaluator(
        pe: PatternEvaluator,
        alpha: f32,
        beta1: f32,
        beta2: f32,
    ) -> Self {
        let mut tpe = TrainablePatternEvaluator::new(alpha, beta1, beta2);

        for pat in pe.patterns.iter() {
            tpe.push_pattern(pat);
        }

        return tpe;
    }

    pub fn push_pattern(&mut self, pat: &Pattern) {
        self.pats_m.push(Pattern::zeros_like(&pat));
        self.pats_v.push(Pattern::zeros_like(&pat));
        self.pats.push(pat.clone());
    }

    pub fn to_pattern_evaluator(&self) -> PatternEvaluator {
        let mut pe = PatternEvaluator::new();

        for i in 0..self.pats.len() {
            pe.push_pattern(self.pats[i].clone());
        }

        return pe;
    }

    pub fn get_value(&self, b: &Board) -> f32 {
        let mut val = 0.0;
        for i in 0..self.pats.len() {
            val += self.pats[i].get_value(b).iter().sum::<f32>();
        }

        return val;
    }

    pub fn update(&mut self, b: &Board, delta: f32) {
        for i in 0..self.pats.len() {
            let vt_ = self.pats_v[i].get_value(b);
            let vt = self.pats_v[i].update(
                b,
                &vt_.iter()
                    .map(|v_| (self.beta1 - 1.0) * v_ + (1.0 - self.beta1) * delta)
                    .collect::<Vec<f32>>(),
            );
            let mt_ = self.pats_m[i].get_value(b);
            let mt = self.pats_m[i].update(
                b,
                &mt_.iter()
                    .map(|m_| (self.beta2 - 1.0) * m_ + (1.0 - self.beta2) * delta * delta)
                    .collect::<Vec<f32>>(),
            );
            let _ = self.pats[i].update(
                b,
                &vt.iter()
                    .zip(mt.iter())
                    .map(|(v, m)| -self.alpha * v / (m + EPS).sqrt())
                    .collect::<Vec<f32>>(),
            );
        }
    }
}

impl EvaluatorF for PatternEvaluator {
    fn eval_func(&self, b: &Board) -> f32 {
        let val = self.get_value(b);
        if val < 0.001 {
            return 0.001;
        } else if val > 0.999 {
            return 0.999;
        } else {
            val
        }
    }
}

impl Analyzer for PatternEvaluator {
    fn analyze_eval(&self, b: &Board) -> f32 {
        let val = self.get_value(b);
        if val < 0.001 {
            return 0.001;
        } else if val > 0.999 {
            return 0.999;
        } else {
            val
        }
    }
}

impl EvalAndAnalyze for PatternEvaluator {}

impl EvaluatorF for TrainablePatternEvaluator {
    fn eval_func(&self, b: &Board) -> f32 {
        let val = self.get_value(b);
        if val < 0.001 {
            return 0.001;
        } else if val > 0.999 {
            return 0.999;
        } else {
            val
        }
    }
}

pub fn train_with_db(
    load: bool,
    save: bool,
    name: String,
    db_name: String,
    eval_db_name: String,
    epochs: usize,
) {
    let mut trainable_pe = TrainablePatternEvaluator::from_pattern_evaluator(
        test_pattern_evaluator(),
        0.01,
        0.9,
        0.999,
    );

    let eval_n = 30;
    let data_size = 1 << 20;
    let eval_data_size = 1 << 14;
    let lambda = 0.5;

    let test_actor1 = Agent::Minimax(3);
    let test_actor2 = Agent::Mcts(50, 500);
    let mut rng = thread_rng();

    // TODO: resume and save

    let mut step = 0;

    let mut eval_db = BoardDB::new(&eval_db_name, eval_data_size);
    let eval_ts = eval_db.get_batch();

    for epoch in 0..epochs {
        let pe = trainable_pe.to_pattern_evaluator();
        let neg = NegAlphaF::new(Box::new(pe), 3);

        let (e11, e12) = eval_actor(&neg, &test_actor1, eval_n, false);
        let (e21, e22) = eval_actor(&neg, &test_actor2, eval_n, false);
        println!("[epoch-{}][minimax(3)]:({}, {})", epoch, e11, e12);
        println!("[epoch-{}][mcts(50, 500)]:({}, {})", epoch, e21, e22);

        let mut db: BoardDB = BoardDB::new(&db_name, data_size);

        let pb = ProgressBar::new(data_size as u64 * 1);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) \n {msg}")
            .unwrap()
            .progress_chars("#>-"));

        let mut i = 0;
        let mut smoothed_loss = None;

        let ts = db.get_batch();

        for _ in 0..1 {
            for t in ts.iter() {
                let b = &u128_to_b(random_rot(t.board, rng.gen()));
                let actual = trainable_pe.eval_func(b);
                let expected;
                if t.result == 0 {
                    expected = 0.5;
                } else if t.result == 1 {
                    expected = 0.999;
                } else {
                    expected = 0.001;
                }
                let expected = t.t_val * lambda + expected * (1.0 - lambda);
                let delta = actual - expected;
                let loss = delta.powi(2);

                trainable_pe.update(b, delta);

                match smoothed_loss {
                    None => {
                        smoothed_loss = Some(loss);
                    }
                    Some(s) => smoothed_loss = Some(0.99999 * s + 0.00001 * loss),
                }

                pb.inc(1);

                if i % 100 == 0 {
                    pb.set_message(format!(
                        "[loss]:{} \n[smoothed]:{}",
                        loss,
                        smoothed_loss.unwrap()
                    ));
                }

                if i % 1000 == 0 {
                    thread::sleep(Duration::from_millis(100));
                }

                // println!("[smoothed_loss]:{}", smoothed_loss.unwrap());
                i += 1;
            }
        }
        let mut eval_loss = 0.0;
        for t in eval_ts.iter() {
            let actual = trainable_pe.eval_func(&u128_to_b(t.board));
            let expected;
            if t.result == 0 {
                expected = 0.5;
            } else if t.result == 1 {
                expected = 0.999;
            } else {
                expected = 0.001;
            }
            let expected = t.t_val * lambda + expected * (1.0 - lambda);
            let delta = actual - expected;
            eval_loss += delta.powi(2);
        }
        pb.println(format!(
            "[eval_loss]:{}",
            eval_loss / (eval_data_size as f32)
        ));
    }
}
