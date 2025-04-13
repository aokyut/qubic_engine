pub mod pattern;

use crate::board::{self, count_1row, count_2row, count_3row, mate_check_horizontal, pprint_board};

use super::board::{Board, GetAction};
use super::ml::{Graph, Tensor};
use ndarray::{s, CowArray};
use ort::{Environment, GraphOptimizationLevel, Session, SessionBuilder};
use rand::Rng;
use std::time::Instant;

pub const MAX: i32 = 1600;

pub fn negmax<F>(b: &Board, depth: u8, eval_func: &F) -> (u8, i32, i32)
where
    F: Fn(&Board) -> i32,
{
    let mut count = 0;
    let mut max_val = -MAX - 1;
    let max_action = 16;
    let mut max_action: u8 = 16;
    let actions = b.valid_actions();
    for action in actions.iter() {
        let next_board = &b.next(*action);
        if next_board.is_win() {
            return (*action, -MAX, count);
        } else if next_board.is_draw() {
            return (*action, 0, count);
        } else if depth <= 1 {
            let val = eval_func(b);
            if max_val < val {
                max_val = val;
                max_action = *action;
            }
        } else {
            let (_, val, _count) = negmax(next_board, depth - 1, eval_func);
            count += 1 + _count;
            if max_val < val {
                max_val = val;
                max_action = *action;
            }
        }
    }

    return (max_action, max_val, count);
}

pub fn negalpha(
    b: &Board,
    depth: u8,
    alpha: i32,
    beta: i32,
    e: &Box<dyn Evaluator>,
) -> (u8, i32, i32) {
    // println!("depth:{depth}, alpha:{alpha}, beta:{beta}");
    // pprint_board(b);
    let mut count = 0;
    let actions = b.valid_actions();
    let mut max_val = -MAX - 1;
    let mut max_actions = Vec::new();
    let mut alpha = alpha;

    if depth <= 1 {
        for action in actions.iter() {
            let next_board = &b.next(*action);
            if next_board.is_win() {
                return (*action, MAX, count);
            } else if next_board.is_draw() {
                return (*action, 0, count);
            }
            let val = -e.eval_func(next_board);
            if max_val < val {
                max_val = val;
                max_actions = vec![*action];
                if max_val > alpha {
                    alpha = max_val;
                    if alpha > beta {
                        // println!("[{depth}]->max_val:{max_val}");
                        return (*action, max_val, count);
                    }
                }
            } else if max_val == val {
                max_actions.push(*action);
            }
        }
    } else {
        let mut action_nb_vals: Vec<(u8, Board, i32)> = Vec::new();

        for action in actions.into_iter() {
            let next_board = b.next(action);
            if next_board.is_win() {
                return (action, MAX, count);
            } else if next_board.is_draw() {
                return (action, 0, count);
            }

            let val = -e.eval_func(&next_board);
            action_nb_vals.push((action, next_board, val));
        }

        action_nb_vals.sort_by(|a, b| a.2.cmp(&b.2).reverse());
        // for (a, b, c) in action_nb_vals.iter() {
        //     print!("[{}, {}]", a, c);
        // }
        // println!("");

        for (action, next_board, val) in action_nb_vals {
            let (_, val, _count) = negalpha(&next_board, depth - 1, -beta, -alpha, e);
            count += 1 + _count;
            let val = -999 * val / 1000;
            if max_val < val {
                max_val = val;
                max_actions = vec![action];
                if max_val > alpha {
                    alpha = max_val;
                    if alpha > beta {
                        // println!("[{depth}]->max_val:{max_val}");
                        return (action, max_val, count);
                    }
                }
            } else if max_val == val {
                max_actions.push(action);
            }
        }
    }
    let mut rng = rand::thread_rng();
    return (
        max_actions[rng.gen::<usize>() % max_actions.len()],
        max_val,
        count,
    );
}

pub fn negalphaf(
    b: &Board,
    depth: u8,
    alpha: f32,
    beta: f32,
    e: &Box<dyn EvalAndAnalyze>,
) -> (u8, f32, i32) {
    // println!("depth:{depth}, alpha:{alpha}, beta:{beta}");
    // pprint_board(b);
    let mut count = 0;
    let actions = b.valid_actions();
    let mut max_val = -2.0;
    let mut max_actions = Vec::new();
    let mut alpha = alpha;
    for action in actions.iter() {
        let next_board = &b.next(*action);
        if next_board.is_win() {
            return (*action, 1.0, count);
        } else if next_board.is_draw() {
            return (*action, 0.5, count);
        } else if depth <= 1 {
            let val = 1.0 - e.eval_func(next_board);
            if max_val < val {
                max_val = val;
                max_actions = vec![*action];
                if max_val > alpha {
                    alpha = max_val;
                    if alpha > beta {
                        // println!("[{depth}]->max_val:{max_val}");
                        return (*action, max_val, count);
                    }
                }
            } else if max_val == val {
                max_actions.push(*action);
            }
        } else {
            // println!("a:{}, max_a:{:?}", *action, max_actions);
            let (_, val, _count) = negalphaf(next_board, depth - 1, 1.0 - beta, 1.0 - alpha, e);
            count += 1 + _count;
            let val = 1.0 - val;
            if max_val < val {
                max_val = val;
                max_actions = vec![*action];
                if max_val > alpha {
                    alpha = max_val;
                    if alpha > beta {
                        // println!("[{depth}]->max_val:{max_val}");
                        return (*action, max_val, count);
                    }
                }
            } else if max_val == val {
                max_actions.push(*action);
            }
        }
    }
    // println!("[{}]max_val:{}", max_action, max_val);
    let mut rng = rand::thread_rng();
    return (
        max_actions[rng.gen::<usize>() % max_actions.len()],
        max_val,
        count,
    );
}

pub struct NegAlpha {
    evaluator: Box<dyn Evaluator>,
    depth: u8,
}

impl NegAlpha {
    pub fn new(e: Box<dyn Evaluator>, depth: u8) -> Self {
        return NegAlpha {
            evaluator: e,
            depth: depth,
        };
    }
    pub fn eval_with_negalpha(&self, b: &Board) -> (u8, i32, i32) {
        return negalpha(b, self.depth, -MAX - 1, MAX + 1, &self.evaluator);
    }
}

impl GetAction for NegAlpha {
    fn get_action(&self, b: &Board) -> u8 {
        let (action, v, count) = negalpha(b, self.depth, -MAX - 1, MAX + 1, &self.evaluator);
        // pprint_board(b);
        // println!("action:{action}, val:{v}, count:{count}");
        return action;
    }
}

pub trait EvalAndAnalyze: EvaluatorF + Analyzer {}

pub struct NegAlphaF {
    evaluator: Box<dyn EvalAndAnalyze>,
    depth: u8,
}

impl NegAlphaF {
    pub fn new(e: Box<dyn EvalAndAnalyze>, depth: u8) -> Self {
        return NegAlphaF {
            evaluator: e,
            depth: depth,
        };
    }
    pub fn eval_with_negalpha(&self, b: &Board) -> (u8, f32, i32) {
        let actions = b.valid_actions();

        // for &action in actions.iter() {
        //     let next_b = b.next(action);
        //     let (_, val, _) = negalphaf(&next_b, self.depth - 1, -2.0, 2.0, &self.evaluator);
        //     let val = 1.0 - val;
        //     println!("[{}]:{}", action, val);
        // }

        return negalphaf(b, self.depth, -2.0, 2.0, &self.evaluator);
    }
}

impl GetAction for NegAlphaF {
    fn get_action(&self, b: &Board) -> u8 {
        let (action, _, _) = self.eval_with_negalpha(b);
        return action;
    }
}

pub trait Evaluator {
    fn eval_func(&self, b: &Board) -> i32;
}

pub trait EvaluatorF {
    fn eval_func(&self, b: &Board) -> f32;
}

pub trait Analyzer {
    fn analyze_eval(&self, b: &Board) -> f32;
    fn analyze(&self, b: &Board) {
        pprint_board(b);
        let actions = b.valid_actions();

        for &action in actions.iter() {
            let next_b = b.next(action);
            let val = 1.0 - self.analyze_eval(&next_b);
            println!("[{}]:{}", action, val);
        }
    }
}

pub struct PositionEvaluator {
    posmap: Vec<i32>,
}

impl PositionEvaluator {
    pub fn new(posmap: &[i32]) -> Self {
        return PositionEvaluator {
            posmap: posmap.to_vec(),
        };
    }
    pub fn simpl(vertex: i32, edge: i32, surface: i32, core: i32) -> Self {
        let (v, e, s, c) = (vertex, edge, surface, core);
        let posmap = vec![
            v, e, e, v, e, s, s, e, e, s, s, e, v, e, e, v, e, s, s, e, s, c, c, s, s, c, c, s, e,
            s, s, e, e, s, s, e, s, c, c, s, s, c, c, s, e, s, s, e, v, e, e, v, e, s, s, e, e, s,
            s, e, v, e, e, v,
        ];
        return PositionEvaluator { posmap: posmap };
    }

    pub fn simpl_alpha(
        vertex1: i32,
        vertex2: i32,
        vertex3: i32,
        vertex4: i32,
        up_surface: i32,
        bt_surfacce: i32,
        edge1: i32,
        edge2: i32,
        edge3: i32,
        edge4: i32,
        up_core: i32,
        bt_core: i32,
    ) -> Self {
        let (v0, v1, v2, v3, s0, s1, e0, e1, e2, e3, c0, c1) = (
            vertex1,
            vertex2,
            vertex3,
            vertex4,
            bt_surfacce,
            up_surface,
            edge1,
            edge2,
            edge3,
            edge4,
            bt_core,
            up_core,
        );

        let posmap = vec![
            v0, e0, e0, v0, e0, s0, s0, e0, e0, s0, s0, e0, v0, e0, e0, v0, v1, e1, e1, v1, e1, c0,
            c0, e1, e1, c0, c0, e1, v1, e1, e1, v1, v2, e2, e2, v2, e2, c1, c1, e2, e2, c1, c1, e2,
            v2, e2, e2, v2, v3, e3, e3, v3, e3, s1, s1, e3, e3, s1, s1, e3, v3, e3, e3, v3,
        ];

        return PositionEvaluator { posmap: posmap };
    }

    pub fn best() -> Self {
        return PositionEvaluator::simpl(6, 1, 5, 8);
    }
}

impl Evaluator for PositionEvaluator {
    fn eval_func(&self, b: &Board) -> i32 {
        let (mut att, mut def) = b.get_att_def();
        let mut val = 0;
        for i in 0..64 {
            if 1 & att == 1 {
                val += self.posmap[i];
            } else if 1 & def == 1 {
                val -= self.posmap[i]
            }
            att >>= 1;
            def >>= 1;
        }
        return val;
    }
}

pub struct MLEvaluator {
    pub g: Graph,
    loss: usize,
    g_out: usize,
    input: usize,
    t: usize,
}

impl MLEvaluator {
    pub fn new(g: Graph) -> Self {
        return MLEvaluator {
            g: g,
            loss: 0,
            g_out: 0,
            input: 0,
            t: 0,
        };
    }

    pub fn default() -> Self {
        use super::ml::*;
        use super::ml::{funcs::*, optim::*, params::*};
        let mut g = Graph::new();
        g.optimizer = Some(Box::new(MomentumSGD::new(0.01, 0.9)));
        let i1: usize = g.push_placeholder();
        let i2: usize = g.push_placeholder();
        // let t = g.push_placeholder();

        let l1 = Linear::auto(128, 64);
        let l1 = g.add_layer(vec![i1], Box::new(l1));

        let activate = ClippedReLU::default();
        let relu = g.add_layer(vec![l1], Box::new(activate));

        let l2 = Linear::auto(64, 16);
        let l2 = g.add_layer(vec![relu], Box::new(l2));

        let relu2 = g.add_layer(vec![l2], Box::new(ClippedReLU::default()));

        let l3 = Linear::auto(16, 1);
        let l3 = g.add_layer(vec![relu2], Box::new(l3));

        let last = g.add_layer(vec![l3], Box::new(Tanh::new()));

        // t = lambda * result + (1 - lambda) * t_in
        let loss = g.add_layer(vec![last, i2], Box::new(MSE::new()));

        g.set_target(last);
        g.set_placeholder(vec![i1]);

        return MLEvaluator {
            g: g,
            loss: loss,
            g_out: last,
            input: i1,
            t: i2,
        };
    }

    pub fn inference(&self, b: &Board) -> f32 {
        let (att, def) = b.get_att_def();

        let mut att_vec = Vec::new();
        let mut def_vec = Vec::new();
        for i in 0..64 {
            if (att >> i) & 1 == 1 {
                att_vec.push(1.0);
            } else {
                att_vec.push(0.0);
            }

            if (def >> i) & 1 == 1 {
                def_vec.push(1.0);
            } else {
                def_vec.push(0.0);
            }
        }

        let onehot = [att_vec, def_vec].concat();
        let onehot = Tensor::new(onehot, vec![128, 1]);

        let val = self.g.inference(vec![onehot]);

        return val.get_item().unwrap();
    }

    pub fn eval_with_negalpha(&self, b: &Board, depth: u8) -> (u8, f32, i32) {
        return self.eval_with_negalpha_(b, depth, -2.0, 2.0);
    }

    pub fn eval_with_negalpha_(
        &self,
        b: &Board,
        depth: u8,
        alpha: f32,
        beta: f32,
    ) -> (u8, f32, i32) {
        let mut count = 0;
        let actions = b.valid_actions();
        let mut max_val = -2.0;
        let mut max_action: u8 = 16;
        for action in actions.iter() {
            let next_board = &b.next(*action);
            if next_board.is_win() {
                return (*action, 1.0, count);
            } else if next_board.is_draw() {
                return (*action, 0.0, count);
            } else if depth <= 1 {
                let val = -self.inference(next_board);
                count += 1;
                if max_val < val {
                    max_val = val;
                    max_action = *action;
                    if max_val > beta {
                        return (max_action, max_val, count);
                    }
                }
            } else {
                let (_, val, _count) =
                    self.eval_with_negalpha_(next_board, depth - 1, -max_val, -alpha);
                let val = -val;
                count += _count;
                if max_val < val {
                    max_val = val;
                    max_action = *action;
                    if max_val > beta {
                        return (max_action, max_val, count);
                    }
                }
            }
        }
        // println!("[{}]max_val:{}", max_action, max_val);
        return (max_action, max_val, count);
    }

    pub fn train(&mut self) {
        self.g.set_placeholder(vec![self.input, self.t]);
        self.g.set_target(self.loss);
    }

    pub fn eval(&mut self) {
        self.g.set_placeholder(vec![self.input]);
        self.g.set_target(self.g_out);
    }

    pub fn save(&self, s: String) {
        self.g.save(s);
    }

    pub fn load(&mut self, s: String) {
        self.g.load(s);
    }
}

pub fn u2vec(board: u128) -> Vec<f32> {
    let mut att_vec = Vec::new();
    for i in 0..128 {
        if (board >> i) & 1 == 1 {
            att_vec.push(1.0);
        } else {
            att_vec.push(0.0);
        }
    }
    return att_vec;
}

pub fn onehot_vec(n: usize, idx: usize) -> Vec<f32> {
    let mut v = Vec::new();

    for i in 0..n {
        if i == idx {
            v.push(1.0);
        } else {
            v.push(0.0)
        }
    }

    return v;
}

impl Evaluator for MLEvaluator {
    fn eval_func(&self, b: &Board) -> i32 {
        let (att, def) = b.get_att_def();

        let mut att_vec = Vec::new();
        let mut def_vec = Vec::new();
        for i in 0..64 {
            if (att >> i) & 1 == 1 {
                att_vec.push(1.0);
            } else {
                att_vec.push(0.0);
            }

            if (def >> i) & 1 == 1 {
                def_vec.push(1.0);
            } else {
                def_vec.push(0.0);
            }
        }

        let onehot = [att_vec, def_vec].concat();
        let onehot = Tensor::new(onehot, vec![128, 1]);

        let val = self.g.inference(vec![onehot]);

        return (val.get_item().unwrap() * MAX as f32) as i32;
    }
}

impl GetAction for MLEvaluator {
    fn get_action(&self, b: &Board) -> u8 {
        let start = Instant::now();
        let (action, val, count) = self.eval_with_negalpha(b, 4);
        let end = start.elapsed();
        let time = end.as_nanos();
        println!(
            "[MLEvaluator]action:{action}, val:{val}, count:{count}, time:{}",
            time,
        );
        return action;
    }
}

pub fn b2u128(b: &Board) -> u128 {
    let (att, def) = b.get_att_def();
    return (att as u128) | ((def as u128) << 64);
}

pub fn u128_to_b(b: u128) -> Board {
    let mut board = Board::new();
    let black = b as u64;
    let white = (b >> 64) as u64;
    board.black = black;
    board.white = white;

    return board;
}

pub struct NNUE {
    pub g: Graph,
    loss: usize,
    g_out: usize,
    input: usize,
    t: usize,
    pub w1: usize,
    pub w1_size: usize,
    base_vec: Vec<Vec<f32>>,
    depth: usize,
}

impl NNUE {
    pub fn new(g: Graph) -> Self {
        return NNUE {
            g: g,
            loss: 0,
            g_out: 0,
            input: 0,
            t: 0,
            w1: 0,
            w1_size: 0,
            base_vec: Vec::new(),
            depth: 0,
        };
    }

    pub fn default() -> Self {
        use super::ml::*;
        use super::ml::{funcs::*, optim::*, params::*};

        let w1_size = 256;
        let middle_size = 16;

        let mut g = Graph::new();
        g.optimizer = Some(Box::new(MomentumSGD::new(0.01, 0.9)));
        let i1: usize = g.push_placeholder();
        let i2: usize = g.push_placeholder();
        // let t = g.push_placeholder();

        let w1 = MM::auto(128, w1_size);
        let w1 = g.add_layer(vec![i1], Box::new(w1));

        let l1 = Bias::auto(w1_size);
        let l1 = g.add_layer(vec![w1], Box::new(l1));

        let activate = LeaklyReLU::default();
        let relu = g.add_layer(vec![l1], Box::new(activate));

        let l2 = Linear::auto(w1_size, middle_size);
        let l2 = g.add_layer(vec![relu], Box::new(l2));
        let relu2 = g.add_layer(vec![l2], Box::new(LeaklyReLU::default()));

        let l3 = Linear::auto(middle_size, middle_size);
        let l3 = g.add_layer(vec![relu2], Box::new(l3));
        let relu3 = g.add_layer(vec![l3], Box::new(LeaklyReLU::default()));

        let l4 = Linear::auto(middle_size, 1);
        let l4 = g.add_layer(vec![relu3], Box::new(l4));

        // t = lambda * result + (1 - lambda) * t_in
        let sig = g.add_layer(vec![l4], Box::new(Sigmoid::new(1.0)));
        let loss = g.add_layer(vec![sig, i2], Box::new(BinaryCrossEntropy::default()));
        // let loss = g.add_layer(vec![sig, i2], Box::new(MSE::new()));

        g.set_target(sig);
        g.set_placeholder(vec![i1]);

        return NNUE {
            g: g,
            loss: loss,
            g_out: sig,
            input: i1,
            t: i2,
            w1: w1,
            w1_size: w1_size,
            base_vec: Vec::new(),
            depth: 3,
        };
    }

    pub fn inference(&self, b: &Board) -> f32 {
        let onehot = u2vec(Self::b2u128(b));
        let onehot = Tensor::new(onehot, vec![128, 1]);

        let val = self.g.inference(vec![onehot]);

        return val.get_item().unwrap();
    }

    fn b2u128(b: &Board) -> u128 {
        let (att, def) = b.get_att_def();
        return (att as u128) | ((def as u128) << 64);
    }

    pub fn set_inference(&mut self) {
        self.set_before_w1();
        for i in 0..128 {
            let onehot = onehot_vec(128, i);
            let onehot = Tensor::new(onehot, vec![128, 1]);

            let val = self.g.inference(vec![onehot]);
            self.base_vec.push(val.clone().data);
        }
        // println!("{:?}", self.base_vec);
        self.set_after_w1();
    }

    pub fn set_depth(&mut self, depth: usize) {
        self.depth = depth;
    }

    pub fn eval_with_negalpha(&self, b: &Board) -> (u8, f32, i32) {
        let b_hash = Self::b2u128(b);
        let b_vec = self.create_diff_vec(0, b_hash);
        // let start = Instant::now();
        // let result = eval_actor(&m7, &m6, 10, false);
        // let (a, b, c) = self.eval_with_negalpha_1(b, b_hash, b_vec, self.depth as u8, -2.0, 2.0);
        let (a, b, c) =
            self.eval_with_negalpha_(b.clone(), b_hash, b_vec, None, self.depth as u8, -2.0, 2.0);
        return (a, b, c);
    }

    pub fn eval_with_negalpha_(
        &self,
        b: Board,
        b_hash: u128,
        b_vec: Vec<f32>,
        next: Option<(u128, &Vec<f32>)>,
        depth: u8,
        alpha: f32,
        beta: f32,
    ) -> (u8, f32, i32) {
        use std::cmp::Ordering::*;

        let mut count = 0;
        let actions = b.valid_actions();
        let mut max_val = -2.0;
        let mut max_action: u8 = 16;
        let mut next_info: Option<(u128, &Vec<f32>)> = next;
        let mut alpha = alpha;

        if depth <= 1 {
            let mut next_vec;
            for &action in actions.iter() {
                let next_board = &b.next(action);

                let next_hash = Self::b2u128(next_board);

                if next_board.is_win() {
                    return (action, 1.0, count);
                } else if next_board.is_draw() {
                    return (action, 0.5, count);
                }

                let feed_vec: Vec<f32>;
                match next_info {
                    None => {
                        next_vec = self.create_diff_vec(0, next_hash);
                        next_info = Some((next_hash, &next_vec));
                        feed_vec = next_vec.clone();
                    }
                    Some((hash, ref vec)) => {
                        let next_vec_ = self.create_diff_vec(hash, next_hash);

                        feed_vec = next_vec_
                            .iter()
                            .zip(vec.iter())
                            .map(|(a, b)| a + b)
                            .collect();
                    }
                }

                let w1 = Tensor::new(feed_vec, vec![self.w1_size, 1]);

                let val = 1.0 - self.g.inference(vec![w1]).get_item().unwrap();

                count += 1;
                if max_val < val {
                    max_val = val;
                    max_action = action;
                    if max_val > alpha {
                        alpha = max_val;
                        if alpha > beta {
                            return (max_action, max_val, count);
                        }
                    }
                }
            }
        } else {
            let mut nexts = Vec::new();
            let mut next_vec;
            for &action in actions.iter() {
                let next_board = b.next(action);
                let next_hash = Self::b2u128(&next_board);

                if next_board.is_win() {
                    return (action, 1.0, count);
                } else if next_board.is_draw() {
                    return (action, 0.5, count);
                }

                let feed_vec: Vec<f32>;
                match next_info {
                    None => {
                        next_vec = self.create_diff_vec(0, next_hash);
                        next_info = Some((next_hash, &next_vec));
                        feed_vec = next_vec.clone();
                    }
                    Some((hash, ref vec)) => {
                        let next_vec_ = self.create_diff_vec(hash, next_hash);
                        feed_vec = next_vec_
                            .iter()
                            .zip(vec.iter())
                            .map(|(a, b)| a + b)
                            .collect();
                    }
                }
                let w1 = Tensor::new(feed_vec.clone(), vec![self.w1_size, 1]);

                let val = 1.0 - self.g.inference(vec![w1]).get_item().unwrap();

                nexts.push((action, next_board, next_hash, feed_vec, val))
            }
            nexts.sort_by(|a, b| {
                if a.4 < b.4 {
                    return Greater;
                } else {
                    return Less;
                }
            });

            for (action, next_board, next_hash, next_vec, val) in nexts {
                let (_, val, _count) = self.eval_with_negalpha_(
                    next_board,
                    next_hash,
                    next_vec,
                    Some((b_hash, &b_vec)),
                    depth - 1,
                    1.0 - beta,
                    1.0 - alpha,
                );
                let val = -0.999 * (val - 0.5) + 0.5;

                count += _count;
                if max_val < val {
                    max_val = val;
                    max_action = action;
                    if max_val > alpha {
                        alpha = max_val;
                        if alpha > beta {
                            return (max_action, max_val, count);
                        }
                    }
                }
            }
        }
        return (max_action, max_val, count);
    }

    fn create_diff_vec(&self, a: u128, b: u128) -> Vec<f32> {
        // a -> b を考える
        let minus = a & !b;
        let plus = b & !a;

        let mut minus_vec = vec![0.0; self.w1_size];
        let mut plus_vec = vec![0.0; self.w1_size];

        for i in 0..128 {
            if (minus >> i) & 1 == 1 {
                for j in 0..self.w1_size {
                    minus_vec[j] += self.base_vec[i][j];
                }
                continue;
            }
            if (plus >> i) & 1 == 1 {
                for j in 0..self.w1_size {
                    plus_vec[j] += self.base_vec[i][j];
                }
            }
        }

        for i in 0..self.w1_size {
            plus_vec[i] -= minus_vec[i];
        }

        return plus_vec;
    }

    pub fn train(&mut self) {
        self.g.set_placeholder(vec![self.input, self.t]);
        self.g.set_target(self.loss);
    }

    pub fn eval(&mut self) {
        self.g.set_placeholder(vec![self.input]);
        self.g.set_target(self.g_out);
    }

    pub fn set_before_w1(&mut self) {
        self.g.set_placeholder(vec![self.input]);
        self.g.set_target(self.w1);
    }

    pub fn set_after_w1(&mut self) {
        self.g.set_placeholder(vec![self.w1]);
        self.g.set_target(self.g_out)
    }

    pub fn save(&self, s: String) {
        self.g.save(s);
    }

    pub fn load(&mut self, s: String) {
        self.g.load(s);
    }
}

impl GetAction for NNUE {
    fn get_action(&self, b: &Board) -> u8 {
        let start = Instant::now();
        let (action, val, count) = self.eval_with_negalpha(b);
        let end = start.elapsed();
        let hoge: Box<dyn Evaluator> = Box::new(CoEvaluator::best());
        let (action_, val_, count) = negalpha(b, 3, -MAX - 1, MAX + 1, &hoge);
        let time = end.as_nanos();
        // println!(
        //     "[NNUE]action:{action}-{action_}, val:{val}, val_:{val_}, count:{count}, time:{}",
        //     time,
        // );
        let res = mate_check_horizontal(b);
        if let Some((flag, action)) = res {
            // println!("{flag}");
            return action;
        }
        return action;
    }
}

impl Analyzer for NNUE {
    fn analyze_eval(&self, b: &Board) -> f32 {
        let (action, val, count) = self.eval_with_negalpha(b);
        return val;
    }
}

pub struct RowEvaluator {
    w_row3: i32,
    w_row2: i32,
    w_row1: i32,
}

impl RowEvaluator {
    pub fn new() -> Self {
        return RowEvaluator {
            w_row1: 1,
            w_row2: 1,
            w_row3: 1,
        };
    }

    pub fn best() -> Self {
        RowEvaluator::from(2, 5, 12)
    }

    pub fn from(w1: i32, w2: i32, w3: i32) -> Self {
        return RowEvaluator {
            w_row3: w3,
            w_row2: w2,
            w_row1: w1,
        };
    }
}

impl Evaluator for RowEvaluator {
    fn eval_func(&self, b: &Board) -> i32 {
        let (att, def) = b.get_att_def();
        let blank = !att & !def;

        let att3 = count_3row(att, blank) as i32;
        let def3 = count_3row(def, blank) as i32;

        let att2 = count_2row(att, blank) as i32;
        let def2 = count_2row(def, blank) as i32;

        let att1 = count_1row(att, blank) as i32;
        let def1 = count_1row(def, blank) as i32;

        // pprint_board(b);
        // println!("{att}, {def}");

        return self.w_row3 * (att3 - def3)
            + self.w_row2 * (att2 - def2)
            + self.w_row1 * (att1 - def1);
    }
}

pub struct NullEvaluator {}
impl NullEvaluator {
    pub fn new() -> Self {
        return NullEvaluator {};
    }
}

impl Evaluator for NullEvaluator {
    fn eval_func(&self, b: &Board) -> i32 {
        return 0;
    }
}

pub struct CoEvaluator {
    a: Box<dyn Evaluator>,
    b: Box<dyn Evaluator>,
    a_weight: i32,
    b_weight: i32,
}

impl CoEvaluator {
    pub fn new(a: Box<dyn Evaluator>, b: Box<dyn Evaluator>, a_weight: i32, b_weight: i32) -> Self {
        return CoEvaluator {
            a: a,
            b: b,
            a_weight: a_weight,
            b_weight: b_weight,
        };
    }

    pub fn best() -> Self {
        let a = RowEvaluator::best();
        let b = PositionEvaluator::best();
        return CoEvaluator {
            a: Box::new(a),
            b: Box::new(b),
            a_weight: 3,
            b_weight: 1,
        };
    }
}

impl Evaluator for CoEvaluator {
    fn eval_func(&self, b: &Board) -> i32 {
        let a_score = self.a.eval_func(b);
        let b_score = self.b.eval_func(b);

        let score = 10 * (a_score * self.a_weight + b_score * self.b_weight)
            / (self.a_weight + self.b_weight);

        return score;
    }
}

pub struct RandomEvaluator {}

impl RandomEvaluator {
    pub fn new() -> Self {
        return RandomEvaluator {};
    }
}

impl Evaluator for RandomEvaluator {
    fn eval_func(&self, b: &Board) -> i32 {
        let mut rng = rand::thread_rng();
        let u: usize = rng.gen();
        return (u % 1300) as i32;
    }
}

pub struct MateNegAlpha {
    main_eval: Box<dyn Evaluator>,
    depth: u8,
}

impl MateNegAlpha {
    pub fn new(main_eval: Box<dyn Evaluator>, depth: u8) -> Self {
        return MateNegAlpha {
            main_eval: main_eval,
            depth: depth,
        };
    }

    pub fn eval_with_negalpha(&self, b: &Board) -> (u8, i32, i32) {
        let res = mate_check_horizontal(b);
        if let Some((flag, action)) = res {
            if flag {
                return (action, MAX, 0);
            }
        }
        let (action, v, count) = negalpha(b, self.depth, -MAX - 1, MAX + 1, &self.main_eval);

        return (action, v, count);
    }
}
impl GetAction for MateNegAlpha {
    fn get_action(&self, b: &Board) -> u8 {
        use proconio::input;
        let res = mate_check_horizontal(b);
        if let Some((flag, action)) = res {
            if flag {
                println!("mate")
            }
            println!("->[{action}]");
            // return action;
        } else {
            let (action, v, count) = negalpha(b, self.depth, -MAX - 1, MAX + 1, &self.main_eval);
            println!("action:{action}, val:{v}, count:{count}");
        }
        input! {
            action: u8
        }
        return action;
    }
}

pub struct OnnxEvaluator {
    session: Session,
}

impl OnnxEvaluator {
    pub fn new(file_path: &str) -> Self {
        let environment = Environment::builder().build().unwrap().into_arc();
        let session = SessionBuilder::new(&environment)
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(4)
            .unwrap()
            .with_model_from_file(file_path)
            .unwrap();
        return OnnxEvaluator { session: session };
    }

    pub fn inference(&self, b: &Board) -> f32 {
        let input_vec = u2vec(b2u128(b));
        let inputs = ndarray::Array::from_vec(input_vec);
        let inputs = CowArray::from(inputs.slice(s![..]).into_dyn());
        let inputs = vec![ort::Value::from_array(self.session.allocator(), &inputs).unwrap()];

        let output = self.session.run(inputs).unwrap();
        let output = output[0].try_extract::<f32>().unwrap();
        let output = output.view();

        return output[0];
    }
}

impl EvaluatorF for OnnxEvaluator {
    fn eval_func(&self, b: &Board) -> f32 {
        return self.inference(b);
    }
}

impl Analyzer for OnnxEvaluator {
    fn analyze_eval(&self, b: &Board) -> f32 {
        return self.eval_func(b);
    }
}

impl EvalAndAnalyze for OnnxEvaluator {}
