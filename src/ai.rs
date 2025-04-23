pub mod line;
pub mod mpc;
pub mod pattern;

use crate::board::{
    self, count_1row, count_2row, count_3row, get_reach_mask, mate_check_horizontal, pprint_board,
};
use crate::train::Transition;

use super::board::{Board, GetAction};
use super::ml::{Graph, Tensor};
// use ndarray::{s, CowArray};
// use ort::{Environment, GraphOptimizationLevel, Session, SessionBuilder};
use anyhow::{Ok, Result};
use rand::Rng;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32;
use std::time::Instant;

pub const MAX: i32 = 1600;
const F32_INVERSE_LAMBDA: f32 = 0.999;
const F32_INVERSE_BIAS: f32 = 0.9995;

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
    e: &Box<dyn EvaluatorF>,
) -> (u8, f32, i32) {
    //
    let mut count = 0;
    let actions = b.valid_actions();
    let mut max_val = -2.0;
    let mut max_actions = Vec::new();
    let mut alpha = alpha;

    if depth <= 1 {
        for action in actions.iter() {
            let next_board = &b.next(*action);
            if next_board.is_win() {
                return (*action, 1.0, count);
            } else if next_board.is_draw() {
                return (*action, 0.5, count);
            }
            let val = 1.0 - e.eval_func_f32(next_board);
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
        let mut action_nb_vals: Vec<(u8, Board, f32)> = Vec::new();

        for action in actions.into_iter() {
            let next_board = b.next(action);
            if next_board.is_win() {
                return (action, 1.0, count);
            } else if next_board.is_draw() {
                return (action, 0.5, count);
            }

            let val = 1.0 - e.eval_func_f32(&next_board);
            action_nb_vals.push((action, next_board, val));
        }

        action_nb_vals.sort_by(|a, b| {
            // a.2.cmp(&b.2).reverse();
            b.2.partial_cmp(&a.2).unwrap()
        });

        for (action, next_board, _) in action_nb_vals {
            let (_, val, _count) = negalphaf(&next_board, depth - 1, 1.0 - beta, 1.0 - alpha, e);
            count += 1 + _count;
            let val = 0.9995 - 0.999 * val;
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

#[derive(Clone, Copy)]
pub enum Fail {
    High(f32),
    Low(f32),
    Ex(f32),
}

impl Fail {
    fn inverse(&self) -> Self {
        use Fail::*;
        match self {
            High(beta) => Low(F32_INVERSE_BIAS - F32_INVERSE_LAMBDA * beta),
            Low(alpha) => High(F32_INVERSE_BIAS - F32_INVERSE_LAMBDA * alpha),
            Ex(val) => Ex(F32_INVERSE_BIAS - F32_INVERSE_LAMBDA * val),
        }
    }

    fn is_fail(&self) -> bool {
        use Fail::*;
        match self {
            Ex(_) => false,
            _ => true,
        }
    }

    fn is_equal(&self, x: f32) -> bool {
        match self {
            Fail::Ex(val) => *val == x,
            _ => false,
        }
    }

    fn f32_minus(&self, x: f32) -> Self {
        use Fail::*;

        match self {
            Ex(val) => Ex(x - val),
            High(val) => Low(x - val),
            Low(val) => High(x - val),
        }
    }

    fn get_val(&self) -> f32 {
        use Fail::*;

        match self {
            Ex(val) => *val,
            High(val) => *val,
            Low(val) => *val,
        }
    }

    fn is_fail_low(&self) -> bool {
        match self {
            Fail::Low(_) => true,
            _ => false,
        }
    }

    fn is_fail_high(&self) -> bool {
        match self {
            Fail::High(_) => true,
            _ => false,
        }
    }

    fn get_exval(&self) -> Option<f32> {
        match self {
            Fail::Ex(x) => Some(*x),
            _ => None,
        }
    }

    fn to_string(&self) -> String {
        use Fail::*;
        match self {
            High(x) => format!("High({x})"),
            Low(x) => format!("Low({x})"),
            Ex(x) => format!("Ex({x})"),
        }
    }
}

fn print_blank(n: u8) {
    for i in 0..n {
        print!(" ");
    }
}

pub fn negalphaf_hash(
    b: &Board,
    depth: u8,
    alpha: f32,
    beta: f32,
    hashmap: &mut HashMap<u128, Fail>,
    e: &Box<dyn EvaluatorF>,
) -> (u8, Fail, i32) {
    use Fail::*;

    let mut count = 0;
    let actions = b.valid_actions();
    let mut max_val = -2.0;
    let mut max_actions = Vec::new();
    let mut alpha = alpha;

    // pprint_board(b);
    // print_blank(5 - depth);
    // println!("[depth:{depth}]alpha:{alpha}, beta:{beta}");

    if depth <= 1 {
        for action in actions.iter() {
            let next_board = &b.next(*action);
            // let hash = next_board.hash();
            let hash = b2u128(next_board);
            let map_val = hashmap.get(&hash);
            let val;
            match map_val {
                Some(&old_val) => {
                    if old_val.is_equal(0.0) {
                        return (*action, Ex(1.0), count);
                    }
                    val = 1.0 - old_val.get_val();
                }
                None => {
                    if next_board.is_win() {
                        hashmap.insert(hash, Ex(0.0));
                        return (*action, Ex(1.0), count);
                    } else if next_board.is_draw() {
                        hashmap.insert(hash, Ex(0.5));
                        return (*action, Ex(0.5), count);
                    }
                    let next_val = e.eval_func_f32(next_board);
                    val = 1.0 - next_val;
                    hashmap.insert(hash, Ex(next_val));
                }
            }
            if max_val < val {
                max_val = val;
                max_actions = vec![*action];
                if max_val > alpha {
                    alpha = max_val;
                    if alpha > beta {
                        // println!("[{depth}]->max_val:{max_val}");
                        return (*action, High(max_val), count);
                    }
                }
            } else if max_val == val {
                max_actions.push(*action);
            }
        }
    } else {
        let mut action_nb_vals: Vec<(u8, Board, f32, u128, Option<Fail>)> = Vec::new();

        for action in actions.into_iter() {
            let next_board = b.next(action);
            // let hash = next_board.hash();
            let hash = b2u128(&next_board);
            let map_val = hashmap.get(&hash);

            match map_val {
                Some(&old_val) => {
                    if old_val.is_equal(0.0) {
                        return (action, Ex(1.0), count);
                    }
                    action_nb_vals.push((
                        action,
                        next_board,
                        old_val.f32_minus(1.0).get_val(),
                        hash,
                        Some(old_val.inverse()),
                    ));
                }
                None => {
                    if next_board.is_win() {
                        hashmap.insert(hash, Ex(0.0));
                        // print_blank(5 - depth);
                        // println!("[depth:{depth}]action:{action}, win");
                        return (action, Ex(1.0), count);
                    } else if next_board.is_draw() {
                        hashmap.insert(hash, Ex(0.5));
                        // print_blank(5 - depth);
                        // println!("[depth:{depth}]action:{action}, draw");
                        return (action, Ex(0.5), count);
                    }
                    let val = 1.0 - e.eval_func_f32(&next_board);
                    action_nb_vals.push((action, next_board, val, hash, None));
                }
            }
        }

        action_nb_vals.sort_by(|a, b| {
            // a.2.cmp(&b.2).reverse();
            b.2.partial_cmp(&a.2).unwrap()
        });

        for (action, next_board, old_val, hash, hit) in action_nb_vals {
            let val;
            // println!("[depth:{depth}], action:{action}, alpha:{alpha}, beta:{beta}",);
            if let Some(fail_val) = hit {
                // if depth > 2 {
                //     println!("[{depth}]hit, {}", fail_val.to_string());
                // }
                match fail_val {
                    High(x) => {
                        if beta < x {
                            return (action, High(x), count);
                        } else {
                            let new_alpha = x.max(alpha);
                            let (_, _val, _count) = negalphaf_hash(
                                &next_board,
                                depth - 1,
                                1.0 - beta,
                                1.0 - new_alpha,
                                hashmap,
                                e,
                            );
                            count += _count;
                            hashmap.insert(hash, _val);
                            let _val = _val.inverse();
                            if _val.is_fail_high() {
                                return (action, High(beta), count);
                            }
                            val = _val.get_val();
                        }
                    }
                    Low(x) => {
                        if alpha > x {
                            continue;
                        } else {
                            let new_beta = x.min(beta);
                            // fial_low(alpha) or fail_ex(val) < new_beta
                            let (_, _val, _count) = negalphaf_hash(
                                &next_board,
                                depth - 1,
                                1.0 - new_beta,
                                1.0 - alpha,
                                hashmap,
                                e,
                            );
                            hashmap.insert(hash, _val);
                            let _val = _val.inverse();
                            if _val.is_fail_low() {
                                continue;
                            }
                            val = _val.get_val();
                        }
                    }
                    Ex(x) => {
                        // val = F32_INVERSE_BIAS - F32_INVERSE_BIAS * x;
                        val = x;
                    }
                }
            } else {
                let (_, _val, _count) =
                    negalphaf_hash(&next_board, depth - 1, 1.0 - beta, 1.0 - alpha, hashmap, e);
                hashmap.insert(hash, _val);
                count += 1 + _count;
                let _val = _val.inverse();
                // for i in 0..(6 - depth) {
                //     print!(" ");
                // }
                // println!(
                //     "[depth:{depth}], action:{action}, val:{}, alpha:{alpha}, beta:{beta}",
                //     _val.to_string()
                // );

                match _val {
                    High(x) => return (action, High(x), count),
                    Low(_) => continue,
                    Ex(x) => {
                        val = x;
                    }
                }
            }
            if max_val < val {
                max_val = val;
                max_actions = vec![action];
                if max_val > alpha {
                    alpha = max_val;
                    if alpha > beta {
                        // println!("[{depth}]->max_val:{max_val}");
                        return (action, High(max_val), count);
                    }
                }
            } else if max_val == val {
                max_actions.push(action);
            }
        }
    }
    let mut rng = rand::thread_rng();
    if max_actions.len() == 0 {
        return (201, Low(alpha), count);
    }

    return (
        max_actions[rng.gen::<usize>() % max_actions.len()],
        Ex(max_val),
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
        let start = Instant::now();
        let (action, v, count) = negalpha(b, self.depth, -MAX - 1, MAX + 1, &self.evaluator);
        // pprint_board(b);
        let t = start.elapsed().as_nanos();

        if cfg!(feature = "render") {
            println!("action:{action}, val:{v}, count:{count}/{t}");
        }

        return action;
    }
}

pub trait EvalAndAnalyze: EvaluatorF + Analyzer {}
pub trait EvalAndActF {
    fn eval_and_act(&self, b: &Board) -> (u8, f32);
}

pub struct NegAlphaF {
    evaluator: Box<dyn EvaluatorF>,
    depth: u8,
    pub hashmap: bool,
}

impl NegAlphaF {
    pub fn new(e: Box<dyn EvaluatorF>, depth: u8) -> Self {
        return NegAlphaF {
            evaluator: e,
            depth: depth,
            hashmap: false,
        };
    }

    pub fn eval_with_negalpha_hash(&self, b: &Board) -> (u8, f32, i32) {
        let mut hashmap = HashMap::new();
        let (action2, val2, count2) =
            negalphaf_hash(b, self.depth, -2.0, 2.0, &mut hashmap, &self.evaluator);

        return (action2, val2.get_exval().unwrap(), count2);
    }

    pub fn eval_with_negalpha(&self, b: &Board) -> (u8, f32, i32) {
        // for &action in actions.iter() {
        //     let next_b = b.next(action);
        //     let (_, val, _) = negalphaf(&next_b, self.depth - 1, -2.0, 2.0, &self.evaluator);
        //     let val = 1.0 - val;
        //     println!("[{}]:{}", action, val);
        // }
        let (att, def) = b.get_att_def();
        let stone = (att | def).count_ones() as usize;
        if cfg!(feature = "render") {
            println!("att:{att}, def:{def}");
            let start = Instant::now();
            let (action, val, count) = negalphaf(b, self.depth, -2.0, 2.0, &self.evaluator);
            let t = start.elapsed().as_nanos();
            let mut hashmap = HashMap::new();
            let start = Instant::now();
            let (action2, val2, count2) =
                negalphaf_hash(b, self.depth, -2.0, 2.0, &mut hashmap, &self.evaluator);
            let t2 = start.elapsed().as_nanos();
            println!(
                "action:{action}-{action2}, val:{val}-{}, count:{count}/{t}-{count2}/{t2}, rate:{}%",
                val2.get_exval().unwrap(),
                t2 * 100 / t
            );
        }

        if self.hashmap {
            return self.eval_with_negalpha_hash(b);
        }

        let (a, b, c) = negalphaf(b, self.depth, -2.0, 2.0, &self.evaluator);
        return (a, b, c);
    }
}

impl GetAction for NegAlphaF {
    fn get_action(&self, b: &Board) -> u8 {
        let (action, _, _) = self.eval_with_negalpha(b);
        return action;
    }
}

impl EvalAndActF for NegAlphaF {
    fn eval_and_act(&self, b: &Board) -> (u8, f32) {
        let (action, val, _) = self.eval_with_negalpha(b);
        return (action, val);
    }
}

pub trait Evaluator {
    fn eval_func(&self, b: &Board) -> i32;
}

pub trait EvaluatorF {
    fn eval_func_f32(&self, b: &Board) -> f32;
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
        let middle_size = 32;

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
            self.base_vec.push(val.data);
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
        if cfg!(feature = "render") {
            println!(
                "[NNUE]action:{action}-{action_}, val:{val}, val_:{val_}, count:{count}, time:{}",
                time,
            );
        }
        let res = mate_check_horizontal(b);
        if let Some((flag, action)) = res {
            if cfg!(feature = "render") {
                println!("{flag}");
            }
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

impl EvaluatorF for CoEvaluator {
    fn eval_func_f32(&self, b: &Board) -> f32 {
        let val = self.eval_func(b) as f32;
        return 1.0 / (1.0 + (-val / 400.0).exp());
    }
}

#[derive(Clone)]
pub struct LineEvaluator {
    pub w_float_3_line: [f32; 32],
    pub w_ground_3_line: [f32; 32],
    pub w_float_2_line: [f32; 64],
    pub w_ground_2_line: [f32; 64],
    pub w_float_1_line: [f32; 96],
    pub w_ground_1_line: [f32; 96],
    pub w_dif_float_3_line: [f32; 64],
    pub w_dif_ground_3_line: [f32; 64],
    pub w_dif_float_2_line: [f32; 128],
    pub w_dif_ground_2_line: [f32; 128],
    pub w_dif_float_1_line: [f32; 192],
    pub w_dif_ground_1_line: [f32; 192],
    pub w_pos_3_line: [f32; 64],
    pub w_pos_2_line: [f32; 64],
    pub w_pos_1_line: [f32; 64],
    pub w_dpos_3_line: [f32; 64],
    pub w_dpos_2_line: [f32; 64],
    pub w_dpos_1_line: [f32; 64],
    pub w_bf_3_line: [f32; 64],
    pub w_bf_2_line: [f32; 64],
    pub w_bf_1_line: [f32; 64],
    pub w_bg_3_line: [f32; 64],
    pub w_bg_2_line: [f32; 64],
    pub w_bg_1_line: [f32; 64],
    pub w_dbf_3_line: [f32; 64],
    pub w_dbf_2_line: [f32; 64],
    pub w_dbf_1_line: [f32; 64],
    pub w_dbg_3_line: [f32; 64],
    pub w_dbg_2_line: [f32; 64],
    pub w_dbg_1_line: [f32; 64],
    pub w_pos: [f32; 64],
    pub w_dpos: [f32; 64],
    pub w_trap_3_num: [f32; 13],
    pub bias: f32,
}

type LineMaskBundle = (
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
);

pub fn acum_or(bundle: LineMaskBundle) -> u64 {
    return bundle.0
        | bundle.1
        | bundle.2
        | bundle.3
        | bundle.4
        | bundle.5
        | bundle.6
        | bundle.7
        | bundle.8
        | bundle.9
        | bundle.10
        | bundle.11
        | bundle.12;
}

pub fn acum_mask_bundle(bundle: LineMaskBundle) -> u32 {
    return bundle.0.count_ones()
        + bundle.1.count_ones()
        + bundle.2.count_ones()
        + bundle.3.count_ones()
        + bundle.4.count_ones()
        + bundle.5.count_ones()
        + bundle.6.count_ones()
        + bundle.7.count_ones()
        + bundle.8.count_ones()
        + bundle.9.count_ones()
        + bundle.10.count_ones()
        + bundle.11.count_ones()
        + bundle.12.count_ones();
}
pub fn apply_mask_bundle(bundle: LineMaskBundle, mask: u64) -> LineMaskBundle {
    return (
        bundle.0 & mask,
        bundle.1 & mask,
        bundle.2 & mask,
        bundle.3 & mask,
        bundle.4 & mask,
        bundle.5 & mask,
        bundle.6 & mask,
        bundle.7 & mask,
        bundle.8 & mask,
        bundle.9 & mask,
        bundle.10 & mask,
        bundle.11 & mask,
        bundle.12 & mask,
    );
}

impl LineEvaluator {
    pub fn new() -> Self {
        return LineEvaluator {
            w_float_3_line: [0.0; 32],
            w_ground_3_line: [0.0; 32],
            w_float_2_line: [0.0; 64],
            w_ground_2_line: [0.0; 64],
            w_float_1_line: [0.0; 96],
            w_ground_1_line: [0.0; 96],
            w_dif_float_3_line: [0.0; 64],
            w_dif_ground_3_line: [0.0; 64],
            w_dif_float_2_line: [0.0; 128],
            w_dif_ground_2_line: [0.0; 128],
            w_dif_float_1_line: [0.0; 192],
            w_dif_ground_1_line: [0.0; 192],
            w_pos_3_line: [0.0; 64],
            w_pos_2_line: [0.0; 64],
            w_pos_1_line: [0.0; 64],
            w_dpos_3_line: [0.0; 64],
            w_dpos_2_line: [0.0; 64],
            w_dpos_1_line: [0.0; 64],
            w_bf_3_line: [0.0; 64],
            w_bg_3_line: [0.0; 64],
            w_bf_2_line: [0.0; 64],
            w_bg_2_line: [0.0; 64],
            w_bf_1_line: [0.0; 64],
            w_bg_1_line: [0.0; 64],
            w_dbf_3_line: [0.0; 64],
            w_dbg_3_line: [0.0; 64],
            w_dbf_2_line: [0.0; 64],
            w_dbg_2_line: [0.0; 64],
            w_dbf_1_line: [0.0; 64],
            w_dbg_1_line: [0.0; 64],
            w_pos: [0.0; 64],
            w_dpos: [0.0; 64],
            w_trap_3_num: [0.0; 13],
            bias: 0.0,
        };
    }
    // pub fn from(
    //     wf1: [f32; 96],
    //     wf2: [f32; 64],
    //     wf3: [f32; 32],
    //     wg1: [f32; 96],
    //     wg2: [f32; 64],
    //     wg3: [f32; 32],
    //     wp1: [f32; 64],
    //     wp2: [f32; 64],
    //     wp3: [f32; 64],
    //     wb1: [f32; 64],
    //     wb2: [f32; 64],
    //     wb3: [f32; 64],

    //     bias: f32,
    // ) -> Self {
    //     return LineEvaluator {
    //         w_float_3_line: wf3,
    //         w_ground_3_line: wg3,
    //         w_float_2_line: wf2,
    //         w_ground_2_line: wg2,
    //         w_float_1_line: wf1,
    //         w_ground_1_line: wg1,
    //         w_pos_1_line: wp1,
    //         w_pos_2_line: wp2,
    //         w_pos_3_line: wp3,
    //         w_bf_1_line: wbf1,
    //         w_bg_1_line: wbg1,
    //         w_bf_2_line: wbf2,
    //         w_bg_2_line: wbg2,
    //         w_bf_3_line: wbf3,
    //         w_bg_3_line: wbg3,
    //         w_dbf_1_line: wdbf1,
    //         w_dbg_1_line: wdbg1,
    //         w_dbf_2_line: wdbf2,
    //         w_dbg_2_line: wdbg2,
    //         w_dbf_3_line: wdbf3,
    //         w_dbg_3_line: wdbg3,
    //         bias: bias,
    //     };
    // }

    pub fn _analyze_line(
        a1: u64,
        a2: u64,
        a3: u64,
        a4: u64,
        b1: u64,
        b2: u64,
        b3: u64,
        b4: u64,
        mask: u64,
        magic: u64,
    ) -> (u64, u64, u64) {
        return (
            ((b1 & b2 & b3 & a4 | b1 & b2 & a3 & b4 | b1 & a2 & b3 & b4 | a1 & b2 & b3 & b4)
                & mask)
                * magic,
            ((a1 & a2 & b3 & b4
                | a1 & b2 & a3 & b4
                | a1 & b2 & b3 & a4
                | b1 & a2 & a3 & b4
                | b1 & a2 & b3 & a4
                | b1 & b2 & a3 & a4)
                & mask)
                * magic,
            ((a1 & a2 & a3 & b4 | a1 & a2 & b3 & a4 | a1 & b2 & a3 & a4 | b1 & a2 & a3 & a4)
                & mask)
                * magic,
        );
    }
    pub fn analyze_board(
        a: u64,
        d: u64,
    ) -> (
        LineMaskBundle,
        LineMaskBundle,
        LineMaskBundle,
        LineMaskBundle,
        LineMaskBundle,
        LineMaskBundle,
    ) {
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
            LineEvaluator::_analyze_line(a, a1, a2, a3, b, b1, b2, b3, 0x1111_1111_1111_1111, 0xf);
        let (x1, x2, x3, px1, px2, px3) = (x1 & b, x2 & b, x3 & b, x1 & a, x2 & a, x3 & a);

        let (y1, y2, y3) = LineEvaluator::_analyze_line(
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
        let (y1, y2, y3, py1, py2, py3) = (y1 & b, y2 & b, y3 & b, y1 & a, y2 & a, y3 & a);

        let (z1, z2, z3) = LineEvaluator::_analyze_line(
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
        let (xy1, xy2, xy3) = LineEvaluator::_analyze_line(
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
        let (xy1_, xy2_, xy3_) = LineEvaluator::_analyze_line(
            a,
            a3,
            a6,
            a9,
            b,
            b3,
            b6,
            b9,
            0x0008_0008_0008_0008,
            0x249,
        );
        let (xz1, xz2, xz3) = LineEvaluator::_analyze_line(
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
        let (xz1_, xz2_, xz3_) = LineEvaluator::_analyze_line(
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
        let (yz1, yz2, yz3) = LineEvaluator::_analyze_line(
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
        let (yz1_, yz2_, yz3_) = LineEvaluator::_analyze_line(
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
        let (xyz11, xyz12, xyz13) = LineEvaluator::_analyze_line(
            a,
            a21,
            a42,
            a63,
            b,
            b21,
            b42,
            b63,
            0x1,
            0x8000_0400_0020_0001,
        );
        let (xyz21, xyz22, xyz23) = LineEvaluator::_analyze_line(
            a,
            a19,
            a38,
            a57,
            b,
            b19,
            b38,
            b57,
            0x8,
            0x0200_0040_0008_0001,
        );
        let (xyz31, xyz32, xyz33) = LineEvaluator::_analyze_line(
            a,
            a13,
            a26,
            a39,
            b,
            b13,
            b26,
            b39,
            0x1000,
            0x0000_0080_0400_2001,
        );
        let (xyz41, xyz42, xyz43) = LineEvaluator::_analyze_line(
            a,
            a11,
            a22,
            a33,
            b,
            b11,
            b22,
            b33,
            0x8000,
            0x0000_0002_0040_0801,
        );

        let (z1, z2, z3, pz1, pz2, pz3) = (z1 & b, z2 & b, z3 & b, z1 & a, z2 & a, z3 & a);
        let (xy1, xy2, xy3, pxy1, pxy2, pxy3) =
            (xy1 & b, xy2 & b, xy3 & b, xy1 & a, xy2 & a, xy3 & a);
        let (xy1_, xy2_, xy3_, pxy1_, pxy2_, pxy3_) =
            (xy1_ & b, xy2_ & b, xy3_ & b, xy1_ & a, xy2_ & a, xy3_ & a);
        let (yz1, yz2, yz3, pyz1, pyz2, pyz3) =
            (yz1 & b, yz2 & b, yz3 & b, yz1 & a, yz2 & a, yz3 & a);
        let (yz1_, yz2_, yz3_, pyz1_, pyz2_, pyz3_) =
            (yz1_ & b, yz2_ & b, yz3_ & b, yz1_ & a, yz2_ & a, yz3_ & a);
        let (xz1, xz2, xz3, pxz1, pxz2, pxz3) =
            (xz1 & b, xz2 & b, xz3 & b, xz1 & a, xz2 & a, xz3 & a);
        let (xz1_, xz2_, xz3_, pxz1_, pxz2_, pxz3_) =
            (xz1_ & b, xz2_ & b, xz3_ & b, xz1_ & a, xz2_ & a, xz3_ & a);
        let (xyz11, xyz12, xyz13, pxyz11, pxyz12, pxyz13) = (
            xyz11 & b,
            xyz12 & b,
            xyz13 & b,
            xyz11 & a,
            xyz12 & a,
            xyz13 & a,
        );
        let (xyz21, xyz22, xyz23, pxyz21, pxyz22, pxyz23) = (
            xyz21 & b,
            xyz22 & b,
            xyz23 & b,
            xyz21 & a,
            xyz22 & a,
            xyz23 & a,
        );
        let (xyz31, xyz32, xyz33, pxyz31, pxyz32, pxyz33) = (
            xyz31 & b,
            xyz32 & b,
            xyz33 & b,
            xyz31 & a,
            xyz32 & a,
            xyz33 & a,
        );
        let (xyz41, xyz42, xyz43, pxyz41, pxyz42, pxyz43) = (
            xyz41 & b,
            xyz42 & b,
            xyz43 & b,
            xyz41 & a,
            xyz42 & a,
            xyz43 & a,
        );

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
            (
                px1, py1, pz1, pxy1, pxy1_, pyz1, pyz1_, pxz1, pxz1_, pxyz11, pxyz21, pxyz31,
                pxyz41,
            ),
            (
                px2, py2, pz2, pxy2, pxy2_, pyz2, pyz2_, pxz2, pxz2_, pxyz12, pxyz22, pxyz32,
                pxyz42,
            ),
            (
                px3, py3, pz3, pxy3, pxy3_, pyz3, pyz3_, pxz3, pxz3_, pxyz13, pxyz23, pxyz33,
                pxyz43,
            ),
        );
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
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        usize,
        u64,
        u64,
    ) {
        let (att, def) = b.get_att_def();
        let (a1, a2, a3, pa1, pa2, pa3) = LineEvaluator::analyze_board(att, def);
        let (d1, d2, d3, pd1, pd2, pd3) = LineEvaluator::analyze_board(def, att);
        let stone = att | def;
        let ground = !stone & (stone << 16 | 0xffff);
        let float = !stone ^ ground;
        let a1_float = acum_mask_bundle(apply_mask_bundle(a1, float)) as usize;
        let a1_ground = acum_mask_bundle(apply_mask_bundle(a1, ground)) as usize;
        let a2_float = acum_mask_bundle(apply_mask_bundle(a2, float)) as usize;
        let a2_ground = acum_mask_bundle(apply_mask_bundle(a2, ground)) as usize;
        let a3_float = acum_mask_bundle(apply_mask_bundle(a3, float)) as usize;
        let a3_ground = acum_mask_bundle(apply_mask_bundle(a3, ground)) as usize;
        let d1_float = acum_mask_bundle(apply_mask_bundle(d1, float)) as usize;
        let d1_ground = acum_mask_bundle(apply_mask_bundle(d1, ground)) as usize;
        let d2_float = acum_mask_bundle(apply_mask_bundle(d2, float)) as usize;
        let d2_ground = acum_mask_bundle(apply_mask_bundle(d2, ground)) as usize;
        let d3_float = acum_mask_bundle(apply_mask_bundle(d3, float)) as usize;
        let d3_ground = acum_mask_bundle(apply_mask_bundle(d3, ground)) as usize;

        let af1_mask = acum_or(a1) & float;
        let ag1_mask = acum_or(a1) & ground;
        let af2_mask = acum_or(a2) & float;
        let ag2_mask = acum_or(a2) & ground;
        let af3_mask = acum_or(a3) & float;
        let ag3_mask = acum_or(a3) & ground;
        let df1_mask = acum_or(d1) & float;
        let dg1_mask = acum_or(d1) & ground;
        let df2_mask = acum_or(d2) & float;
        let dg2_mask = acum_or(d2) & ground;
        let df3_mask = acum_or(d3) & float;
        let dg3_mask = acum_or(d3) & ground;
        let trap_3_num = (((acum_or(d3) | acum_or(a3)) & 0x0000_ffff_0000_0000).count_ones()
            + (stone.count_ones() % 2)) as usize;

        let pa1_mask = acum_or(pa1);
        let pa2_mask = acum_or(pa2);
        let pa3_mask = acum_or(pa3);
        let pd1_mask = acum_or(pd1);
        let pd2_mask = acum_or(pd2);
        let pd3_mask = acum_or(pd3);

        return (
            a1_float, a2_float, a3_float, a1_ground, a2_ground, a3_ground, d1_float, d2_float,
            d3_float, d1_ground, d2_ground, d3_ground, af1_mask, af2_mask, af3_mask, ag1_mask,
            ag2_mask, ag3_mask, df1_mask, df2_mask, df3_mask, dg1_mask, dg2_mask, dg3_mask,
            pa1_mask, pa2_mask, pa3_mask, pd1_mask, pd2_mask, pd3_mask, trap_3_num, att, def,
        );
    }

    pub fn evaluate_board(&self, b: &Board) -> f32 {
        let (att, def) = b.get_att_def();
        let (a1, a2, a3, pa1, pa2, pa3) = LineEvaluator::analyze_board(att, def);
        let (d1, d2, d3, pd1, pd2, pd3) = LineEvaluator::analyze_board(def, att);
        let stone = att | def;
        let ground = !stone & (stone << 16 | 0xffff);
        let float = !stone ^ ground;
        let a1_float = acum_mask_bundle(apply_mask_bundle(a1, float)) as usize;
        let a1_ground = acum_mask_bundle(apply_mask_bundle(a1, ground)) as usize;
        let a2_float = acum_mask_bundle(apply_mask_bundle(a2, float)) as usize;
        let a2_ground = acum_mask_bundle(apply_mask_bundle(a2, ground)) as usize;
        let a3_float = acum_mask_bundle(apply_mask_bundle(a3, float)) as usize;
        let a3_ground = acum_mask_bundle(apply_mask_bundle(a3, ground)) as usize;
        let d1_float = acum_mask_bundle(apply_mask_bundle(d1, float)) as usize;
        let d1_ground = acum_mask_bundle(apply_mask_bundle(d1, ground)) as usize;
        let d2_float = acum_mask_bundle(apply_mask_bundle(d2, float)) as usize;
        let d2_ground = acum_mask_bundle(apply_mask_bundle(d2, ground)) as usize;
        let d3_float = acum_mask_bundle(apply_mask_bundle(d3, float)) as usize;
        let d3_ground = acum_mask_bundle(apply_mask_bundle(d3, ground)) as usize;

        let af1_mask = acum_or(a1) & float;
        let ag1_mask = acum_or(a1) & ground;
        let af2_mask = acum_or(a2) & float;
        let ag2_mask = acum_or(a2) & ground;
        let af3_mask = acum_or(a3) & float;
        let ag3_mask = acum_or(a3) & ground;
        let df1_mask = acum_or(d1) & float;
        let dg1_mask = acum_or(d1) & ground;
        let df2_mask = acum_or(d2) & float;
        let dg2_mask = acum_or(d2) & ground;
        let df3_mask = acum_or(d3) & float;
        let dg3_mask = acum_or(d3) & ground;

        let pa1_mask = acum_or(pa1);
        let pa2_mask = acum_or(pa2);
        let pa3_mask = acum_or(pa3);
        let pd1_mask = acum_or(pd1);
        let pd2_mask = acum_or(pd2);
        let pd3_mask = acum_or(pd3);

        let trap_3_num = (((acum_or(d3) | acum_or(a3)) & 0x0000_ffff_0000_0000).count_ones()
            + (stone.count_ones() % 2)) as usize;

        let mut val = 0.0;

        for i in 0..64 {
            let bit = 1 << i;
            if af1_mask & bit != 0 {
                val += self.w_bf_1_line[i];
            }
            if af2_mask & bit != 0 {
                val += self.w_bf_2_line[i];
            }
            if af3_mask & bit != 0 {
                val += self.w_bf_3_line[i];
            }
            if df1_mask & bit != 0 {
                val += self.w_dbf_1_line[i];
            }
            if df2_mask & bit != 0 {
                val += self.w_dbf_2_line[i];
            }
            if df3_mask & bit != 0 {
                val += self.w_dbf_3_line[i];
            }
            if ag1_mask & bit != 0 {
                val += self.w_bg_1_line[i];
            }
            if ag2_mask & bit != 0 {
                val += self.w_bg_2_line[i];
            }
            if ag3_mask & bit != 0 {
                val += self.w_bg_3_line[i];
            }
            if dg1_mask & bit != 0 {
                val += self.w_dbg_1_line[i];
            }
            if dg2_mask & bit != 0 {
                val += self.w_dbg_2_line[i];
            }
            if dg3_mask & bit != 0 {
                val += self.w_dbg_3_line[i];
            }
            if pa1_mask & bit != 0 {
                val += self.w_pos_1_line[i];
            } else if pd1_mask & bit != 0 {
                val += self.w_dpos_1_line[i];
            }
            if pa2_mask & bit != 0 {
                val += self.w_pos_2_line[i];
            } else if pd2_mask & bit != 0 {
                val += self.w_dpos_2_line[i];
            }
            if pa3_mask & bit != 0 {
                val += self.w_pos_3_line[i];
            } else if pd3_mask & bit != 0 {
                val += self.w_dpos_3_line[i];
            }
            if att & bit != 0 {
                val += self.w_pos[i];
            } else if def & bit != 0 {
                val += self.w_dpos[i];
            }
        }

        val += self.w_float_1_line[a1_float]
            + self.w_float_2_line[a2_float]
            + self.w_float_3_line[a3_float]
            + self.w_ground_1_line[a1_ground]
            + self.w_ground_2_line[a2_ground]
            + self.w_ground_3_line[a3_ground]
            + (self.w_dif_float_1_line[96 - d1_float + a1_float]
                + self.w_dif_float_2_line[64 - d2_float + a2_float]
                + self.w_dif_float_3_line[32 - d3_float + a3_float]
                + self.w_dif_ground_1_line[96 + a1_ground - d1_ground]
                + self.w_dif_ground_2_line[64 + a2_ground - d2_ground]
                + self.w_dif_ground_3_line[32 + a3_ground - d3_ground])
            + self.w_trap_3_num[trap_3_num]
            + self.bias;
        return 1.0 / (1.0 + (-val).exp());
    }

    pub fn move_average(&mut self) {
        let base = self.w_float_1_line[0];
        self.bias += base;
        for i in self.w_float_1_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_float_2_line[0];
        self.bias += base;
        for i in self.w_float_2_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_float_3_line[0];
        self.bias += base;
        for i in self.w_float_3_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_dif_float_1_line[96];
        self.bias += base;
        for i in self.w_dif_float_1_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_dif_float_2_line[64];
        self.bias += base;
        for i in self.w_dif_float_2_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_dif_float_3_line[32];
        self.bias += base;
        for i in self.w_dif_float_3_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }

        let base = self.w_ground_1_line[0];
        self.bias += base;
        for i in self.w_ground_1_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_ground_2_line[0];
        self.bias += base;
        for i in self.w_ground_2_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_ground_3_line[0];
        self.bias += base;
        for i in self.w_ground_3_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_dif_ground_1_line[96];
        self.bias += base;
        for i in self.w_dif_ground_1_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_dif_ground_2_line[64];
        self.bias += base;
        for i in self.w_dif_ground_2_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_dif_ground_3_line[32];
        self.bias += base;
        for i in self.w_dif_ground_3_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }

        let base = self.w_pos_1_line[0];
        self.bias += base;
        for i in self.w_pos_1_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_pos_2_line[0];
        self.bias += base;
        for i in self.w_pos_2_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_pos_3_line[0];
        self.bias += base;
        for i in self.w_pos_3_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_dpos_1_line[0];
        self.bias += base;
        for i in self.w_dpos_1_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_dpos_2_line[0];
        self.bias += base;
        for i in self.w_dpos_2_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_dpos_3_line[0];
        self.bias += base;
        for i in self.w_dpos_3_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }

        let base = self.w_bf_1_line[0];
        self.bias += base;
        for i in self.w_bf_1_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_bf_2_line[0];
        self.bias += base;
        for i in self.w_bf_2_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_bf_3_line[0];
        self.bias += base;
        for i in self.w_bf_3_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_dbf_1_line[0];
        self.bias += base;
        for i in self.w_dbf_1_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_dbf_2_line[0];
        self.bias += base;
        for i in self.w_dbf_2_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_dbf_3_line[0];
        self.bias += base;
        for i in self.w_dbf_3_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }

        let base = self.w_bg_1_line[0];
        self.bias += base;
        for i in self.w_bg_1_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_bg_2_line[0];
        self.bias += base;
        for i in self.w_bg_2_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_bg_3_line[0];
        self.bias += base;
        for i in self.w_bg_3_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_dbg_1_line[0];
        self.bias += base;
        for i in self.w_dbg_1_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_dbg_2_line[0];
        self.bias += base;
        for i in self.w_dbg_2_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_dbg_3_line[0];
        self.bias += base;
        for i in self.w_dbg_3_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }

        let base = self.w_bf_1_line[0];
        self.bias += base;
        for i in self.w_bf_1_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_bf_2_line[0];
        self.bias += base;
        for i in self.w_bf_2_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_bf_3_line[0];
        self.bias += base;
        for i in self.w_bf_3_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_bg_1_line[0];
        self.bias += base;
        for i in self.w_bg_1_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_bg_2_line[0];
        self.bias += base;
        for i in self.w_bg_2_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_bg_3_line[0];
        self.bias += base;
        for i in self.w_bg_3_line.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }

        let base = self.w_pos[0];
        self.bias += base;
        for i in self.w_pos.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_dpos[0];
        self.bias += base;
        for i in self.w_dpos.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
        let base = self.w_trap_3_num[0];
        self.bias += base;
        for i in self.w_trap_3_num.iter_mut() {
            if *i == 0.0 {
                continue;
            }
            *i -= base;
        }
    }

    pub fn load(&mut self, file: String) -> Result<()> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};
        let mut line = BufReader::new(File::open(file)?).lines();

        let _ = line.next();
        for i in 0..96 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_float_1_line[i] = num;
        }
        let _ = line.next();
        for i in 0..96 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_ground_1_line[i] = num;
        }
        let _ = line.next();
        for i in 0..64 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_float_2_line[i] = num;
        }
        let _ = line.next();
        for i in 0..64 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_ground_2_line[i] = num;
        }
        let _ = line.next();
        for i in 0..32 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_float_3_line[i] = num;
        }
        let _ = line.next();
        for i in 0..32 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_ground_3_line[i] = num;
        }

        let _ = line.next();
        for i in 0..192 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_dif_float_1_line[i] = num;
        }
        let _ = line.next();
        for i in 0..192 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_dif_ground_1_line[i] = num;
        }
        let _ = line.next();
        for i in 0..128 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_dif_float_2_line[i] = num;
        }
        let _ = line.next();
        for i in 0..128 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_dif_ground_2_line[i] = num;
        }
        let _ = line.next();
        for i in 0..64 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_dif_float_3_line[i] = num;
        }
        let _ = line.next();
        for i in 0..64 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_dif_ground_3_line[i] = num;
        }

        let _ = line.next();
        for i in 0..64 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_pos_1_line[i] = num;
        }
        let _ = line.next();
        for i in 0..64 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_pos_2_line[i] = num;
        }
        let _ = line.next();
        for i in 0..64 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_pos_3_line[i] = num;
        }
        let _ = line.next();
        for i in 0..64 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_dpos_1_line[i] = num;
        }
        let _ = line.next();
        for i in 0..64 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_dpos_2_line[i] = num;
        }
        let _ = line.next();
        for i in 0..64 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_dpos_3_line[i] = num;
        }

        let _ = line.next();
        for i in 0..64 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_bf_1_line[i] = num;
        }
        let _ = line.next();
        for i in 0..64 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_bf_2_line[i] = num;
        }
        let _ = line.next();
        for i in 0..64 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_bf_3_line[i] = num;
        }
        let _ = line.next();
        for i in 0..64 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_bg_1_line[i] = num;
        }
        let _ = line.next();
        for i in 0..64 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_bg_2_line[i] = num;
        }
        let _ = line.next();
        for i in 0..64 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_bg_3_line[i] = num;
        }

        let _ = line.next();
        for i in 0..64 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_dbf_1_line[i] = num;
        }
        let _ = line.next();
        for i in 0..64 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_dbf_2_line[i] = num;
        }
        let _ = line.next();
        for i in 0..64 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_dbf_3_line[i] = num;
        }
        let _ = line.next();
        for i in 0..64 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_dbg_1_line[i] = num;
        }
        let _ = line.next();
        for i in 0..64 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_dbg_2_line[i] = num;
        }
        let _ = line.next();
        for i in 0..64 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_dbg_3_line[i] = num;
        }

        let _ = line.next();
        for i in 0..64 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_pos[i] = num;
        }
        let _ = line.next();
        for i in 0..64 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_dpos[i] = num;
        }

        let _ = line.next();
        for i in 0..13 {
            let l = line.next().unwrap()?;
            let num: f32 = l.parse().unwrap();
            self.w_trap_3_num[i] = num;
        }

        let l = line.next().unwrap()?;
        let bias = l.parse().unwrap();
        self.bias = bias;

        Ok(())
    }

    pub fn save(&self, file: String) -> Result<()> {
        use std::fs::File;
        use std::{
            io,
            io::{BufRead, BufReader, Write},
        };
        let mut file = File::create(file)?;

        writeln!(file, "w_float_1_line");
        for i in 0..96 {
            writeln!(file, "{}", self.w_float_1_line[i]);
        }
        writeln!(file, "w_ground_1_line");
        for i in 0..96 {
            writeln!(file, "{}", self.w_ground_1_line[i]);
        }
        writeln!(file, "w_float_2_line");
        for i in 0..64 {
            writeln!(file, "{}", self.w_float_2_line[i]);
        }
        writeln!(file, "w_ground_2_line");
        for i in 0..64 {
            writeln!(file, "{}", self.w_ground_2_line[i]);
        }
        writeln!(file, "w_float_3_line");
        for i in 0..32 {
            writeln!(file, "{}", self.w_float_3_line[i]);
        }
        writeln!(file, "w_ground_3_line");
        for i in 0..32 {
            writeln!(file, "{}", self.w_ground_3_line[i]);
        }

        writeln!(file, "w_dif_float_1_line");
        for i in 0..192 {
            writeln!(file, "{}", self.w_dif_float_1_line[i]);
        }
        writeln!(file, "w_dif_ground_1_line");
        for i in 0..192 {
            writeln!(file, "{}", self.w_dif_ground_1_line[i]);
        }
        writeln!(file, "w_dif_float_2_line");
        for i in 0..128 {
            writeln!(file, "{}", self.w_dif_float_2_line[i]);
        }
        writeln!(file, "w_dif_ground_2_line");
        for i in 0..128 {
            writeln!(file, "{}", self.w_dif_ground_2_line[i]);
        }
        writeln!(file, "w_dif_float_3_line");
        for i in 0..64 {
            writeln!(file, "{}", self.w_dif_float_3_line[i]);
        }
        writeln!(file, "w_dif_ground_3_line");
        for i in 0..64 {
            writeln!(file, "{}", self.w_dif_ground_3_line[i]);
        }

        writeln!(file, "w_pos_1_line");
        for i in 0..64 {
            writeln!(file, "{}", self.w_pos_1_line[i]);
        }
        writeln!(file, "w_pos_2_line");
        for i in 0..64 {
            writeln!(file, "{}", self.w_pos_2_line[i]);
        }
        writeln!(file, "w_pos_3_line");
        for i in 0..64 {
            writeln!(file, "{}", self.w_pos_3_line[i]);
        }
        writeln!(file, "w_dpos_1_line");
        for i in 0..64 {
            writeln!(file, "{}", self.w_dpos_1_line[i]);
        }
        writeln!(file, "w_dpos_2_line");
        for i in 0..64 {
            writeln!(file, "{}", self.w_dpos_2_line[i]);
        }
        writeln!(file, "w_dpos_3_line");
        for i in 0..64 {
            writeln!(file, "{}", self.w_dpos_3_line[i]);
        }

        writeln!(file, "w_bf_1_line");
        for i in 0..64 {
            writeln!(file, "{}", self.w_bf_1_line[i]);
        }
        writeln!(file, "w_bf_2_line");
        for i in 0..64 {
            writeln!(file, "{}", self.w_bf_2_line[i]);
        }
        writeln!(file, "w_bf_3_line");
        for i in 0..64 {
            writeln!(file, "{}", self.w_bf_3_line[i]);
        }
        writeln!(file, "w_bg_1_line");
        for i in 0..64 {
            writeln!(file, "{}", self.w_bg_1_line[i]);
        }
        writeln!(file, "w_bg_2_line");
        for i in 0..64 {
            writeln!(file, "{}", self.w_bg_2_line[i]);
        }
        writeln!(file, "w_bg_3_line");
        for i in 0..64 {
            writeln!(file, "{}", self.w_bg_3_line[i]);
        }

        writeln!(file, "w_dbf_1_line");
        for i in 0..64 {
            writeln!(file, "{}", self.w_dbf_1_line[i]);
        }
        writeln!(file, "w_dbf_2_line");
        for i in 0..64 {
            writeln!(file, "{}", self.w_dbf_2_line[i]);
        }
        writeln!(file, "w_dbf_3_line");
        for i in 0..64 {
            writeln!(file, "{}", self.w_dbf_3_line[i]);
        }
        writeln!(file, "w_dbg_1_line");
        for i in 0..64 {
            writeln!(file, "{}", self.w_dbg_1_line[i]);
        }
        writeln!(file, "w_dbg_2_line");
        for i in 0..64 {
            writeln!(file, "{}", self.w_dbg_2_line[i]);
        }
        writeln!(file, "w_dbg_3_line");
        for i in 0..64 {
            writeln!(file, "{}", self.w_dbg_3_line[i]);
        }

        writeln!(file, "w_pos");
        for i in 0..64 {
            writeln!(file, "{}", self.w_pos[i]);
        }
        writeln!(file, "w_dpos");
        for i in 0..64 {
            writeln!(file, "{}", self.w_dpos[i]);
        }

        writeln!(file, "w_trap_3_num");
        for i in 0..13 {
            writeln!(file, "{}", self.w_trap_3_num[i]);
        }

        writeln!(file, "{}", self.bias);

        file.flush()?;
        Ok(())
    }
}

impl EvaluatorF for LineEvaluator {
    fn eval_func_f32(&self, b: &Board) -> f32 {
        return self.evaluate_board(b);
    }
}

pub trait Trainable {
    fn update(&mut self, b: &Board, delta: f32) {}
    fn get_val(&self, b: &Board) -> f32;
    fn save(&self, file: String) -> Result<()>;
    fn load(&mut self, file: String) -> Result<()>;
    fn eval(&mut self) {}
    fn train(&mut self) {}
}

#[derive(Clone)]
pub struct LELearningParam {
    pub wfl1: bool,
    pub wfl2: bool,
    pub wfl3: bool,
    pub wgl1: bool,
    pub wgl2: bool,
    pub wgl3: bool,
    pub wdfl1: bool,
    pub wdfl2: bool,
    pub wdfl3: bool,
    pub wdgl1: bool,
    pub wdgl2: bool,
    pub wdgl3: bool,
    pub wpl1: bool,
    pub wpl2: bool,
    pub wpl3: bool,
    pub wdpl1: bool,
    pub wdpl2: bool,
    pub wdpl3: bool,
    pub wbfl1: bool,
    pub wbfl2: bool,
    pub wbfl3: bool,
    pub wbgl1: bool,
    pub wbgl2: bool,
    pub wbgl3: bool,
    pub wdbfl1: bool,
    pub wdbfl2: bool,
    pub wdbfl3: bool,
    pub wdbgl1: bool,
    pub wdbgl2: bool,
    pub wdbgl3: bool,
    pub wp: bool,
    pub wdp: bool,
    pub wtn3: bool,
    pub bias: bool,
}

impl LELearningParam {
    pub fn new() -> Self {
        LELearningParam {
            wfl1: true,
            wfl2: true,
            wfl3: true,
            wgl1: true,
            wgl2: true,
            wgl3: true,
            wdfl1: true,
            wdfl2: true,
            wdfl3: true,
            wdgl1: true,
            wdgl2: true,
            wdgl3: true,
            wpl1: true,
            wpl2: true,
            wpl3: true,
            wdpl1: true,
            wdpl2: true,
            wdpl3: true,
            wbfl1: true,
            wbfl2: true,
            wbfl3: true,
            wbgl1: true,
            wbgl2: true,
            wbgl3: true,
            wdbfl1: true,
            wdbfl2: true,
            wdbfl3: true,
            wdbgl1: true,
            wdbgl2: true,
            wdbgl3: true,
            wp: true,
            wdp: true,
            wtn3: true,
            bias: true,
        }
    }
    pub fn from(hash: u64) -> Self {
        LELearningParam {
            wfl1: (hash >> 0) & 1 == 1,
            wfl2: (hash >> 1) & 1 == 1,
            wfl3: (hash >> 2) & 1 == 1,
            wgl1: (hash >> 3) & 1 == 1,
            wgl2: (hash >> 4) & 1 == 1,
            wgl3: (hash >> 5) & 1 == 1,

            wdfl1: (hash >> 6) & 1 == 1,
            wdfl2: (hash >> 7) & 1 == 1,
            wdfl3: (hash >> 8) & 1 == 1,
            wdgl1: (hash >> 9) & 1 == 1,
            wdgl2: (hash >> 10) & 1 == 1,
            wdgl3: (hash >> 11) & 1 == 1,

            wpl1: (hash >> 12) & 1 == 1,
            wpl2: (hash >> 13) & 1 == 1,
            wpl3: (hash >> 14) & 1 == 1,
            wdpl1: (hash >> 15) & 1 == 1,
            wdpl2: (hash >> 16) & 1 == 1,
            wdpl3: (hash >> 17) & 1 == 1,

            wbfl1: (hash >> 18) & 1 == 1,
            wbfl2: (hash >> 19) & 1 == 1,
            wbfl3: (hash >> 20) & 1 == 1,
            wbgl1: (hash >> 21) & 1 == 1,
            wbgl2: (hash >> 22) & 1 == 1,
            wbgl3: (hash >> 23) & 1 == 1,

            wdbfl1: (hash >> 24) & 1 == 1,
            wdbfl2: (hash >> 25) & 1 == 1,
            wdbfl3: (hash >> 26) & 1 == 1,
            wdbgl1: (hash >> 27) & 1 == 1,
            wdbgl2: (hash >> 28) & 1 == 1,
            wdbgl3: (hash >> 29) & 1 == 1,

            wp: (hash >> 30) & 1 == 1,
            wdp: (hash >> 31) & 1 == 1,

            wtn3: (hash >> 32) & 1 == 1,
            bias: (hash >> 33) & 1 == 1,
        }
    }
}

#[derive(Clone)]
pub struct TrainableLineEvaluator {
    main: LineEvaluator,
    v: LineEvaluator,
    m: LineEvaluator,
    lr: f32,
    pub param: LELearningParam,
}

impl TrainableLineEvaluator {
    pub fn new(lr: f32) -> Self {
        TrainableLineEvaluator {
            main: LineEvaluator::new(),
            v: LineEvaluator::new(),
            m: LineEvaluator::new(),
            lr: lr,
            param: LELearningParam::new(),
        }
    }

    pub fn from(e: LineEvaluator, lr: f32) -> Self {
        TrainableLineEvaluator {
            main: e,
            v: LineEvaluator::new(),
            m: LineEvaluator::new(),
            lr: lr,
            param: LELearningParam::new(),
        }
    }

    pub fn set_param(&mut self, hash: u64) {
        self.param = LELearningParam::from(hash);
    }
}

impl Trainable for TrainableLineEvaluator {
    fn update(&mut self, b: &Board, delta: f32) {
        let (
            a1,
            a2,
            a3,
            a1_,
            a2_,
            a3_,
            d1,
            d2,
            d3,
            d1_,
            d2_,
            d3_,
            af1_mask,
            af2_mask,
            af3_mask,
            ag1_mask,
            ag2_mask,
            ag3_mask,
            df1_mask,
            df2_mask,
            df3_mask,
            dg1_mask,
            dg2_mask,
            dg3_mask,
            pa1_mask,
            pa2_mask,
            pa3_mask,
            pd1_mask,
            pd2_mask,
            pd3_mask,
            trap_3_num,
            att,
            def,
        ) = self.main.get_counts(b);
        // とりあえずsgd
        let val = self.main.evaluate_board(b);
        let dv = val * (1.0 - val);
        let delta = self.lr * delta * dv;
        if self.param.wfl1 {
            self.main.w_float_1_line[a1] += delta;
        }
        if self.param.wfl2 {
            self.main.w_float_2_line[a2] += delta;
        }
        if self.param.wfl3 {
            self.main.w_float_3_line[a3] += delta;
        }
        if self.param.wgl1 {
            self.main.w_ground_1_line[a1_] += delta;
        }
        if self.param.wgl2 {
            self.main.w_ground_2_line[a2_] += delta;
        }
        if self.param.wgl3 {
            self.main.w_ground_3_line[a3_] += delta;
        }
        if self.param.wdfl1 {
            self.main.w_dif_float_1_line[96 + a1 - d1] += delta;
        }
        if self.param.wdfl2 {
            self.main.w_dif_float_2_line[64 + a2 - d2] += delta;
        }
        if self.param.wdfl3 {
            self.main.w_dif_float_3_line[32 + a3 - d3] += delta;
        }
        if self.param.wdgl1 {
            self.main.w_dif_ground_1_line[96 + a1_ - d1_] += delta;
        }
        if self.param.wdgl2 {
            self.main.w_dif_ground_2_line[64 + a2_ - d2_] += delta;
        }
        if self.param.wdgl3 {
            self.main.w_dif_ground_3_line[32 + a3_ - d3_] += delta;
        }
        if self.param.wtn3 {
            self.main.w_trap_3_num[trap_3_num] += delta;
        }
        if self.param.bias {
            self.main.bias += delta;
        }

        for i in 0..64 {
            let bit = 1 << i;
            if af1_mask & bit != 0 && self.param.wbfl1 {
                self.main.w_bf_1_line[i] += delta;
            }
            if af2_mask & bit != 0 && self.param.wbfl2 {
                self.main.w_bf_2_line[i] += delta;
            }
            if af3_mask & bit != 0 && self.param.wbfl3 {
                self.main.w_bf_3_line[i] += delta;
            }
            if df1_mask & bit != 0 && self.param.wdbfl1 {
                self.main.w_dbf_1_line[i] += delta;
            }
            if df2_mask & bit != 0 && self.param.wdbfl2 {
                self.main.w_dbf_2_line[i] += delta;
            }
            if df3_mask & bit != 0 && self.param.wdbfl3 {
                self.main.w_dbf_3_line[i] += delta;
            }
            if ag1_mask & bit != 0 && self.param.wbgl1 {
                self.main.w_bg_1_line[i] += delta;
            }
            if ag2_mask & bit != 0 && self.param.wbgl2 {
                self.main.w_bg_2_line[i] += delta;
            }
            if ag3_mask & bit != 0 && self.param.wbgl3 {
                self.main.w_bg_3_line[i] += delta;
            }
            if dg1_mask & bit != 0 && self.param.wdbgl1 {
                self.main.w_dbg_1_line[i] += delta;
            }
            if dg2_mask & bit != 0 && self.param.wdbgl2 {
                self.main.w_dbg_2_line[i] += delta;
            }
            if dg3_mask & bit != 0 && self.param.wdbgl3 {
                self.main.w_dbg_3_line[i] += delta;
            }
            if pa1_mask & bit != 0 && self.param.wpl1 {
                self.main.w_pos_1_line[i] += delta;
            } else if pd1_mask & bit != 0 && self.param.wdpl1 {
                self.main.w_dpos_1_line[i] += delta;
            }
            if pa2_mask & bit != 0 && self.param.wpl2 {
                self.main.w_pos_2_line[i] += delta;
            } else if pd2_mask & bit != 0 && self.param.wdpl2 {
                self.main.w_dpos_2_line[i] += delta;
            }
            if pa3_mask & bit != 0 && self.param.wpl3 {
                self.main.w_pos_3_line[i] += delta;
            } else if pd3_mask & bit != 0 && self.param.wdpl3 {
                self.main.w_dpos_3_line[i] += delta;
            }
            if att & bit != 0 && self.param.wp {
                self.main.w_pos[i] += delta;
            } else if def & bit != 0 && self.param.wdp {
                self.main.w_dpos[i] += delta;
            }
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
    fn train(&mut self) {
        self.main.move_average();
        return;
        print!("wf1:[");
        for f in self.main.w_float_1_line.iter() {
            print!("{},", f);
        }
        println!("]");
        print!("wf2:[");
        for f in self.main.w_float_2_line.iter() {
            print!("{},", f);
        }
        println!("]");
        print!("wf3:[");
        for f in self.main.w_float_3_line.iter() {
            print!("{},", f);
        }
        println!("]");
        print!("wg1:[");
        for f in self.main.w_ground_1_line.iter() {
            print!("{},", f);
        }
        println!("]");
        print!("wg2:[");
        for f in self.main.w_ground_2_line.iter() {
            print!("{},", f);
        }
        println!("]");
        print!("wg3:[");
        for f in self.main.w_ground_3_line.iter() {
            print!("{},", f);
        }
        println!("]");

        println!("bias:{}", self.main.bias);
    }
}

impl EvaluatorF for TrainableLineEvaluator {
    fn eval_func_f32(&self, b: &Board) -> f32 {
        return self.main.evaluate_board(b).clamp(0.0, 1.0);
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

pub struct MateWrapperActor {
    main_actor: Box<dyn GetAction>,
}

impl MateWrapperActor {
    pub fn new(actor: Box<dyn GetAction>) -> Self {
        return MateWrapperActor { main_actor: actor };
    }
}

impl GetAction for MateWrapperActor {
    fn get_action(&self, b: &Board) -> u8 {
        use proconio::input;
        let res = mate_check_horizontal(b);
        if let Some((flag, action)) = res {
            if cfg!(feature = "render") {
                if flag {
                    println!("mate -> {action}");
                } else {
                    println!("not mate -> {action}");
                }
            }
            // println!("->[{action}]");
            return action;
        } else {
            let action = self.main_actor.get_action(b);
            return action;
        }
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
                // println!("mate")
            }
            // println!("->[{action}]");
            return action;
        } else {
            let (action, v, count) = negalpha(b, self.depth, -MAX - 1, MAX + 1, &self.main_eval);
            return action;
        }
        // input! {
        //     action: u8
        // }
        // return action;
    }
}

// pub struct OnnxEvaluator {
//     session: Session,
// }

// impl OnnxEvaluator {
//     pub fn new(file_path: &str) -> Self {
//         let environment = Environment::builder().build().unwrap().into_arc();
//         let session = SessionBuilder::new(&environment)
//             .unwrap()
//             .with_optimization_level(GraphOptimizationLevel::Level3)
//             .unwrap()
//             .with_intra_threads(4)
//             .unwrap()
//             .with_model_from_file(file_path)
//             .unwrap();
//         return OnnxEvaluator { session: session };
//     }

//     pub fn inference(&self, b: &Board) -> f32 {
//         let input_vec = u2vec(b2u128(b));
//         let inputs = ndarray::Array::from_vec(input_vec);
//         let inputs = CowArray::from(inputs.slice(s![..]).into_dyn());
//         let inputs = vec![ort::Value::from_array(self.session.allocator(), &inputs).unwrap()];

//         let output = self.session.run(inputs).unwrap();
//         let output = output[0].try_extract::<f32>().unwrap();
//         let output = output.view();

//         return output[0];
//     }
// }

// impl EvaluatorF for OnnxEvaluator {
//     fn eval_func(&self, b: &Board) -> f32 {
//         return self.inference(b);
//     }
// }

// impl Analyzer for OnnxEvaluator {
//     fn analyze_eval(&self, b: &Board) -> f32 {
//         return self.eval_func(b);
//     }
// }

// impl EvalAndAnalyze for OnnxEvaluator {}
