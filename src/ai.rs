use super::board::{Board, GetAction};
use super::ml::{Graph, Tensor};
use rand::Rng;

pub const MAX: i32 = 1600;

pub fn negmax<F>(b: &Board, depth: u8, eval_func: &F) -> (u8, i32, i32)
where
    F: Fn(&Board) -> i32,
{
    let mut count = 0;
    let mut max_val = -MAX - 1;
    let mut max_action = 16;
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
    let mut count = 0;
    let actions = b.valid_actions();
    let mut max_val = -MAX - 1;
    let mut max_action: u8 = 16;
    for action in actions.iter() {
        let next_board = &b.next(*action);
        if next_board.is_win() {
            return (*action, -MAX, count);
        } else if next_board.is_draw() {
            return (*action, 0, count);
        } else if depth <= 1 {
            let val = e.eval_func(b);
            if max_val < val {
                max_val = val;
                max_action = *action;
                if max_val > beta {
                    return (max_action, -max_val, count);
                }
            }
        } else {
            let (_, val, _count) = negalpha(next_board, depth - 1, -max_val, -alpha, e);
            count += 1 + _count;
            if max_val < val {
                max_val = val;
                max_action = *action;
                if max_val > beta {
                    return (max_action, -max_val, count);
                }
            }
        }
    }
    // println!("[{}]max_val:{}", max_action, max_val);
    return (max_action, max_val, count);
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
}

impl GetAction for NegAlpha {
    fn get_action(&self, b: &Board) -> u8 {
        let (action, _, _) = negalpha(b, self.depth, -MAX - 1, MAX + 1, &self.evaluator);
        return action;
    }
}

pub trait Evaluator {
    fn eval_func(&self, b: &Board) -> i32;
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

        let l1 = Linear::auto(128, 16);
        let l1 = g.add_layer(vec![i1], Box::new(l1));

        let activate = ClippedReLU::default();
        let relu = g.add_layer(vec![l1], Box::new(activate));

        let l2 = Linear::auto(16, 4);
        let l2 = g.add_layer(vec![relu], Box::new(l2));

        let relu2 = g.add_layer(vec![l2], Box::new(ClippedReLU::default()));

        let l3 = Linear::auto(4, 1);
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
        let (mut att, mut def) = b.get_att_def();

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
                let val = self.inference(b);
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
                count += 1 + _count;
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

pub fn b2onehot(att: u64, def: u64) -> Tensor {
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
    return Tensor::new(onehot, vec![128, 1]);
}

impl Evaluator for MLEvaluator {
    fn eval_func(&self, b: &Board) -> i32 {
        let (mut att, mut def) = b.get_att_def();

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
        let (action, val, count) = self.eval_with_negalpha(b, 2);

        println!("{action}, {val}, {count}");
        return action;
    }
}

impl GetAction for &MLEvaluator {
    fn get_action(&self, b: &Board) -> u8 {
        let (action, val, count) = self.eval_with_negalpha(b, 3);

        println!("{action}, {val}, {count}");
        return action;
    }
}

pub struct NNUEEvaluator {
    g: Graph,
    l1: usize,
    eval_out: usize,
}
