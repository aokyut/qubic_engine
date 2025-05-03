use super::EvaluatorF;
use crate::board::{Board, GetAction};
use rand::Rng;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::Instant;

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

    pub fn search(
        &mut self,
        expand_n: usize,
        search_n: usize,
        e: &Box<dyn EvaluatorF>,
        limit_millis: u128,
    ) -> Vec<Score> {
        let start = Instant::now();
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
        loop {
            self.evaluate(expand_n, e);
            if start.elapsed().as_millis() > limit_millis {
                break;
            }
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

    fn evaluate(&mut self, expand_n: usize, e: &Box<dyn EvaluatorF>) -> f32 {
        if self.board.is_win() {
            self.w += 1.0;
            self.n += 1.0;
            return 1.0;
        } else if self.board.is_draw() {
            self.w += 0.5;
            self.n += 1.0;
            return 0.5;
        } else if self.children.len() == 0 {
            let value = 1.0 - e.eval_func_f32(&self.board);
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
            let value = 1.0
                - self
                    .children
                    .get(&next_node_action)
                    .unwrap()
                    .borrow_mut()
                    .evaluate(expand_n, e);
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

pub struct Mcts {
    limit_millis: u128,
    search_n: usize,
    expand_n: usize,
    evaluator: Box<dyn EvaluatorF>,
}

impl Mcts {
    pub fn new(
        search_n: usize,
        expand_n: usize,
        limit_millis: u128,
        evaluator: impl EvaluatorF + 'static,
    ) -> Self {
        return Mcts {
            search_n: search_n,
            expand_n: expand_n,
            evaluator: Box::new(evaluator),
            limit_millis: limit_millis,
        };
    }

    pub fn evaluate(&self, board: &Board) -> (u8, f32) {
        let start = Instant::now();
        let mut node = Node::new(board.clone());
        let mut scores = node.search(
            self.expand_n,
            self.search_n,
            &self.evaluator,
            self.limit_millis,
        );
        let t = start.elapsed().as_micros();
        // let mut max_action = 0;
        let mut max_actions = Vec::new();
        let mut max_score = -2.0;
        scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        for score in scores {
            if cfg!(feature = "view") {
                println!(
                    "action:{}, val:{}, na:{}, n:{}",
                    score.action, score.q, score.na, score.n
                );
            }
            if score.score > max_score {
                max_score = score.score;
                max_actions = vec![score.action];
            } else if score.score == max_score {
                max_actions.push(score.action);
            }
        }
        if cfg!(feature = "view") {
            println!("total_time:{}ms, {}μs", t / 1000, t);
        }
        let mut rng = rand::thread_rng();
        return (
            max_actions[rng.gen::<usize>() % max_actions.len()],
            max_score,
        );
    }
}

impl GetAction for Mcts {
    fn get_action(&self, b: &Board) -> u8 {
        let (action, _) = self.evaluate(b);
        return action;
    }
}

impl EvaluatorF for Mcts {
    fn eval_func_f32(&self, b: &Board) -> f32 {
        let (_, val) = self.evaluate(b);
        return val;
    }
}
