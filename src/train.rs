use crate::db::BoardDB;

use super::{ai::*, board::*, ml::*};
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use std::time::Duration;
use std::{thread, time};

const EPOCH: usize = 100;
const DEPTH: u8 = 3;
const RANDOM_MOVE: usize = 7;
const RANDOM_MOVE_MAX: usize = 12;
const DATASET_SIZE: usize = 1 << 14;
const REPLAY_DELETE: usize = 1 << 13;
const BATCH_SIZE: usize = 1 << 4;
const BATCH_NUM: usize = 1 << 10;
pub const LAMBDA: f32 = 0.0;
const EVAL_NUM: usize = 100;
const LOG_LOSS_N: usize = 5000;
const SMOOTHING: f32 = 0.9975;

#[derive(Debug, Clone)]
pub struct Transition {
    pub board: u128,
    pub result: i32,
    pub t_val: f32,
}

impl Transition {
    pub fn new() -> Transition {
        return Transition {
            board: 0,
            result: 0,
            t_val: 0.0,
        };
    }
}

pub fn create_db(load_model: Option<&str>, db_name: &str, depth: usize) {
    use super::db;
    let board_db = db::BoardDB::new(db_name, 1);
    let base = board_db.get_count();
    let mut count = 0;
    let start = time::Instant::now();
    let mut rng = rand::thread_rng();

    let bundle_num = 50;
    let mut step = 0;

    board_db.begine();

    loop {
        thread::sleep(Duration::from_millis(100));
        println!(
            "count:{base}+{count}, {}count/sec",
            count / (1 + start.elapsed().as_secs())
        );
        let random_step: usize = rng.gen::<usize>() % 12;
        let ts = play_with_eval(depth, random_step);
        count += ts.len() as u64;
        step += 1;
        for t in ts {
            let att = t.board as u64;
            let def = (t.board >> 64) as u64;
            board_db.add(att, def, t.result, t.t_val);
        }

        if step % bundle_num == 0 {
            board_db.end();
            board_db.begine();
        }
    }
}

fn play_with_eval(depth: usize, random_step: usize) -> Vec<Transition> {
    let mut b = Board::new();
    let mut transitions = Vec::new();
    let mut reward = 0;

    let mut turn = 0;
    let evaluator = super::ai::CoEvaluator::best();
    let neg = super::ai::NegAlpha::new(Box::new(evaluator), depth as u8);

    loop {
        // pprint_board(&b);
        let action;
        let val: i32;
        let count: i32;
        if turn < random_step {
            action = get_random(&b);
        } else {
            (action, val, count) = neg.eval_with_negalpha(&b);
            // action = mcts_action(&b, 500, 50);
            transitions.push(Transition {
                board: b2u128(&b),
                result: 0,
                t_val: 1.0 / (1.0 + (-(val as f32) / 250.0).exp()),
            });
        }

        let b_ = b.next(action);
        if b_.is_win() {
            reward = 1;
            break;
        } else if b_.is_draw() {
            reward = 0;
            break;
        }
        b = b_;
        turn += 1;
    }

    let size = transitions.len();
    for i in 0..size {
        transitions[size - i - 1].result = reward;
        reward *= -1;
    }

    return transitions;
}

fn play_and_record(agent: &NNUE) -> Vec<Transition> {
    let mut b = Board::new();
    let mut transitions = Vec::new();
    let mut reward = 0;

    let mut turn = 0;
    let evaluator = super::ai::CoEvaluator::best();
    let neg = super::ai::NegAlpha::new(Box::new(evaluator), 3);

    let mut rng = thread_rng();
    let random_step = rng.gen::<usize>() % RANDOM_MOVE_MAX;

    loop {
        // pprint_board(&b);
        // let (_, val, count) = agent.eval_with_negalpha(&b);
        let (action_, _val, _count) = neg.eval_with_negalpha(&b);
        let val = 1.0 / (1.0 + (-(_val as f32) / 400.0).exp());
        let action;
        if turn < random_step {
            action = get_random(&b);
        } else {
            // action = mcts_action(&b, 500, 50);
            action = action_;
            // action = get_random(&b);
            transitions.push(Transition {
                board: b2u128(&b),
                result: 0,
                t_val: val,
            });
        }
        // pprint_board(&b);
        // println!("[{action}]");

        let b_ = b.next(action);
        if b_.is_win() {
            reward = 1;
            break;
        } else if b_.is_draw() {
            reward = 0;
            break;
        }
        b = b_;
        turn += 1;
    }

    let size = transitions.len();
    for i in 0..size {
        transitions[size - i - 1].result = reward;
        reward *= -1;
    }

    return transitions;
}

struct BatchIterator {
    data: Vec<Transition>,
    cursor: usize,
    batch_size: usize,
    batch_num: usize,
    lambda: f32,
    rng: ThreadRng,
}

impl BatchIterator {
    fn new(data: Vec<Transition>, batch_size: usize, num: usize, lambda: f32) -> Self {
        let rng = rand::thread_rng();
        return BatchIterator {
            data: data,
            cursor: 0,
            batch_num: num,
            batch_size: batch_size,
            rng: rng,
            lambda: lambda,
        };
    }
    fn reset(&mut self) {
        self.data.shuffle(&mut self.rng);
        self.cursor = 0;
    }
}

fn random_rot(b: u128, id: usize) -> u128 {
    let id = id % 8;
    let mut b = b;
    if id < 4 {
        b = Board::hflip(b);
    }
    for i in 0..(id % 4) {
        b = Board::rot(b);
    }
    return b;
}

impl Iterator for BatchIterator {
    type Item = (Tensor, Tensor);
    fn next(&mut self) -> Option<Self::Item> {
        if self.batch_num == 0 {
            return None;
        } else {
            if self.data.len() < self.cursor + self.batch_size {
                self.reset();
            }
            let mut board = Vec::new();
            let mut result = Vec::new();

            for t in &self.data[self.cursor..(self.cursor + self.batch_size)] {
                // pprint_board(&u128_to_b(t.board));
                let res;
                if t.result == 1 {
                    res = 1.0;
                } else if t.result == -1 {
                    res = 0.0;
                } else {
                    res = 0.5;
                }
                // println!("res:{res}, val:{}", t.t_val);
                let rot_b = random_rot(t.board, self.rng.gen());
                board.push(Tensor::new(u2vec(rot_b), vec![128, 1]));
                result.push(Tensor::new(
                    vec![res * self.lambda + (1.0 - self.lambda) * t.t_val],
                    vec![1, 1],
                ));
            }
            let board = create_batch(board);
            let result = create_batch(result);
            self.cursor += self.batch_size;

            self.batch_num -= 1;

            return Some((board, result));
        }
    }
}

pub fn eval_model(model: &NNUE, tar: &impl GetAction) -> (f32, f32) {
    let (result1, result2) = eval_actor(model, tar, EVAL_NUM, false);
    return (result1, result2);
}

fn play_with_analyze(agent: &NNUE) -> Vec<Transition> {
    let mut b = Board::new();
    let mut transitions = Vec::new();
    let mut reward = 0;

    loop {
        let (_, val, count) = agent.eval_with_negalpha(&b);
        agent.analyze(&b);
        let action = mcts_action(&b, 1000, 50);
        // pprint_board(&b);
        // println!("[{action}]");
        transitions.push(Transition {
            board: b2u128(&b),
            result: 0,
            t_val: val,
        });

        let b_ = b.next(action);
        if b_.is_win() {
            reward = 1;
            break;
        } else if b_.is_draw() {
            reward = 0;
            break;
        }
        b = b_;
    }

    let size = transitions.len();
    for i in 0..size {
        transitions[size - i - 1].result = reward;
        reward *= -1;
    }

    return transitions;
}

pub fn train(load: bool, save: bool, name: String, depth: usize) {
    let mut model = NNUE::default();
    let mut rng = rand::thread_rng();

    let test_actor1 = Agent::Minimax(3);
    let test_actor2 = Agent::Mcts(50, 500);
    let evaluator = super::ai::CoEvaluator::best();
    let neg = super::ai::NegAlpha::new(Box::new(evaluator), 3);

    if load {
        model.load(name.clone());
    } else if save {
        model.save(name.clone());
    }

    model.set_depth(depth);
    let mut dataset = Vec::new();

    for epoch in 0..EPOCH {
        model.eval();
        model.set_inference();

        // if epoch != 0 {
        let (e11, e12) = eval_model(&model, &test_actor1);
        let (e21, e22) = eval_model(&model, &test_actor2);
        let (e31, e32) = eval_model(&model, &neg);
        println!("[minimax(3)]:({}, {})", e11, e12);
        println!("[mcts(50, 500)]:({}, {})", e21, e22);
        println!("[neg(3)]:({}, {})", e31, e32);
        // }

        // play_with_analyze(&model);

        let pb = ProgressBar::new(DATASET_SIZE as u64);

        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) \n {msg}")
            .unwrap()
            .progress_chars("#>-"));

        pb.inc(dataset.len() as u64);

        while dataset.len() < DATASET_SIZE {
            let mut record = play_and_record(&model);
            thread::sleep(Duration::from_millis(100));
            pb.inc(record.len() as u64);
            dataset.append(&mut record);
            // print!(
            //     "loading:{}",
            //     dataset.len() as f32 * 100.0 / DATASET_SIZE as f32
            // );
        }

        pb.finish();

        let mut it = BatchIterator::new(dataset.clone(), BATCH_SIZE, BATCH_NUM, LAMBDA);
        it.reset();
        dataset = dataset[REPLAY_DELETE..].to_vec();

        model.train();

        let pb = ProgressBar::new(BATCH_NUM as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) \n {msg}")
            .unwrap()
            .progress_chars("#>-"));

        let mut i = 0;
        let mut smoothed_loss = None;
        for (board, result) in it {
            model.g.reset();
            let loss = model.g.forward(vec![board, result]);
            model.g.backward();
            model.g.optimize();
            thread::sleep(Duration::from_millis(100));

            let loss = loss.get_item().unwrap();
            match smoothed_loss {
                None => {
                    smoothed_loss = Some(loss);
                }
                Some(s) => {
                    smoothed_loss = Some(SMOOTHING * s + (1.0 - SMOOTHING) * loss);
                }
            }

            pb.inc(1);
            pb.set_message(format!(
                "[loss]:{} \n[smoothed]:{}",
                loss,
                smoothed_loss.unwrap()
            ));
            if i % LOG_LOSS_N == 0 {
                pb.println(format!("[smoothed_loss]:{}", smoothed_loss.unwrap()));
                // println!("[smoothed_loss]:{}", smoothed_loss.unwrap());
            }
            i += 1;
        }

        if save {
            model.save(name.clone());
        }
    }
}

pub fn train_with_db(load: bool, save: bool, name: String, db_name: String, eval_db_name: String) {
    let mut model = NNUE::default();

    let test_actor1 = Agent::Minimax(3);
    let test_actor2 = Agent::Mcts(50, 500);
    let evaluator = super::ai::CoEvaluator::best();
    let neg = super::ai::NegAlpha::new(Box::new(evaluator), 3);

    if load {
        model.load(name.clone());
    } else if save {
        model.save(name.clone());
    }

    let mut step = 0;

    let mut db: BoardDB = BoardDB::new(&db_name, 0);
    let mut eval_db = BoardDB::new(&eval_db_name, 0);
    let ts = db.get_batch();
    let eval_ts = eval_db.get_batch()[..1024].to_vec();

    for epoch in 0..EPOCH {
        model.eval();
        model.set_inference();

        let (e11, e12) = eval_model(&model, &test_actor1);
        let (e21, e22) = eval_model(&model, &test_actor2);
        let (e31, e32) = eval_actor(&model, &neg, EVAL_NUM, false);
        println!("[minimax(3)]:({}, {})", e11, e12);
        println!("[mcts(50, 500)]:({}, {})", e21, e22);
        println!("[neg(3)]:({}, {})", e31, e32);

        // play_with_analyze(&model);
        model.train();
        db.set_batch_num();
        // db.set_lambda(LAMBDA);

        let batch_num = ts.len() / BATCH_SIZE;

        let pb = ProgressBar::new(batch_num as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) \n {msg}")
            .unwrap()
            .progress_chars("#>-"));

        let mut smoothing_loss = None;
        let mut it = BatchIterator::new(ts.clone(), BATCH_SIZE, batch_num, LAMBDA);
        it.reset();

        for (board, result) in it {
            model.g.reset();
            let loss = model.g.forward(vec![board, result]);
            model.g.backward();
            model.g.optimize();
            thread::sleep(Duration::from_millis(50));

            let loss = loss.get_item().unwrap();
            match smoothing_loss {
                None => smoothing_loss = Some(loss),
                Some(loss_) => smoothing_loss = Some(SMOOTHING * loss_ + (1.0 - SMOOTHING) * loss),
            }

            pb.inc(1);
            pb.set_message(format!(
                "[loss]:{} \n[smoothed]:{}",
                loss,
                smoothing_loss.unwrap(),
            ));
            if step % LOG_LOSS_N == 0 {
                pb.println(format!("[loss]:{}", smoothing_loss.unwrap()));

                let mut losses = Vec::new();

                let mut eval_it =
                    BatchIterator::new(ts.clone(), BATCH_SIZE, eval_ts.len() / BATCH_SIZE, LAMBDA);

                for (board, result) in eval_it {
                    model.g.reset();
                    let loss = model.g.forward(vec![board, result]);
                    model.g.backward();
                    // model.g.optimize();
                    thread::sleep(Duration::from_millis(50));
                    losses.push(loss.get_item().unwrap());
                }
                let size = losses.len();
                pb.println(format!(
                    "[eval_loss]:{}",
                    losses.iter().sum::<f32>() / size as f32
                ));
            }
            step += 1;
        }

        if save {
            model.save(name.clone());
        }
    }
}
