use super::{ai::*, board::*, ml::*};
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use rand::Rng;

const EPOCH: usize = 100;
const DEPTH: u8 = 3;
const DATASET_SIZE: usize = 1 << 14;
const BATCH_SIZE: usize = 1 << 6;
const BATCH_NUM: usize = 1 << 14;
const LAMBDA: f32 = 0.3;
const EVAL_NUM: usize = 20;
const LOG_LOSS_N: usize = 1000;

// TODO: data argumentation

#[derive(Debug)]
pub struct Transition {
    board: u128,
    result: i32,
    t_val: f32,
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

fn play_and_record(agent: &NNUE) -> Vec<Transition> {
    let mut b = Board::new();
    let mut transitions = Vec::new();
    let mut reward = 0;

    loop {
        let (_, val, count) = agent.eval_with_negalpha(&b, DEPTH);
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

struct BatchIterator {
    data: Vec<Transition>,
    cursor: usize,
    batch_size: usize,
    batch_num: usize,
    rng: ThreadRng,
}

impl BatchIterator {
    fn new(data: Vec<Transition>, batch_size: usize, num: usize) -> Self {
        let rng = rand::thread_rng();
        return BatchIterator {
            data: data,
            cursor: 0,
            batch_num: num,
            batch_size: batch_size,
            rng: rng,
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
                    vec![res * LAMBDA + (1.0 - LAMBDA) * t.t_val],
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
        let (_, val, count) = agent.eval_with_negalpha(&b, DEPTH);
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

pub fn train(load: bool, save: bool, name: String) {
    let mut model = NNUE::default();
    let mut rng = rand::thread_rng();

    let test_actor1 = Agent::Minimax(3);
    let test_actor2 = Agent::Mcts(50, 500);

    if load {
        model.load(name.clone());
    } else if save {
        model.save(name.clone());
    }

    for epoch in 0..EPOCH {
        model.eval();
        model.set_inference();

        let (e11, e12) = eval_model(&model, &test_actor1);
        let (e21, e22) = eval_model(&model, &test_actor2);
        println!("[minimax(3)]:({}, {})", e11, e12);
        println!("[mcts(50, 500)]:({}, {})", e21, e22);

        play_with_analyze(&model);

        let mut dataset = Vec::new();

        let pb = ProgressBar::new(DATASET_SIZE as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) \n {msg}")
            .unwrap()
            .progress_chars("#>-"));

        while dataset.len() < DATASET_SIZE {
            let mut record = play_and_record(&model);
            pb.inc(record.len() as u64);
            dataset.append(&mut record);
            // print!(
            //     "loading:{}",
            //     dataset.len() as f32 * 100.0 / DATASET_SIZE as f32
            // );
        }

        pb.finish();

        dataset.shuffle(&mut rng);

        let mut dataset = BatchIterator::new(dataset, BATCH_SIZE, BATCH_NUM);

        model.train();

        let pb = ProgressBar::new(BATCH_NUM as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) \n {msg}")
            .unwrap()
            .progress_chars("#>-"));

        let mut i = 0;
        for (board, result) in dataset {
            model.g.reset();
            let loss = model.g.forward(vec![board, result]);
            model.g.backward();
            model.g.optimize();

            pb.inc(1);
            pb.set_message(format!("[loss]:{}", loss.get_item().unwrap()));
            if i % LOG_LOSS_N == 0 {
                println!("[loss]:{}", loss.get_item().unwrap());
            }
            i += 1;
        }

        if save {
            model.save(name.clone());
        }
    }
}
