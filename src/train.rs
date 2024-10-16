use super::{ai::*, board::*, ml::*};
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use rand::Rng;

const EPOCH: usize = 10;
const DEPTH: u8 = 3;
const DATASET_SIZE: usize = 2 << 11;
const BATCH_SIZE: usize = 2 << 5;
const BATCH_NUM: usize = 2 << 8;
const LAMBDA: f32 = 0.5;
const EVAL_NUM: usize = 5;

#[derive(Debug)]
pub struct Transition {
    att: u64,
    def: u64,
    result: i32,
    t_val: f32,
}

impl Transition {
    pub fn new() -> Transition {
        return Transition {
            att: 0,
            def: 0,
            result: 0,
            t_val: 0.0,
        };
    }
}

fn play_and_record(agent: &MLEvaluator) -> Vec<Transition> {
    let mut b = Board::new();
    let mut transitions = Vec::new();
    let mut reward = 0;

    loop {
        let (_, val, count) = agent.eval_with_negalpha(&b, DEPTH);
        let action = mcts_action(&b, 1000, 50);
        // pprint_board(&b);
        // println!("[{action}]");
        let (att, def) = b.get_att_def();
        transitions.push(Transition {
            att: att,
            def: def,
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
                board.push(b2onehot(t.att, t.def));
                let res = t.result as f32;
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

pub fn eval_model(model: &MLEvaluator, tar: &impl GetAction) -> (f32, f32) {
    let (result1, result2) = eval_actor(model, tar, EVAL_NUM);
    return (result1, result2);
}

pub fn train() {
    let mut model = MLEvaluator::default();
    let mut rng = rand::thread_rng();

    let test_actor1 = Agent::Mcts(10, 100);
    let test_actor2 = Agent::Mcts(50, 500);

    model.save(format!("test_graph"));

    for epoch in 0..EPOCH {
        model.eval();

        let (e11, e12) = eval_model(&model, &test_actor1);
        let (e21, e22) = eval_model(&model, &test_actor2);
        println!("[a1]:({}, {})", e11, e12);
        println!("[a2]:({}, {})", e21, e22);

        let mut dataset = Vec::new();

        while dataset.len() < DATASET_SIZE {
            let mut record = play_and_record(&model);
            dataset.append(&mut record);
            println!(
                "loading:{}",
                dataset.len() as f32 * 100.0 / DATASET_SIZE as f32
            );
        }

        dataset.shuffle(&mut rng);

        let mut dataset = BatchIterator::new(dataset, BATCH_SIZE, BATCH_NUM);

        model.train();
        for (board, result) in dataset {
            model.g.reset();
            let loss = model.g.forward(vec![board, result]);
            model.g.backward();
            model.g.optimize();

            println!("[loss]:{}", loss.get_item().unwrap());
        }
    }
}