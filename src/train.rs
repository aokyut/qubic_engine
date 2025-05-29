#[allow(warnings)]
use crate::db::BoardDB;

use super::{ai::*, board::*, ml::*};
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use std::f32::EPSILON;
use std::time::Duration;
use std::{thread, time};

const EPOCH: usize = 10000;
const DEPTH: u8 = 3;
const RANDOM_MOVE: usize = 7;
const RANDOM_MOVE_MAX: usize = 1;
const RANDOM_MOVE_WIDTH: usize = 48;
const RANDOM_MOVE_MIN: usize = 4;
const DATASET_SIZE: usize = 1 << 14;
const REPLAY_DELETE: usize = 1 << 13;
const BATCH_SIZE: usize = 1 << 0;
const BATCH_NUM: usize = 1 << 10;
pub const LAMBDA: f32 = 0.0;
const DECAY_ALPHA: f32 = 0.92;
const EVAL_NUM: usize = 25;
const LOG_LOSS_N: usize = 100000;
const SMOOTHING: f32 = 0.999999;

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

pub fn create_db(load_model: Option<impl EvalAndActF>, db_name: &str, depth: usize) {
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
        let random_offset: usize = rng.gen::<usize>() % RANDOM_MOVE_MAX;
        let random_step: usize =
            RANDOM_MOVE_MIN + (rng.gen::<usize>() % (RANDOM_MOVE_WIDTH - RANDOM_MOVE_MIN));
        let ts = play_with_eval(depth, random_offset, random_step, &load_model);
        count += ts.len() as u64;
        if ts.len() == 0 {
            continue;
        }
        println!(
            "count:{base}+{count}({}), {}count/sec, {}count/hour",
            ts.len(),
            count / (1 + start.elapsed().as_secs()),
            3600 * count / (1 + start.elapsed().as_secs())
        );
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

fn play_with_eval(
    depth: usize,
    random_offset: usize,
    random_step: usize,
    model: &Option<impl EvalAndActF>,
) -> Vec<Transition> {
    let mut b = Board::new();
    let mut transitions = Vec::new();
    let mut reward = 0;

    let mut turn = 0;
    let evaluator = super::ai::CoEvaluator::best();

    let neg = super::ai::NegAlpha::new(Box::new(evaluator), depth as u8);
    let play_agent = super::board::Agent::Mcts(50, 500);

    let mut rng = rand::thread_rng();
    let b_id: usize = rng.gen::<usize>() % 4;
    let w_id: usize = rng.gen::<usize>() % 4;

    let evaluator = super::ai::CoEvaluator::best();
    let b_actor;
    let w_actor;
    if b_id == 0 {
        b_actor = super::ai::NegAlpha::new(Box::new(evaluator), 3);
    } else if b_id == 1 {
        b_actor = super::ai::NegAlpha::new(Box::new(evaluator), 4);
    } else if b_id == 2 {
        b_actor = super::ai::NegAlpha::new(Box::new(evaluator), 5);
    } else {
        b_actor = super::ai::NegAlpha::new(Box::new(evaluator), 3);
    }
    let evaluator = super::ai::CoEvaluator::best();
    if w_id == 0 {
        w_actor = super::ai::NegAlpha::new(Box::new(evaluator), 3);
    } else if w_id == 1 {
        w_actor = super::ai::NegAlpha::new(Box::new(evaluator), 4);
    } else if w_id == 2 {
        w_actor = super::ai::NegAlpha::new(Box::new(evaluator), 5);
    } else {
        w_actor = super::ai::NegAlpha::new(Box::new(evaluator), 3);
    }

    loop {
        // pprint_board(&b);
        let action;
        let val: i32;
        let valf: f32;
        let count: i32;
        if random_offset <= turn && (random_offset + random_step) >= turn {
            action = get_random(&b);
            // action = play_agent.get_action(&b);
            // thread::sleep(Duration::from_micros(3000));
        } else {
            match model {
                Some(evaluator) => {
                    if b_id == 3 && b.is_black() || w_id == 3 && !b.is_black() {
                        (action, valf) = evaluator.eval_and_act(&b);
                    } else {
                        (_, valf) = evaluator.eval_and_act(&b);
                        if b.is_black() {
                            action = b_actor.get_action(&b);
                        } else {
                            action = w_actor.get_action(&b);
                        }
                    }
                }
                None => {
                    (action, val, count) = neg.eval_with_negalpha(&b);
                    valf = 1.0 / (1.0 + (-(val as f32) / 250.0).exp());
                }
            }
            if (random_offset + random_step) < turn {
                transitions.push(Transition {
                    board: b2u128(&b),
                    result: 0,
                    t_val: valf,
                });
            }
            if cfg!(feature = "slow") {
                thread::sleep(Duration::from_micros(2500));
            }
            // action = mcts_action(&b, 500, 50);
        }

        let b_ = b.next(action);
        let end = mate_check_horizontal(&b);
        if let Some((flag, _)) = end {
            if flag {
                reward = 1;
                break;
            }
        }
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
    let mut decay = LAMBDA;
    for i in 0..size {
        transitions[size - i - 1].result = reward;
        let win_rate;
        if reward == 1 {
            win_rate = 1.0;
        } else if reward == 0 {
            win_rate = 0.5;
        } else {
            win_rate = 0.0;
        }
        transitions[size - i - 1].t_val =
            decay * win_rate + (1.0 - decay) * transitions[size - i - 1].t_val;
        // decay * win_rate + (1.0 - decay) * 0.5;
        reward *= -1;
        decay *= DECAY_ALPHA;
    }

    // if transitions.len() == 0 {
    //     return transitions;
    // }
    // let transitions = vec![transitions[0].clone()];

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
    let rng = rand::thread_rng();

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
    let mut i = 0;
    let mut smoothed_loss = None;

    for epoch in 0..EPOCH {
        model.eval();
        model.set_inference();

        // if epoch != 0 {
        let (e11, e12) = eval_model(&model, &test_actor1);
        let (e21, e22) = eval_model(&model, &test_actor2);
        let (e31, e32) = eval_model(&model, &neg);
        println!("[{epoch}, minimax(3)]:({}, {})", e11, e12);
        println!("[{epoch}, mcts(50, 500)]:({}, {})", e21, e22);
        println!("[{epoch}, neg(3)]:({}, {})", e31, e32);
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
        for (board, result) in it {
            i += 1;
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
        }

        if save {
            model.save(name.clone());
        }
    }
}

pub fn bce_loss(x: f32, t: f32) -> (f32, f32) {
    let loss = -t * (x + 0.000001).ln() - (1.0 - t) * (1.000001 - x).ln()
        + t * (t + 0.000001).ln()
        + (1.0 - t) * (1.000001 - t).ln();
    let dloss = (x - t) / (x * (1.0 - x) + 0.000001);
    return (loss, -dloss);
}

pub fn mse_loss(x: f32, t: f32) -> (f32, f32) {
    let error = x - t;
    let loss = error * error;
    return (loss, -2.0 * error);
}

pub fn train_with_db(load: bool, save: bool, name: String, db_name: String, eval_db_name: String) {
    let mut model = NNUE::default();
    model.g.optimizer = Some(Box::new(optim::MomentumSGD::new(0.01, 0.9)));

    let test_actor1 = Agent::Minimax(3);
    let test_actor2 = Agent::Mcts(50, 500);
    let evaluator = super::ai::CoEvaluator::best();
    let neg = super::ai::NegAlpha::new(Box::new(evaluator), 3);
    let mut rng = thread_rng();

    if load {
        model.load(name.clone());
    } else if save {
        model.save(name.clone());
    }

    let mut db: BoardDB = BoardDB::new(&db_name, 0);
    let eval_db = BoardDB::new(&eval_db_name, 0);
    let ts = db.get_batch();
    let eval_ts = eval_db.get_batch()[..1024].to_vec();
    let mut data = Vec::new();
    let mut smoothing_loss = None;

    for epoch in 0..EPOCH {
        let mut step = 0;

        model.eval();
        model.set_inference();

        if true {
            let (e11, e12) = eval_model(&model, &test_actor1);
            let (e21, e22) = eval_model(&model, &test_actor2);
            let (e31, e32) = eval_actor(&model, &neg, EVAL_NUM, false);
            println!("[{epoch}][minimax(3)]:({}, {})", e11, e12);
            println!("[{epoch}][mcts(50, 500)]:({}, {})", e21, e22);
            println!("[{epoch}][neg(3)]:({}, {})", e31, e32);
        }

        // play_with_analyze(&model);
        model.train();
        db.set_batch_num();
        // db.set_lambda(LAMBDA);

        let batch_num = ts.len() / BATCH_SIZE;
        let n = BATCH_SIZE * 50000;
        let batch_num = n / BATCH_SIZE;

        let pb = ProgressBar::new(batch_num as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) \n {msg}")
            .unwrap()
            .progress_chars("#>-"));

        data = ts.choose_multiple(&mut rng, n).cloned().collect();
        let mut it = BatchIterator::new(data, BATCH_SIZE, batch_num, LAMBDA);
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

                let eval_it =
                    BatchIterator::new(ts.clone(), BATCH_SIZE, eval_ts.len() / BATCH_SIZE, LAMBDA);

                for (board, result) in eval_it {
                    model.g.reset();
                    let loss = model.g.forward(vec![board, result]);
                    model.g.backward();
                    // model.g.optimize();
                    if cfg!(feature = "slow") {
                        thread::sleep(Duration::from_millis(50));
                    }
                    losses.push(loss.get_item().unwrap());
                }
                let size = losses.len();
                pb.println(format!(
                    "[eval_loss]:{}",
                    losses.iter().sum::<f32>() / size as f32
                ));
                println!(
                    "[epoch:{epoch}][step:{step}][loss]:{} \n[eval_loss]:{}",
                    smoothing_loss.unwrap(),
                    losses.iter().sum::<f32>() / size as f32
                );
            }
            step += 1;
        }

        if save {
            model.save(name.clone());
        }
    }
}

pub fn train_model_with_db(
    mut model: impl Trainable + EvaluatorF + Clone + 'static,
    load: bool,
    save: bool,
    name: String,
    load_name: String,
    db_name: String,
    eval_db_name: String,
) {
    let test_actor1 = Agent::Minimax(3);
    let test_actor2 = Agent::Mcts(50, 500);
    let evaluator = super::ai::CoEvaluator::best();
    let neg = super::ai::NegAlpha::new(Box::new(evaluator), 3);
    let mut l = super::ai::line::SimplLineEvaluator::new();
    l.load("simple.json".to_string());
    let mut l3 = NegAlphaF::new(Box::new(l.clone()), 29);
    l3.hashmap = true;
    l3.min_depth = 5;
    l3.timelimit = 1;
    let le = MateWrapperActor::new(Box::new(l3));
    let mut rng = thread_rng();
    let mut max_score = 0.0;

    if load {
        model.load(load_name.clone());
    }

    let mut db: BoardDB = BoardDB::new(&db_name, 0);
    let eval_db = BoardDB::new(&eval_db_name, 0);
    let ts = db.get_all();
    let eval_ts = eval_db.get_all()[..1024].to_vec();
    let mut smoothing_loss = None;
    let mut step = 0;

    for epoch in 0..EPOCH {
        // play_with_analyze(&model);
        model.train();
        db.set_batch_num();
        // db.set_lambda(LAMBDA);

        let batch_num = ts.len() / BATCH_SIZE;
        let n = BATCH_SIZE * 1000_000;
        let batch_num = n / BATCH_SIZE;

        let pb = ProgressBar::new(batch_num as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) \n {msg}")
            .unwrap()
            .progress_chars("#>-"));

        let data: Vec<Transition> = ts.choose_multiple(&mut rng, n).cloned().collect();
        // let data = vec![data[0].clone(); n];

        for t in data.iter() {
            let b = &u128_to_b(random_rot(t.board, rng.gen()));
            let val = model.get_val(b);

            if cfg!(feature = "slow") {
                thread::sleep(Duration::from_micros(200));
            }
            // println!("val:{:#?}", bce_loss(0.5, t.t_val));
            let (loss, delta) = bce_loss(val, t.t_val);
            // let (loss, delta) = mse_loss(val, t.t_val);
            model.update(b, delta);
            // let (loss, delta) = bce_loss(0.5, t.t_val);
            // let (loss, delta) = bce_loss(val, (t.result as f32) * 0.499 + 0.5);
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
                println!(
                    "[epoch:{epoch}][step:{step}][loss]:{}",
                    smoothing_loss.unwrap()
                );
                let mut losses = Vec::new();

                let eval_it =
                    BatchIterator::new(ts.clone(), BATCH_SIZE, eval_ts.len() / BATCH_SIZE, LAMBDA);

                for t in eval_ts.iter() {
                    let b = &u128_to_b(t.board);
                    let val = model.get_val(b);

                    // let (loss, _) = bce_loss(val, (t.result as f32) * 0.499 + 0.5);
                    let (loss, _) = bce_loss(val, t.t_val);
                    // let (loss, _) = mse_loss(val, t.t_val);
                    // let (loss, _) = bce_loss(0.5, t.t_val);
                    losses.push(loss);
                }
                let size = losses.len();
                pb.println(format!(
                    "[eval]:{}",
                    losses.iter().sum::<f32>() / size as f32
                ));
                println!(
                    "[epoch:{epoch}][step:{step}][eval:{}]",
                    losses.iter().sum::<f32>() / size as f32
                );
            }
            step += 1;
        }
        pb.finish();

        if epoch % 1 == 0 {
            model.eval();
            let mut agent = NegAlphaF::new(Box::new(model.clone()), 3);
            agent.hashmap = true;
            agent.min_depth = 3;
            let agent = MateWrapperActor::new(Box::new(agent));
            let (e11, e12) = eval_actor(&agent, &test_actor1, EVAL_NUM, false);
            let (e21, e22) = eval_actor(&agent, &test_actor2, EVAL_NUM, false);
            let (e31, e32) = eval_actor(&agent, &neg, EVAL_NUM, false);
            let mut agent = NegAlphaF::new(Box::new(model.clone()), 29);
            agent.hashmap = true;
            agent.min_depth = 5;
            agent.timelimit = 1;
            let agent = MateWrapperActor::new(Box::new(agent));
            let (e41, e42) = eval_actor(&agent, &le, EVAL_NUM, false);
            println!(
                "[epoch:{epoch}][step:{step}][minimax(3)]:({}, {})",
                e11, e12
            );
            println!(
                "[epoch:{epoch}][step:{step}][mcts(50, 500)]:({}, {})",
                e21, e22
            );
            println!("[epoch:{epoch}][step:{step}][neg(3)]:({}, {})", e31, e32);
            println!("[epoch:{epoch}][step:{step}][sle(3)]:({}, {})", e41, e42);

            if max_score < e41 {
                println!("[epoch:{epoch}]max_score:{}->{}", max_score, e41);
                max_score = e41;
                if save {
                    model.train();
                    model.save(name.clone());
                }
            }
        }
    }
}
