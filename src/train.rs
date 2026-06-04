use crate::{ai::line::SimplLineEvaluator, db::StepbackBoardDB, match_stats::sprt::eval_actor_sqrt_from_boards};
#[allow(warnings)]
use crate::db::{BoardDB, WeightedTransition};

use super::{
    ai::*,
    board::*,
    dfpn::{proof_number_search, threat_space_search, MateType},
};
use minimum_ml::ml::*;

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
const EVAL_NUM: usize = 50;
const LOG_LOSS_N: usize = 100000;
const SMOOTHING: f32 = 0.999;

#[derive(Debug, Clone)]
pub struct Transition {
    pub board: u128,
    pub result: f32,
    pub val: f32,
}

impl Transition {
    pub fn new() -> Transition {
        return Transition {
            board: 0,
            result: 0.0,
            val: 0.0,
        };
    }

    pub fn to_weighted_transition(&self) -> WeightedTransition{
        return WeightedTransition { board: self.board, result: self.result, val: self.val, weight: 1.0 }
    }
}

#[derive(Debug, Clone)]
pub struct StepbackTransition {
    pub board: u128,
    pub result: f32,
    pub val: f32,
    pub frontstep: u64,
}

pub fn create_db(load_model: Option<impl EvalAndActF>, db_name: &str, depth: usize) {
    use super::db;
    let board_db: db::BoardDB = db::BoardDB::new(db_name, 1);
    let base = board_db.get_count();
    let mut count = 0;
    let start = time::Instant::now();
    let mut rng = rand::thread_rng();

    let bundle_num = 50;
    let mut step = 0;

    board_db.begine();

    loop {
        let random_offset: usize = rng.r#gen::<usize>() % RANDOM_MOVE_MAX;
        let random_step: usize =
            RANDOM_MOVE_MIN + (rng.r#gen::<usize>() % (RANDOM_MOVE_WIDTH - RANDOM_MOVE_MIN));
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
            board_db.add(att, def, (t.result * 2.0 - 1.0) as i32, t.val);
        }

        if step % bundle_num == 0 {
            board_db.end();
            board_db.begine();
        }
    }
}

/// Create diverse training dataset using multiple agents and unified evaluator
///
/// # Arguments
/// * `agents` - List of agents to randomly select from for diverse play styles
/// * `evaluator` - Unified evaluator for position values
/// * `db_name` - Database file name
/// * `depth` - Search depth (for compatibility)
/// * `num_games` - Number of games to generate (0 for infinite loop)
///
/// # Example
/// ```ignore
/// // Create agents
/// let minimax2 = Agent::Minimax(2);
/// let minimax3 = Agent::Minimax(3);
/// let mcts = Agent::Mcts(50, 500);
///
/// // Create evaluator
/// let evaluator = SimplLineEvaluator::new();
///
/// // Generate diverse dataset
/// create_db_multi_agent(
///     &[&minimax2, &minimax3, &mcts],
///     &evaluator,
///     "diverse_dataset.db",
///     3,
///     10000
/// );
/// ```
pub fn create_db_multi_agent(
    agents: Vec<Box<dyn GetAction>>,
    evaluator: &dyn EvaluatorF,
    db_name: &str,
    depth: usize,
    num_games: usize,
) {
    use super::db;
    let board_db: db::BoardDB = db::BoardDB::new(db_name, 1);
    let base = board_db.get_count();
    let mut count = 0;
    let start = time::Instant::now();
    let mut rng = rand::thread_rng();

    let bundle_num = 50;
    let mut step = 0;

    board_db.begine();

    // Convert Vec<Box<dyn GetAction>> to Vec<&dyn GetAction>
    let agent_refs: Vec<&dyn GetAction> = agents.iter().map(|a| a.as_ref()).collect();

    let game_limit = if num_games == 0 {
        usize::MAX
    } else {
        num_games
    };

    for game_idx in 0..game_limit {
        let random_offset: usize = rng.r#gen::<usize>() % RANDOM_MOVE_MAX;
        let random_step: usize =
            RANDOM_MOVE_MIN + (rng.r#gen::<usize>() % (RANDOM_MOVE_WIDTH - RANDOM_MOVE_MIN));

        let ts =
            play_with_eval_multi_agent(&agent_refs, evaluator, depth, random_offset, random_step);
        count += ts.len() as u64;

        if ts.len() == 0 {
            continue;
        }

        if step % 10 == 0 {
            println!(
                "Games: {}/{}, Positions: {}+{} ({}), {}pos/sec, {}pos/hour",
                game_idx + 1,
                if num_games == 0 {
                    "∞".to_string()
                } else {
                    num_games.to_string()
                },
                base,
                count,
                ts.len(),
                count / (1 + start.elapsed().as_secs()),
                3600 * count / (1 + start.elapsed().as_secs())
            );
        }

        step += 1;
        for t in ts {
            let att = t.board as u64;
            let def = (t.board >> 64) as u64;
            board_db.add(att, def, (t.result * 2.0 - 1.0) as i32, t.val);
        }

        if step % bundle_num == 0 {
            board_db.end();
            board_db.begine();
        }
    }

    board_db.end();
    println!("Dataset generation complete! Total positions: {}", count);
}

/// Play a game with multiple agents and unified evaluator
///
/// # Arguments
/// * `agents` - List of agents (GetAction implementations) to randomly select from
/// * `evaluator` - Unified evaluator for position values
/// * `depth` - Search depth (for compatibility, may not be used)
/// * `random_offset` - Turn offset before using agents
/// * `random_step` - Number of random moves
///
/// # Returns
/// Vector of transitions with board states and evaluations
fn play_with_eval_multi_agent(
    agents: &[&dyn GetAction],
    evaluator: &dyn EvaluatorF,
    depth: usize,
    random_offset: usize,
    random_step: usize,
) -> Vec<Transition> {
    let mut b = Board::new();
    let mut transitions = Vec::new();
    let mut reward = 0;
    let mut turn = 0;
    let mut rng = rand::thread_rng();

    // Randomly select agents for black and white
    let black_agent_idx = rng.r#gen::<usize>() % agents.len();
    let white_agent_idx = rng.r#gen::<usize>() % agents.len();

    loop {
        let action;
        let valf: f32;

        // Random moves phase
        if random_offset <= turn && (random_offset + random_step) >= turn {
            action = get_random(&b);
        } else {
            // Evaluate position with unified evaluator
            valf = evaluator.eval_func_f32(&b);

            // Select action from appropriate agent
            if b.is_black() {
                action = agents[black_agent_idx].get_action(&b);
            } else {
                action = agents[white_agent_idx].get_action(&b);
            }

            // Save transition after random phase
            if (random_offset + random_step) < turn {
                transitions.push(Transition {
                    board: b2u128(&b),
                    result: 0.0,
                    val: valf,
                });
            }
        }

        let b_ = b.next(action);

        // Check for mate
        let end = proof_number_search(b.clone());
        if let MateType::Three(_) = end.typ {
            reward = 1;
            break;
        }
        if let MateType::Two(_) = end.typ {
            reward = 1;
            break;
        }

        // Check for win/draw
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

    // Apply TD(λ) for backward propagation
    let size = transitions.len();
    let mut decay = LAMBDA;
    for i in 0..size {
        transitions[size - i - 1].result = ((reward as f32) + 1.0) * 0.5;
        let win_rate = if reward == 1 {
            1.0
        } else if reward == 0 {
            0.5
        } else {
            0.0
        };
        reward *= -1;
        decay *= DECAY_ALPHA;
    }

    transitions
}

/// Legacy play_with_eval for backward compatibility
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
    let b_id: usize = rng.r#gen::<usize>() % 4;
    let w_id: usize = rng.r#gen::<usize>() % 4;

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
                    result: 0.0,
                    val: valf,
                });
            }
            // action = mcts_action(&b, 500, 50);
        }

        let b_ = b.next(action);
        let end = proof_number_search(b.clone());
        if let MateType::Three(_) = end.typ {
            reward = 1;
            break;
        }
        if let MateType::Two(_) = end.typ {
            reward = 1;
            break;
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
        transitions[size - i - 1].result = if reward == 1 {
            1.0
        } else if reward == 0 {
            0.0
        } else {
            0.5
        };
        let win_rate;
        if reward == 1 {
            win_rate = 1.0;
        } else if reward == 0 {
            win_rate = 0.5;
        } else {
            win_rate = 0.0;
        }
        transitions[size - i - 1].val =
            decay * win_rate + (1.0 - decay) * transitions[size - i - 1].val;
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
    let random_step = rng.r#gen::<usize>() % RANDOM_MOVE_MAX;

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
                result: 0.0,
                val: val,
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
        transitions[size - i - 1].result = if reward == 1 {
            1.0
        } else if reward == -1 {
            0.0
        } else {
            0.5
        };
        reward *= -1;
    }

    return transitions;
}

struct BatchIterator {
    data: Vec<WeightedTransition>,
    cursor: usize,
    batch_size: usize,
    batch_num: usize,
    lambda: f32,
    rng: ThreadRng,
}

impl BatchIterator {
    fn new(data: Vec<WeightedTransition>, batch_size: usize, num: usize, lambda: f32) -> Self {
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
    fn from_transition(data: Vec<Transition>, batch_size: usize, num: usize, lambda: f32) -> Self{
        let data: Vec<WeightedTransition> = data.iter().map(|t| t.to_weighted_transition()).collect();
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
    type Item = (Tensor, Tensor, Tensor);
    fn next(&mut self) -> Option<Self::Item> {
        if self.batch_num == 0 {
            return None;
        } else {
            if self.data.len() < self.cursor + self.batch_size {
                self.reset();
            }
            let mut board = Vec::new();
            let mut result = Vec::new();
            let mut weight = Vec::new();

            for t in &self.data[self.cursor..(self.cursor + self.batch_size)] {
                // pprint_board(&u128_to_b(t.board));
                // let res = t.result;
                // println!("res:{res}, val:{}", t.t_val);
                let rot_b = random_rot(t.board, self.rng.r#gen());
                board.push(Tensor::new(u2vec(rot_b), vec![crate::ai::INPUT_SIZE]));
                result.push(Tensor::new(
                    vec![t.val],
                    vec![1],
                ));
                weight.push(Tensor::new(
                    vec![t.weight],
                    vec![1]
                ))
            }
            let board = create_batch(board);
            let result = create_batch(result);
            let weight = create_batch(weight);
            self.cursor += self.batch_size;

            self.batch_num -= 1;

            return Some((board, result, weight));
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
            result: 0.5,
            val: val,
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
        transitions[size - i - 1].result = if reward == 1 {
            1.0
        } else if reward == -1 {
            0.0
        } else {
            0.5
        };
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

        let mut it = BatchIterator::from_transition(dataset.clone(), BATCH_SIZE, BATCH_NUM, LAMBDA);
        it.reset();
        dataset = dataset[REPLAY_DELETE..].to_vec();

        model.train();

        let pb = ProgressBar::new(BATCH_NUM as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) \n {msg}")
            .unwrap()
            .progress_chars("#>-"));
        for (board, result,_) in it {
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
    let x = x.clamp(1e-7, 1.0 - 1e-7);
    let t = t.clamp(0.0, 1.0);
    let loss = -t * x.ln() - (1.0 - t) * (1.0 - x).ln()
        + if t > 1e-7 { t * t.ln() } else { 0.0 }
        + if t < 1.0 - 1e-7 {
            (1.0 - t) * (1.0 - t).ln()
        } else {
            0.0
        };
    let dloss = (x - t) / (x * (1.0 - x));

    if loss.is_nan() || loss.is_infinite() || dloss.is_nan() || dloss.is_infinite() {
        return (0.0, 0.0);
    }
    return (loss, -dloss);
}

pub fn mse_loss(x: f32, t: f32) -> (f32, f32) {
    let error = x - t;
    let loss = error * error;
    return (loss, -2.0 * error);
}

pub fn create_eval_board(n: usize, step: usize) -> Vec<Board> {
    let po = PlayoutEvaluator::new(PlayoutLevel::Defence4);
    let mcts = mcts::Mcts::new(10_000, 3, 100, po);
    let mut bs = Vec::new();
    let pb = ProgressBar::new(n as u64);
    pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) \n {msg}")
            .unwrap()
            .progress_chars("#>-"));

    for _ in 0..n {
        bs.push(n_step_board(&mcts, step));
        pb.inc(1);
        pb.set_message("[create_eval_board]");
    }
    pb.finish();

    return bs;
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
    let eval_db: BoardDB = BoardDB::new(&eval_db_name, 0);
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
        let n = BATCH_SIZE * 1000;
        let batch_num = n / BATCH_SIZE;

        let pb = ProgressBar::new(batch_num as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) \n {msg}")
            .unwrap()
            .progress_chars("#>-"));

        data = ts.choose_multiple(&mut rng, n).cloned().collect();
        let mut it = BatchIterator::from_transition(data, BATCH_SIZE, batch_num, LAMBDA);
        it.reset();

        for (board, result, _) in it {
            model.g.reset();
            let loss = model.g.forward(vec![board, result]);
            model.g.backward();
            model.g.optimize();

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
                    BatchIterator::from_transition(ts.clone(), BATCH_SIZE, eval_ts.len() / BATCH_SIZE, LAMBDA);

                for (board, result, _) in eval_it {
                    model.g.reset();
                    let loss = model.g.forward(vec![board, result]);
                    model.g.backward();
                    // model.g.optimize();
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
    mut model: crate::ai::line_acumlator::TrainableSLIE,
    load: bool,
    save: bool,
    name: String,
    load_name: String,
    db_name: String,
    eval_db_name: String,
) {
    use super::db::WeightedTransition;
    let test_boards = create_eval_board(10, 2);
    let test_actor1 = Agent::Minimax(3);
    let test_actor2 = Agent::Mcts(50, 500);
    let evaluator = super::ai::CoEvaluator::best();
    let neg = super::ai::NegAlpha::new(Box::new(evaluator), 3);
    let mut l = super::ai::line::SimplLineEvaluator::new();
    l.load("simple.json".to_string());
    let mut l3 = NegAlphaF::new(Box::new(l.clone()), 29);
    l3.hashmap = true;
    l3.min_depth = 7;
    l3.timelimit = 100;
    let le = MateWrapperActor::new(Box::new(l3));

    let mut l_high = NegAlphaF::new(Box::new(l.clone()), 29);
    l_high.hashmap = true;
    l_high.min_depth = 7;
    l_high.timelimit = 500;
    let lh = MateWrapperActor::new(Box::new(l_high));

    let mut rng = thread_rng();
    let mut max_score = 0.0;

    if load {
        model.load(load_name.clone());
    }

    let mut db: StepbackBoardDB = StepbackBoardDB::new(&db_name, 0.97, 0.1);
    let mut eval_db: StepbackBoardDB = StepbackBoardDB::new(&eval_db_name, 0.97, 0.1);
    println!("load db");
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
        let n = BATCH_SIZE * 1_000_000;
        let batch_num = n / BATCH_SIZE;

        let pb = ProgressBar::new(batch_num as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) \n {msg}")
            .unwrap()
            .progress_chars("#>-"));

        let data: Vec<WeightedTransition> = ts.choose_multiple(&mut rng, n).cloned().collect();
        // let data = vec![data[0].clone(); n];

        for t in data.iter() {
            let b = &u128_to_b(random_rot(t.board, rng.r#gen()));
            let val = model.get_val(b);
            // println!("val:{:#?}", bce_loss(0.5, t.t_val));
            let result = t.result;
            let t_val = t.val;
            // let (loss, delta) = bce_loss(val, t_val);
            let (loss, delta) = mse_loss(val, t_val);
            model.update(b, delta);
            // let (loss, delta) = bce_loss(0.5, t.t_val);
            // let (loss, delta) = bce_loss(val, (t.result as f32) * 0.499 + 0.5);
            match smoothing_loss {
                None => smoothing_loss = Some(loss),
                Some(loss_) => smoothing_loss = Some(0.99999 * loss_ + (1.0 - 0.99999) * loss),
            }

            pb.inc(1);
            pb.set_message(format!(
                "[loss]:{} \n[smoothed]:{}",
                loss,
                smoothing_loss.unwrap(),
            ));
            if step % LOG_LOSS_N == 0 {
                pb.println(format!("[loss]:{}", smoothing_loss.unwrap()));
                // println!(
                //     "[epoch:{epoch}][step:{step}][loss]:{}",
                //     smoothing_loss.unwrap()
                // );
                let mut losses = Vec::new();

                let eval_it =
                    BatchIterator::new(ts.clone(), BATCH_SIZE, eval_ts.len() / BATCH_SIZE, LAMBDA);

                for t in eval_ts.iter() {
                    let b = &u128_to_b(t.board);
                    let val = model.get_val(b);

                    let (loss, _) = mse_loss(val, t.val);
                    losses.push(loss);
                }
                let size = losses.len();
                pb.println(format!(
                    "[eval]:{}",
                    losses.iter().sum::<f32>() / size as f32
                ));
                // println!(
                //     "[epoch:{epoch}][step:{step}][eval:{}]",
                //     losses.iter().sum::<f32>() / size as f32
                // );
            }
            step += 1;
        }
        pb.finish();

        if epoch % 1 == 0 {
            model.eval();

            // === SPRT-based evaluation ===
            use crate::match_stats::sprt::{eval_actor_sprt, SPRTResult, SPRT};

            // Create SPRT with Elo bounds (detect 10+ Elo difference)
            let sprt = SPRT::with_elo_bounds(0.0, 10.0);

            let mut agent = NegAlphaF::new(Box::new(model.clone()), 3);
            agent.hashmap = true;
            agent.min_depth = 3;
            let agent = MateWrapperActor::new(Box::new(agent));

            println!("\n[Epoch {}] Evaluating vs Minimax(3) with SPRT...", epoch);
            let (sprt_result1, e11, e12) = eval_actor_sprt(&agent, &test_actor1, 100, &sprt, false);

            // Evaluate vs MCTS(50, 500) with SPRT
            println!(
                "\n[Epoch {}] Evaluating vs MCTS(50,500) with SPRT...",
                epoch
            );
            let (sprt_result2, e21, e22) = eval_actor_sprt(&agent, &test_actor2, 100, &sprt, false);

            // Evaluate vs NegAlpha(3) with SPRT
            println!("\n[Epoch {}] Evaluating vs NegAlpha(3) with SPRT...", epoch);
            let (sprt_result3, e31, e32) = eval_actor_sprt(&agent, &neg, 100, &sprt, false);

            // Evaluate vs SimplLineEvaluator on test boards (keep original logic)
            // let mut agent = NegAlphaF::new(Box::new(model.clone()), 29);
            let mut sprt = SPRT::with_elo_bounds(0.0, 200.0);
            sprt.alpha = 0.01;
            sprt.beta = 0.01;
            let mut agent = crate::ai::line_acumlator::TestLineAcumModel::new(model.main.clone());
            agent.limit = 100_000;
            let agent = MateWrapperActor::new(Box::new(agent));
            let (sprt_result4, e41, e42) = eval_actor_sqrt_from_boards(&test_boards, &agent, &le, &sprt, false);
            // if e41 > 0.6 {
            //     let mut agent = NegAlphaF::new(Box::new(model.clone()), 29);
            //     agent.hashmap = true;
            //     agent.min_depth = 7;
            //     agent.timelimit = 500;
            //     let (e51, _) = eval_actor_from_boards(&test_boards, &agent, &lh, false);
            //     e41 += e51 * 5.0;
            // }

            println!(
                "[epoch:{epoch}][step:{step}][minimax(3)]:({:.3}, {:.3}) - {:?}",
                e11, e12, sprt_result1
            );
            println!(
                "[epoch:{epoch}][step:{step}][mcts(50, 500)]:({:.3}, {:.3}) - {:?}",
                e21, e22, sprt_result2
            );
            println!(
                "[epoch:{epoch}][step:{step}][neg(3)]:({:.3}, {:.3}) - {:?}",
                e31, e32, sprt_result3
            );
            println!(
                "[epoch:{epoch}][step:{step}][sle(3)]:({:.3}, {:.3}) - {:?}",
                e41, e42, sprt_result4
            );
            
            if max_score < e41 {
                println!("[epoch:{epoch}]max_score:{}->{}", max_score, e41);
                max_score = e41;
                if save {
                    model.train();
                    model.save(name.clone());
                }
            }else{
                println!("[epoch:{epoch}]max_score:{max_score}");
            }
            if save {
                model.train();
                model.save(format!("latest-{}", name));
            }
        }
    }
}

/// Train NNUE model using minimum_ml's Dataloader and logger
///
/// This function provides the same logic as train_model_with_db but uses
/// minimum_ml's Dataloader for batching and TensorBoard logger for metrics.
///
/// # Arguments
/// * `nnue` - The NNUE model to train
/// * `db_name` - Path to the training database
/// * `eval_db_name` - Path to the evaluation database
/// * `save_name` - Name for saving the model
/// * `epochs` - Number of training epochs
/// * `batch_size` - Batch size for training
pub fn train_nnue_with_dataloader<H: NNUEHash>(
    mut nnue: NNUE<H>,
    db_name: String,
    eval_db_name: String,
    save_name: String,
    epochs: usize,
    batch_size: usize,
) {
    use crate::db::{BoardData, BoardDataset};
    use minimum_ml::dataset::{Dataloader, Dataset};
    use minimum_ml::ml::logger::TensorBoardLogger;

    let test_boards = create_eval_board(25, 2);
    let test_actor1 = Agent::Minimax(3);
    let test_actor2 = Agent::Mcts(50, 500);
    let evaluator = super::ai::CoEvaluator::best();
    let neg = super::ai::NegAlpha::new(Box::new(evaluator), 3);
    let mut l = super::ai::line::SimplLineEvaluator::new();
    l.load("simple.json".to_string());
    let mut l3 = NegAlphaF::new(Box::new(l.clone()), 5);
    l3.hashmap = true;
    l3.min_depth = 5;
    l3.timelimit = 1;
    let le = MateWrapperActor::new(Box::new(l3));

    let stepback_alpha = 0.95;

    println!("Loading databases...");
    let train_db: BoardDataset<H> = BoardDataset::from_stepback_db(&db_name, stepback_alpha, 0.0);
    let eval_db_name_clone = eval_db_name.clone();

    let train_size = train_db.len();
    println!("Train size: {}", train_size);

    // Initialize logger if logging feature is enabled
    let mut logger = TensorBoardLogger::new().expect("Failed to create TensorBoard logger");
    println!("TensorBoard logger initialized. Run: tensorboard --logdir=runs");

    // Create dataloader
    let train_dataloader = Dataloader::new(train_db, batch_size, true);

    // Set NNUE to training mode

    let mut global_step = 0;
    let mut smoothed_loss: Option<f32> = None;

    for epoch in 0..epochs {
        nnue.train();
        println!("\n=== Epoch {}/{} ===", epoch + 1, epochs);

        let num_batches = train_size / batch_size;
        let pb = ProgressBar::new(num_batches as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})\n{msg}")
                .unwrap()
                .progress_chars("#>-")
        );

        // Training loop
        for (i, batch_data) in train_dataloader.iter_batch().enumerate() {
            // Forward pass
            let output = nnue
                .g
                .forward(vec![batch_data.input.clone(), batch_data.label.clone(), batch_data.weight.clone()]);

            // Extract loss value
            let loss_val = output.get_item().unwrap_or(0.0);

            // Backward pass and optimize
            nnue.g.backward();
            nnue.g.optimize();
            nnue.g.reset();

            // Update smoothed loss
            match smoothed_loss {
                None => smoothed_loss = Some(loss_val),
                Some(prev) => smoothed_loss = Some(SMOOTHING * prev + (1.0 - SMOOTHING) * loss_val),
            }

            // Update progress bar
            pb.inc(1);
            if let Some(smooth) = smoothed_loss {
                pb.set_message(format!("loss: {:.6}, smoothed: {:.6}", loss_val, smooth));
            }

            // Log to TensorBoard
            {
                if global_step % 100 == 0 {
                    let _ = logger.log_scalar("train/loss", loss_val);
                    if let Some(smooth) = smoothed_loss {
                        let _ = logger.log_scalar("train/smoothed_loss", smooth);
                    }
                    logger.next_step();
                }
            }

            // Periodic evaluation
            if global_step % LOG_LOSS_N == 0 && global_step > 0 {
                pb.println(format!("[eval at step {}]", global_step));

                // Evaluate on eval set
                nnue.train();
                let mut eval_losses = Vec::new();

                let eval_db_for_eval: BoardDataset<H> =
                    BoardDataset::from_stepback_db(&eval_db_name_clone, stepback_alpha, 0.0);
                let eval_size = eval_db_for_eval.len().min(1024);
                let eval_dataloader = Dataloader::new(eval_db_for_eval, batch_size, false);

                let mut eval_count = 0;
                for batch_data in eval_dataloader.iter_batch() {
                    if eval_count >= eval_size / batch_size {
                        break;
                    }

                    let output = nnue
                        .g
                        .inference(vec![batch_data.input.clone(), batch_data.label.clone(), batch_data.weight.clone()]);
                    let loss = output.get_item().unwrap_or(0.0);
                    eval_losses.push(loss);
                    eval_count += 1;
                }

                let avg_eval_loss = if !eval_losses.is_empty() {
                    eval_losses.iter().sum::<f32>() / eval_losses.len() as f32
                } else {
                    0.0
                };

                pb.println(format!(
                    "[epoch:{}][step:{}][eval loss: {:.6}]",
                    epoch, global_step, avg_eval_loss
                ));

                {
                    let _ = logger.log_scalar("eval/loss", avg_eval_loss);
                }
            }

            global_step += 1;
        }

        pb.finish();

        // Agent-based evaluation (same as train_model_with_db)
        if epoch % 1 == 0 {
            nnue.eval();
            nnue.set_inference();
            nnue.depth = 3;

            // Create agents with NNUE evaluator

            let (e11, e12) = eval_actor(&nnue, &test_actor1, EVAL_NUM, false);
            let (e21, e22) = eval_actor(&nnue, &test_actor2, EVAL_NUM, false);
            let (e31, e32) = eval_actor(&nnue, &neg, EVAL_NUM, false);

            nnue.depth = 3;

            let (e41, e42) = eval_actor_from_boards(&test_boards, &nnue, &le, false);

            println!(
                "[epoch:{}][step:{}][minimax(3)]:({}, {})",
                epoch, global_step, e11, e12
            );
            println!(
                "[epoch:{}][step:{}][mcts(50, 500)]:({}, {})",
                epoch, global_step, e21, e22
            );
            println!(
                "[epoch:{}][step:{}][neg(3)]:({}, {})",
                epoch, global_step, e31, e32
            );
            println!(
                "[epoch:{}][step:{}][sle(3)]:({}, {})",
                epoch, global_step, e41, e42
            );

            // Log evaluation metrics
            {
                let _ = logger.log_scalar("eval/agent_minimax3_win", e11);
                let _ = logger.log_scalar("eval/agent_mcts_win", e21);
                let _ = logger.log_scalar("eval/agent_neg3_win", e31);
                let _ = logger.log_scalar("eval/agent_sle3_score", e41);
            }
        }

        // Save model every epoch
        println!("Saving model...");
        nnue.save(format!("{}_epoch{}", save_name, epoch));
        nnue.save(format!("latest-{}", save_name));

        println!("Epoch {} complete!", epoch + 1);
    }

    println!("\nTraining complete!");
    {
        let _ = logger.flush();
        println!("TensorBoard logs saved. View with: tensorboard --logdir=runs");
    }
}

pub fn create_stepback_db(
    model: &Option<impl EvalAndActF>,
    db_name: &str,
    random_start: usize,
    max_random_insert: usize,
    greedy_rate: f32,
    tempature: f32,
    play_num: Option<isize>,
) {
    use super::db;
    let board_db = db::StepbackBoardDB::new(db_name, DECAY_ALPHA, LAMBDA);
    let mut count = 0;
    let base_count = board_db.get_count() as u64;
    let start = time::Instant::now();
    let mut l = SimplLineEvaluator::new();
    l.load("simple.json".to_string());

    let mut l = NegAlphaF::new(Box::new(l.clone()), 5);
    l.scout = true;
    l.timelimit = 1;
    l.min_depth = 5;
    let mut rng = rand::thread_rng();
    let mut play_num = play_num.clone();

    loop {
        if play_num.is_some(){
            let n = play_num.unwrap();
            if n <= 0{
                break;
            }
            play_num = Some(n-1);
            println!("play_num left {n}");
        }
        let greedy_rate_instant = 1.0 - (1.0 - greedy_rate) * rng.r#gen::<f32>();
        let random_start = random_start + (rng.r#gen::<usize>() % 12);
        let random_insert_step = (rng.r#gen::<usize>() % (max_random_insert + 1));
        println!("greedy_rate:{}", greedy_rate_instant);
        let ts = play_stepback(model, random_start, random_insert_step, greedy_rate_instant, &l, tempature);
        count += ts.len() as u64;
        if ts.len() == 0 {
            continue;
        }
        println!(
            "count:{}({}), {}count/sec, {}count/hour",
            base_count + count,
            ts.len(),
            count / (1 + start.elapsed().as_secs()),
            3600 * count / (1 + start.elapsed().as_secs())
        );

        println!("writing");
        for (t, backstep) in ts {
            let att = t.board as u64;
            let def = (t.board >> 64) as u64;
            board_db.add(att, def, t.result, backstep, t.frontstep, t.val);
        }
    }
}

fn play_stepback(
    model: &Option<impl EvalAndActF>,
    random_start: usize,
    random_insert_step: usize, 
    greedy_rate: f32,
    random_model: &NegAlphaF,
    temp: f32
) -> Vec<(StepbackTransition, u64)> {
    let mut b = Board::new();
    let mut transitions = Vec::new();
    let mut reward = 0;
    let mut turn = 0;
    let mut rng = rand::thread_rng();
    let mut end_insert = random_insert_step == 0;
    let mut random_insert_step = random_insert_step;
    let mut start_insert = false;

    loop {
        let action;
        let mut valf: f32 = 0.5;

        // Random start phase
        if b.is_draw() {
            break;
        }

        if turn < random_start {
            // action = get_random(&b);
            action = random_model.get_action_with_temp(&b, temp);
        } else {
            // Greedy or random
            if (rng.r#gen::<f32>() < greedy_rate && !start_insert || end_insert) && model.is_some()  {
                // println!("random action");
                // pprint_board(&b);
                let (a, v) = model.as_ref().unwrap().eval_and_act(&b);
                action = a;
                valf = v;

                transitions.push(StepbackTransition {
                    board: b2u128(&b),
                    result: 0.0,
                    val: valf,
                    frontstep: turn as u64,
                });
            } else {
                start_insert = true;
                // action = get_random(&b);
                action = random_model.get_action_with_temp(&b, temp);
                if let Some(m) = model {
                    (_, valf) = m.eval_and_act(&b);
                }
                transitions = Vec::new();
                random_insert_step -= 1;
                if random_insert_step == 0{
                    end_insert = true;
                    start_insert = false;
                }
            }
        }

        let b_ = b.next(action);

        // Check for mate
        // let end = proof_number_search(b.clone());
        // let end = threat_space_search(b.get_att_def());
        // if end.is_some(){
        //     reward = 1;
        //     break;
        // }
        // if let MateType::Three(_) = end.typ {
        //     reward = 1;
        //     break;
        // }
        // if let MateType::Two(_) = end.typ {
        //     reward = 1;
        //     break;
        // }

        // Check for win/draw
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

    // Apply TD(λ) logic and calculate backstep
    let size = transitions.len();
    let mut results = Vec::new();

    let is_mate = reward == 1;

    for i in 0..size {
        let idx = size - i - 1;
        transitions[idx].result = ((reward as f32) + 1.0) * 0.5;

        let backstep = if is_mate { i as u64 } else { 0 };
        results.push((transitions[idx].clone(), backstep));

        reward *= -1;
    }
    results.reverse();

    results
}
