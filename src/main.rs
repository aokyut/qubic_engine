mod main_utils;
use proconio::input;
use qubic_engine::ai::line::{
    BucketLineEvaluator, SimplLineEvaluator, SimplePatternEvaluator, TrainableBLE, TrainableSLE,
    TrainableSPE,
};
use qubic_engine::ai::line_nn::{
    MMEvaluator, NNLineEvaluator, NNLineEvaluator_, TrainableNLE, TrainableNLE_,
};
use qubic_engine::ai::pattern::TrainablePatternEvaluator;
use qubic_engine::ai::position::{PositionMaskEvaluator, TrainablePME};
use qubic_engine::ai::zhashmap::test_zhash;
use qubic_engine::ai::{
    self, pattern, LineEvaluator, MateNegAlpha, MateWrapperActor, NegAlphaF, PlayoutEvaluator,
    PlayoutLevel, PositionEvaluator, TrainableLineEvaluator,
};
use qubic_engine::board::{
    count_2row_, get_2row_mask, get_random, mate_check_horizontal, play_actor, pprint_board,
    pprint_u64, Board, GetAction, _is_win_board,
};
use qubic_engine::db::BoardDB;
use qubic_engine::train::{create_db, train_with_db};
use qubic_engine::{
    ai::{CoEvaluator, NegAlpha, NNUE},
    board::{compare_agent, Agent},
};
use rand::rngs::ThreadRng;
use rand::thread_rng;
use rand::Rng;
use rand_distr::num_traits::WrappingNeg;
use std::collections::HashSet;
use std::ops::Neg;
use std::os::unix::thread;
use std::time::{Duration, Instant};

fn main() {
    use qubic_engine::ai::NegAlpha;
    use qubic_engine::board::*;

    // let a1 = Agent::Mcts(50, 5000);
    let m1 = NegAlpha::new(Box::new(CoEvaluator::best()), 1);
    let m2 = NegAlpha::new(Box::new(CoEvaluator::best()), 2);
    let m3 = NegAlpha::new(Box::new(CoEvaluator::best()), 3);
    let m3_f32 = NegAlphaF::new(Box::new(CoEvaluator::best()), 3);
    let m4 = NegAlpha::new(Box::new(CoEvaluator::best()), 4);
    let m5 = NegAlpha::new(Box::new(CoEvaluator::best()), 5);
    let m6 = NegAlpha::new(Box::new(CoEvaluator::best()), 6);
    let m7 = NegAlpha::new(Box::new(CoEvaluator::best()), 7);
    let m9 = NegAlpha::new(Box::new(CoEvaluator::best()), 9);
    let mm3 = MateNegAlpha::new(Box::new(CoEvaluator::best()), 5);
    let mm3 = Agent::Struct(String::from("mm3"), Box::new(mm3));
    let m2 = Agent::Struct(String::from("m2"), Box::new(m2));
    let m3 = Agent::Struct(String::from("m3"), Box::new(m3));
    let m5 = Agent::Struct(String::from("m5"), Box::new(m5));
    let mut l = SimplLineEvaluator::new();
    l.load("sle_tl50.json".to_string());
    // let l3 = wrapping_line_eval(l.clone(), 3);
    // let l4 = wrapping_line_eval(l.clone(), 4);
    // let l5 = wrapping_line_eval(l.clone(), 5);
    let mut l3_ = NegAlphaF::new(Box::new(l.clone()), 1);
    let mut l5_ = NegAlphaF::new(Box::new(l.clone()), 29);
    // l5_.scout = true;
    l5_.hashmap = true;
    l5_.timelimit = 0;
    l5_.min_depth = 5;
    let l5_ = MateWrapperActor::new(Box::new(l5_));
    l.load("simple.json".to_string());

    let mut l7_ = NegAlphaF::new(Box::new(l.clone()), 5);
    // l7_.hashmap = true;
    l7_.scout = true;
    l7_.timelimit = 1;
    l7_.min_depth = 3;
    // let l7_ = MateWrapperActor::new(Box::new(l7_));
    //
    let mut b = BucketLineEvaluator::new();
    b.load("bsimple.json".to_string());

    let mut b7 = NegAlphaF::new(Box::new(b), 5);
    b7.scout = true;
    b7.timelimit = 1;
    b7.min_depth = 3;

    let b7 = MateWrapperActor::new(Box::new(b7));

    let po = PlayoutEvaluator::new(PlayoutLevel::Defence4);
    let mcts = ai::mcts::Mcts::new(10_000, 3, 10, po);
    let mcts2 = ai::mcts::Mcts::new(10_000, 3, 500, l3_);

    // let mut l5_ = NegAlphaF::new(Box::new(l.clone()), 5);
    // let l5_ = MateWrapperActor::new(Box::new(l5_));
    main_utils::test_pns();
    return;
    main_utils::generate_problems();
    return;

    // let l6 = wrapping_line_eval(l.clone(), 6);
    // let l7 = wrapping_line_eval(l.clone(), 7);
    // let l8 = wrapping_line_eval(l.clone(), 8);
    // let test = NegAlpha::new(Box::new(PositionEvaluator::simpl_alpha(1, 0, 0, 0, 0, 0)), 3);
    // let att = 13268249354320136724;
    // let def = 4854231148105118187;
    // let b = Board::from(att, def, Player::Black);
    // pprint_board(&b);
    // let _ = l5_.eval_with_negalpha_(&b);

    // make_db();
    // use_aip();
    // let result = play_actor_with_undo(&Agent::Human, &b7, true);
    // let result = play_actor_with_undo(&b7, &Agent::Human, true);
    // println!("{result:#?}");
    // let db = BoardDB::new("mcoe3_insertRandom48_4_decay092", 0);
    // let db_ = BoardDB::new("mcoe3_insertRandom48_4_decay092_", 0);
    // db.concat(db_);
    // return;

    // explore_best_model();

    // let start = Instant::now();
    // let result = eval_actor(&b7, &l7_, 100, true);
    // compare(&l5_, &l7_);
    // println!("time:{}", start.elapsed().as_nanos());
    // println!("{result:#?}");
    // return;
    // let (a, b, c) = compare(&m2, &mm3);

    exp_get_reach_mask();
    // command();
    // mpc_for_coe(8, 4);
    //profile();
    // beam_search();
    // test_zhash();
    // print_pmodel();
    // exp_positoin_eval_get_count();
    // get_magic_number();
}

pub struct BoardIterator<A1: GetAction, A2: GetAction> {
    a1: A1,
    a2: A2,
    board: Board,
}

impl<A1: GetAction, A2: GetAction> BoardIterator<A1, A2> {
    pub fn new(a1: A1, a2: A2) -> Self {
        return BoardIterator {
            a1: a1,
            a2: a2,
            board: Board::new(),
        };
    }

    pub fn reset(&mut self) -> Board {
        self.board = Board::new();
        return self.board.clone();
    }
}

impl<A1: GetAction, A2: GetAction> Iterator for BoardIterator<A1, A2> {
    type Item = Board;

    fn next(&mut self) -> Option<Self::Item> {
        use qubic_engine::board::Player;
        let action = match self.board.player {
            Player::Black => self.a1.get_action(&self.board),
            Player::White => self.a2.get_action(&self.board),
        };
        self.board = self.board.next(action);
        return Some((self.board.clone()));
    }
}

fn exp_prob_compare() {
    use qubic_engine::utills::half_imcomplete_beta_func;
    let n = 10000;
    let step = 100;
    let mut rng = thread_rng();
    let mut counts = vec![0; 100];
    let th = 0.01;

    for i in 0..n {
        let mut a = 0;
        let mut b = 0;
        let mut count = 0;

        for j in 0..step {
            if rng.gen::<f32>() < 0.5 {
                a += 1;
            } else {
                b += 1;
            }
        }
        let p = half_imcomplete_beta_func(a as f64, b as f64);
        if p < th {
            counts[1] += 1;
        } else {
            counts[0] += 1;
        }

        continue;
        for j in 0..step {
            if rng.gen::<f32>() < 0.5 {
                a += 1;
            } else {
                b += 1;
            }
            let p = half_imcomplete_beta_func(a as f64, b as f64);
            println!("({a}, {b}) {p}");
            if p < th {
                count += 1;
                counts[count] += 1;
                break;
            }
            counts[count] += 1;
        }
    }
    println!("{:#?}", counts);
    let p = half_imcomplete_beta_func(62.0, 38.0);
    println!("{p}");
}

fn exp() {
    let mut l = SimplLineEvaluator::new();
    l.load("simple.json".to_string());

    let mut l7_ = NegAlphaF::new(Box::new(l.clone()), 7);
    l7_.hashmap = true;
    l7_.timelimit = 100;
    l7_.min_depth = 5;

    let mut b = Board::new();
    loop {
        let result = mate_check_horizontal(&b);
        if let Some((flag, _)) = result {
            if flag {
                println!("end");
                break;
            }
        }

        let mut max_action = 0;
        let mut max = 0.0;
        let mut queries = Vec::new();
        for action in b.valid_actions() {
            let nb = b.next(action);
            let (_, val, count) = l7_.eval_with_negalpha(&nb);
            queries.push((action, 1.0 - val, count));
        }
        queries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for query in queries {
            println!("action:{}, val:{}, count:{}", query.0, query.1, query.2);
        }
        let (action, val, count) = l7_.eval_with_negalpha(&b);

        println!("action -> {action}");
        b = b.next(action);
        pprint_board(&b);
    }
}

fn command() {
    use std::env;

    let args: Vec<String> = env::args().collect();
    println!("{:#?}", args);
    if args[1] == "train" {
        println!("train_line_eval({}, {})", args[2], args[3]);
        train_line_eval(args[2].clone(), args[3].clone());
    }
}

fn get_magic_number() {
    // let digit = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let digit = vec![0, 21, 42, 63];
    let mut max = 0;
    let mut rng = thread_rng();

    for i in 0_u64..(1 << 32) {
        let tar = rng.gen::<u64>();
        // let tar = tar as u32;
        let mut ok = true;
        // println!("tar:{tar}");

        for i in 0_u64..(1 << digit.len()) {
            let mut j: u64 = 0;
            for k in 0..digit.len() {
                if (i >> k) & 1 == 1 {
                    j |= 1 << digit[k];
                }
            }
            println!("i:{i}, j:{j:b},{}", j.count_ones());
            let produce = ((j.wrapping_mul(tar)) >> 60) & 15;

            let expected: u64 = j.count_ones() as u64;
            if expected != produce.count_ones() as u64 {
                if max < i {
                    max = i;
                }
                println!("tar:{tar:x}, end:{i:x}, produce:{produce}, expected:{expected}");
                ok = false;
                break;
            }
        }
        if ok {
            println!("magic number:{tar:x} !");
            break;
        }
    }
}

fn call_evaluator() {
    let mut l = SimplLineEvaluator::new();
    l.load("simple.json".to_string());
    // let l3 = wrapping_line_eval(l.clone(), 3);
    // let l4 = wrapping_line_eval(l.clone(), 4);
    // let l5 = wrapping_line_eval(l.clone(), 5);
    // let mut l3_ = NegAlphaF::new(Box::new(l.clone()), 1);
    // let l3_ = MateWrapperActor::new(Box::new(l3_));
    let mut l5_ = NegAlphaF::new(Box::new(l.clone()), 29);
    l5_.hashmap = true;
    l5_.timelimit = 1;
    l5_.min_depth = 5;

    input! {
        black: u64,
        white: u64,
    }
}

fn exp_positoin_eval_get_count() {
    use qubic_engine::ai::position::PositionMaskEvaluator;
    use qubic_engine::exp::BoardIter;

    let mut it = BoardIter::new(true);
    let mut time_a = 0;
    let mut time_b = 0;
    let n = 1_000_000;
    let mut count = 0;

    for b in it {
        count += 1;
        let start = Instant::now();
        let a_list = PositionMaskEvaluator::get_counts(&b);
        time_a += start.elapsed().as_nanos();

        let start = Instant::now();
        let b_list = PositionMaskEvaluator::get_counts(&b);
        // let b_list = PositionMaskEvaluator::get_counts_(&b);
        time_b += start.elapsed().as_nanos();

        for j in 0..16 {
            // assert_eq!(a_list[j], b_list[j], "{:#?}, {:#?}", a_list, b_list);
            if a_list[j] != b_list[j] {
                println!("{:#?}, {:#?}", a_list, b_list);
                pprint_board(&b);
                assert!(false);
            }
        }
        if count > n {
            break;
        }
    }
    println!(
        "time_a:{time_a}:{}, time_b:{time_b}:{}, rate:{}%",
        time_a / n,
        time_b / n,
        time_b * 100 / time_a
    );
}

fn print_pmodel() {
    let mut pmodel = PositionMaskEvaluator::new();
    pmodel.load("position.json".to_string());

    println!("{:#?}", pmodel);
}

fn use_ai() {
    let mut l = SimplLineEvaluator::new();
    l.load("simple.json".to_string());
    let mut l5_ = NegAlphaF::new(Box::new(l.clone()), 29);
    l5_.hashmap = true;
    l5_.timelimit = 1000;
    l5_.min_depth = 7;
    let l5_ = MateWrapperActor::new(Box::new(l5_));

    loop {
        println!("0: 人間 vs AI");
        println!("1: AI vs AI");
        input! {
            n: usize,
        }
        if n == 0 {
            loop {
                println!("-> 人間 vs AI");
                println!("0: 人間が先手");
                println!("1: AIが先手");

                input! {
                    n: usize,
                }
                if n == 0 {
                    println!("-> 人間が先手");
                    play_actor(&Agent::Human, &l5_, true);
                    return;
                } else if n == 1 {
                    println!("-> AIが先手");
                    play_actor(&l5_, &Agent::Human, true);
                } else {
                    println!("Please input valid number.")
                }
            }
        } else if n == 1 {
            play_actor(&l5_, &l5_, true);
            return;
        } else {
            println!("Please input valid number");
        }
    }
}

fn get_random_board(size: usize) -> Board {
    let mut b = Board::new();
    for i in 0..size {
        b = b.next(Agent::Random.get_action(&b));
    }
    return b;
}

fn exp_mate_profile(a1: &impl GetAction, a2: &impl GetAction) {
    use qubic_engine::dfpn::{MateType, ProofNumberSearchStatus};
    let mut b = Board::new();
    let mut mcounts = vec![0; 65];
    let mut ncounts = vec![0; 65];
    let mut scounts = vec![0; 65];
    let mut smax = vec![0; 65];
    let mut tcounts = vec![0; 65];
    let mut tmax = vec![0; 65];

    let mut is_black = true;

    let mut step = 0;

    loop {
        let (att, def) = b.get_att_def();
        println!("att:{att}, def:{def}");
        let start = Instant::now();
        let status = qubic_engine::dfpn::proof_number_search(b.clone());

        match status.typ {
            MateType::NoMate => {}
            _ => {
                println!("{:#?}", status);
            }
        }
        let end = start.elapsed().as_nanos();
        let flag = match status.typ {
            MateType::NoMate => false,
            MateType::Three(_) => false,
            _ => true,
        };
        let (att, def) = b.get_att_def();
        let idx = (att.count_ones() + def.count_ones()) as usize;

        ncounts[idx] += 1;

        let reach_mask = qubic_engine::board::get_reach_mask(att, def);

        if reach_mask != 0 {
            b = get_random_board(4);
            is_black = is_black ^ true;
        } else if flag {
            mcounts[idx] += 1;
            tcounts[idx] += end;
            scounts[idx] += status.size;
            if tmax[idx] < end {
                tmax[idx] = end;
            }
            if smax[idx] < status.size {
                smax[idx] = status.size;
            }
            b = get_random_board(4);
            step += 1;
            // println!(
            //     "{idx}, {}, {}, {}, {end}",
            //     status.path_size, status.valid_nodes, status.reach_boards
            // );

            println!("");
            for i in 0..64 {
                if mcounts[i] == 0 {
                    continue;
                }
                let n = mcounts[i];
                println!(
                    "[{i}]: size:~{}({}), time:~{}({}), m:{}/{}({}%)",
                    smax[i],
                    scounts[i] / n,
                    tmax[i],
                    tcounts[i] / n as u128,
                    n,
                    ncounts[i],
                    100 * n / ncounts[i]
                );
            }
        }

        if b.is_win() || b.is_draw() {
            b = get_random_board(4);
            is_black = is_black ^ true;
        }

        let action;
        if is_black {
            action = a1.get_action(&b);
        } else {
            action = a2.get_action(&b);
        }
        println!("action:{action}");
        b = b.next(action);
        is_black = is_black ^ true;
    }
}

fn profile() {
    let mut b = Board::new();
    b = b.next(0);
    let mut l = SimplLineEvaluator::new();
    // let mut l = PositionMaskEvaluator::new();
    let _ = l.load("simple.json".to_string());
    // let _ = l.load("position.json".to_string());

    let mut nmodel = NNLineEvaluator_::new();
    nmodel.load("sle_tl50_.json".to_string());
    // let mut nmodel = MMEvaluator::from(nmodel);
    let mut long = NegAlphaF::new(Box::new(l), 9);
    long.timelimit = 1000;
    long.min_depth = 9;
    long.scout = true;
    // long.hashmap = true;

    let mut counts: Vec<f32> = vec![0.0; 64];
    let mut search_time: Vec<u128> = vec![0; 64];
    let mut step = 0;
    let mut is_black = true;
    let mut rng = thread_rng();

    loop {
        pprint_board(&b);
        let res = mate_check_horizontal(&b);
        if b.is_win() || b.is_draw() {
            b = Board::new();
            is_black = is_black ^ true;
        }
        if let Some((flag, action)) = res {
            if flag {
                b = Board::new();
                is_black = is_black ^ true;
            } else {
                b = b.next(action);
            }
        }

        let action;

        let start = Instant::now();
        let (action2, val_, _) = long.eval_with_negalpha(&b);
        let t = start.elapsed().as_nanos();

        let stones = b.get_att_def();
        let idx = (stones.0.count_ones() + stones.1.count_ones()) as usize;
        counts[idx] += 1.0;
        search_time[idx] += t;

        if rng.gen::<f32>() < 0.01 {
            action = Agent::Random.get_action(&b);
        } else {
            action = action2;
        }
        println!("action:{action}");
        b = b.next(action);
        step += 1;
        if step % 1 == 0 {
            for i in 0..64 {
                if counts[i] == 0.0 {
                    continue;
                }
                println!(
                    "[{i}] count:{}, time:{}",
                    counts[i],
                    search_time[i] as f32 / counts[i],
                );
            }
            println!(
                "[all] count:{}, time:{}",
                counts.iter().sum::<f32>(),
                search_time.iter().sum::<u128>() as f32 / counts.iter().sum::<f32>(),
            );
            let num = counts[17..].iter().sum::<f32>();
            if num == 0.0 {
                continue;
            }
            println!(
                "[16:] count:{}, time:{}",
                num,
                search_time[17..].iter().sum::<u128>() as f32 / num
            )
        }
    }
}

fn mpc_for_coe(long_depth: u8, short_depth: u8) {
    let mut b = Board::new();
    let mut l = BucketLineEvaluator::new();
    l.load("bsimple.json".to_string());

    let mut long = NegAlphaF::new(Box::new(l.clone()), long_depth);
    let mut short = NegAlphaF::new(Box::new(l.clone()), short_depth);

    let po = PlayoutEvaluator::new(PlayoutLevel::Defence4);
    let mcts = ai::mcts::Mcts::new(10_000, 3, 100, po);
    let mut rng = thread_rng();

    long.min_depth = long_depth;
    short.min_depth = short_depth;
    long.scout = true;
    short.scout = true;

    let mut all_count = 0.0;
    let mut all_err_sum = 0.0;
    let mut all_sq_sum = 0.0;

    let mut counts: Vec<f32> = vec![0.0; 64];
    let mut err_sum: Vec<f32> = vec![0.0; 64];
    let mut err_sq_sum: Vec<f32> = vec![0.0; 64];
    let mut search_time: Vec<u128> = vec![0; 64];
    let mut search_time_a: Vec<u128> = vec![0; 64];
    let mut accuracy: Vec<f32> = vec![0.0; 64];
    let mut step = 0;
    let mut is_black = true;

    loop {
        let res = mate_check_horizontal(&b);
        if b.is_win() || b.is_draw() {
            b = Board::new();
            is_black = is_black ^ true;
        }
        if let Some((flag, action)) = res {
            if flag {
                b = Board::new();
                is_black = is_black ^ true;
            } else {
                b = b.next(action);
                continue;
            }
        }

        let action;
        let start = Instant::now();
        let (action1, val, _) = short.eval_with_negalpha(&b);
        let t2 = start.elapsed().as_nanos();

        let start = Instant::now();
        let (action2, val_, _) = long.eval_with_negalpha(&b);
        let t = start.elapsed().as_nanos();

        let stones = b.get_att_def();
        let idx = (stones.0.count_ones() + stones.1.count_ones()) as usize;
        counts[idx] += 1.0;
        err_sum[idx] += val_ - val;
        err_sq_sum[idx] += (val_ - val).powi(2);
        search_time[idx] += t;
        search_time_a[idx] += t2;
        if action1 == action2 || val == val_ {
            accuracy[idx] += 1.0;
        }

        if rng.gen::<f32>() < 0.1 {
            action = mcts.get_action(&b);
        } else {
            action = action2;
        }
        // let action = m3.get_action(&b);
        b = b.next(action);
        step += 1;
        if step % 1 == 0 {
            for i in 0..64 {
                if counts[i] == 0.0 {
                    continue;
                }
                let mean = err_sum[i] / counts[i];
                let dev = (err_sq_sum[i] / counts[i]) - mean.powi(2);
                println!(
                    "[{i}] mean:{}, dev:{}, std:{}, count:{}, time:{}, accuracy:{}%, rate:{}%",
                    err_sum[i] / counts[i],
                    dev,
                    dev.sqrt(),
                    counts[i],
                    search_time[i] as f32 / counts[i],
                    100.0 * accuracy[i] as f32 / counts[i],
                    100.0 * search_time_a[i] as f32 / search_time[i] as f32
                );
            }
            let mean: f32 = err_sum.iter().sum::<f32>() / counts.iter().sum::<f32>();
            let dev: f32 =
                (err_sq_sum.iter().sum::<f32>() / counts.iter().sum::<f32>()) - mean.powi(2);
            println!(
                "[all] mean:{mean}, dev:{dev}, std:{}, count:{}, time:{}, accuracy:{}%, rate:{}%",
                dev.sqrt(),
                counts.iter().sum::<f32>(),
                search_time.iter().sum::<u128>() as f32 / counts.iter().sum::<f32>(),
                100.0 * accuracy.iter().sum::<f32>() / counts.iter().sum::<f32>(),
                100.0 * search_time_a.iter().sum::<u128>() as f32
                    / search_time.iter().sum::<u128>() as f32,
            );
            let num = counts[17..].iter().sum::<f32>();
            if num == 0.0 {
                continue;
            }
            let mean: f32 = err_sum[17..].iter().sum::<f32>() / num;
            let dev: f32 = (err_sq_sum[17..].iter().sum::<f32>() / num) - mean.powi(2);
            println!(
                "[16:] mean:{mean}, dev:{dev}, std:{}, count:{}, time:{}",
                dev.sqrt(),
                num,
                search_time[17..].iter().sum::<u128>() as f32 / num
            )
        }
    }
}

fn wrapping_line_eval(l: LineEvaluator, depth: u8) -> MateWrapperActor {
    let agent = NegAlphaF::new(Box::new(l), depth);
    let agent = MateWrapperActor::new(Box::new(agent));
    return agent;
}

fn make_db() {
    let mut l = SimplLineEvaluator::new();
    l.load("simple.json".to_string());
    let mut le = NegAlphaF::new(Box::new(l.clone()), 29);
    le.hashmap = true;
    le.min_depth = 7;
    le.timelimit = 50;

    create_db(Some(le), "sle_tl50_genCoe345_insertRandom1_4_48", 5);
}

fn train_line_eval(train_db: String, valid_db: String) {
    // let mut model = LineEvaluator::new();
    // let mut model = TrainableLineEvaluator::from(model, 0.001);
    // model.set_param(0b1_1_00_000000_000000_000000_111111_111111);
    let mut model = SimplLineEvaluator::new();
    let _ = model.load("simple.json".to_string());
    // let mut bigmodel = BucketLineEvaluator::from(model);

    // let n = 32;
    // for i in 0..n {
    //     for j in 0..32 {
    //         print!("{:>5.2},", model.wfl3[i * n + j]);
    //         // print!("{:>5},", model.wgl3[i * n + j] != 0.0);
    //     }
    //     println!("");
    // }
    // return;

    // let mut model = TrainableBLE::from(bigmodel, 0.001);
    let mut model = TrainableSLE::from(model, 0.001);
    // model.set_param(0b1_1_00_000000_000000_000000_111111_111111);
    let mut pmodel = PositionMaskEvaluator::new();
    let mut pmodel = TrainablePME::from(pmodel, 0.001);
    let mut nmodel = NNLineEvaluator_::new();
    let mut nmodel = TrainableNLE_::from(nmodel, 0.005);
    let mut pmodel = pattern::test_pattern_evaluator();
    let mut pmodel = TrainablePatternEvaluator::from_pattern_evaluator(pmodel, 0.0001, 0.9, 0.999);

    qubic_engine::train::train_model_with_db(
        nmodel,
        false,
        true,
        String::from("sle_tl50_.json"),
        String::from("sle_tl50_.json"),
        train_db,
        valid_db,
    );
}

fn expand_untill_n(b: Board, n: usize) -> Vec<Board> {
    use std::collections::HashSet;
    let mut vs: Vec<Board> = Vec::new();
    vs.push(b.clone());
    let mut count = 1;
    for i in 0..n {
        let mut vs_ = Vec::new();
        let mut m = HashSet::new();
        for v in vs {
            for action in v.valid_actions() {
                let nb = v.next(action);
                let hash = nb.hash();
                if m.get(&hash).is_none() {
                    m.insert(hash);
                    vs_.push(nb);
                }
            }
        }
        vs = vs_;
        println!("{i}:{count} -> {}({}%)", vs.len(), vs.len() * 100 / count);
        count = vs.len();
    }
    return vs;
}

fn exp_get_reach_mask() {
    let mut b_time = 0;
    let mut b_time_not = 0;
    let mut a_time = 0;
    let mut a_time_not = 0;
    let n = 1_000_000;
    let mut count_n = 0;
    let mut max_path_board = (0, 0);
    let mut max_path = 0;
    let mut max_reach_count_board = (0, 0);
    let mut max_reach_count = 0;
    let mut max_valid_count_board = (0, 0);
    let mut max_valid_count = 0;
    let mut reach_count = 0;
    let mut valid_count = 0;
    for i in 0..n {
        let mut b = Board::new();
        for j in 0..30 {
            b = b.next(Agent::Random.get_action(&b));
        }

        let (att, def) = b.get_att_def();
        if _is_win_board(att) || _is_win_board(def) {
            continue;
        }
        // let att = 4611686019979307469;
        // let def = 80866198721554;
        // b = Board::from(att, def, qubic_engine::board::Player::Black);
        // pprint_board(&Board::from(att, def, qubic_engine::board::Player::Black));
        //let mask_a = qubic_engine::board::mate_check(&b);
        let mask = qubic_engine::board::get_2row_mask(att, def);
        let start = Instant::now();
        let mate = qubic_engine::dfpn::proof_number_search(b.clone());
        let a_time_ = start.elapsed().as_nanos();
        a_time += a_time_;
        // let a_time_ = start.elapsed().as_nanos();
        // if mask_a.is_some() {
        //    a_time += a_time_;
        //    count_n += 1;
        // } else {
        //     a_time_not += a_time_;
        // }
        let start = Instant::now();
        // let mask_b = qubic_engine::board::mate_check_horizontal(&b);
        let result2 = qubic_engine::dfpn::threat_space_search((att, def));
        b_time += start.elapsed().as_nanos();
    }

    println!(
        "{a_time}|{a_time_not}, {b_time}|{b_time_not}, {}/{}",
        count_n, n
    );
    println!("reach/valid:{reach_count}/{valid_count}");
    println!("max_path:[{:#?}]-{max_path},\nmax_valid_count:[{:#?}]-{max_valid_count},\nmax_reach_count:[{:#?}]-{max_reach_count}", max_path_board, max_valid_count_board, max_reach_count_board);
}

fn trase(b: Board) {
    let mut b = b.clone();
    loop {
        if b.is_win() {
            println!("end");
            pprint_board(&b);
            return;
        }
        let result = qubic_engine::board::mate_check_horizontal(&b);
        let (att, def) = b.get_att_def();
        // let result = qubic_engine::dfpn::threat_space_search((att, def));
        if result.is_none() {
            let action = qubic_engine::board::get_reach_mask(att, def);
            println!("att->{}", action.trailing_zeros() % 16);
            assert!(qubic_engine::board::get_reach_mask(att, def) != 0);
            break;
        } else {
            // let (action) = result.unwrap();
            let ((flag, action)) = result.unwrap();
            // let action = (action.trailing_zeros() % 16) as u8;
            println!("att->{action}");
            let nb = b.next(action);
            if nb.is_win() {
                println!("end");
                pprint_board(&b);
                return;
            }
            let (att, def) = nb.get_att_def();
            let action = qubic_engine::board::get_reach_mask(def, att);
            println!("{action:x}");
            let action = action.trailing_zeros() % 16;
            println!("def->{action}");
            b = nb.next(action as u8);
            pprint_board(&b);
        }
    }
}

fn mcts_statistics() {
    let mut table = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let b = Board::new();
    let b = b.next(0);
    let b = b.next(15);
    let b = b.next(3);
    let agent = Agent::Mcts(500, 50);

    for i in 0..10000 {
        let action = agent.get_action(&b);
        table[action as usize] += 1;
    }

    let mut sort_table = Vec::new();
    for i in 0..16 {
        sort_table.push((i, table[i]));
    }
    sort_table.sort_by(|a, b| a.1.cmp(&b.1).reverse());

    for i in 0..16 {
        println!("action:{:>2}-{}", sort_table[i].0, sort_table[i].1,);
    }
}

fn random_i(max: i32) -> i32 {
    let mut rng = thread_rng();
    rng.gen::<i32>() % max
}

fn get_random_model(max: i32) -> Agent {
    get_position_eval_agent_alpha(
        3,
        random_i(max),
        random_i(max),
        random_i(max),
        random_i(max),
        random_i(max),
        random_i(max),
        random_i(max),
        random_i(max),
        random_i(max),
        random_i(max),
        random_i(max),
        random_i(max),
    )
}

fn explore_best_model() {
    let mut rng = thread_rng();
    let max = 30;

    let tests = vec![
        get_co_eval_agent(3, 3, 1),
        Agent::Mcts(50, 500),
        Agent::Minimax(3),
    ];
    let mut best = get_position_eval_agent_alpha(3, 6, 1, 1, 6, 1, 5, 5, 1, 5, 5, 8, 8);
    let mut max_v = 0.0;

    for i in 0..1000 {
        let tar = get_random_model(max);

        let mut r1 = 0.0;
        let mut r2 = 0.0;
        let mut score = 0.0;

        for test in tests.iter() {
            let (i, r1_, r2_) = compare(&tar, test);
            score += (r1_ + r2_) / r1_;
        }

        let score = 3.0 / score;
        // println!("score:{}, max_v:{}:{}", score, max_v, best.name());

        if max_v < score {
            max_v = score;
            best = tar;
            // println!("[{}]swap! best model is {}", score, best.name());
        }
    }

    // println!("best_model:{}", best.name());
}

fn mc(a: usize, b: usize) -> Agent {
    return Agent::Mcts(a, b);
}

fn get_co_eval_agent(depth: usize, row_w: i32, pos_w: i32) -> Agent {
    use qubic_engine::ai::{CoEvaluator, PositionEvaluator, RowEvaluator};

    let row = RowEvaluator::best();
    let pos = PositionEvaluator::best();

    let eval = CoEvaluator::new(Box::new(row), Box::new(pos), row_w, pos_w);
    let actor = NegAlpha::new(Box::new(eval), depth as u8);

    let agent = Agent::Struct(format!("(r:{row_w},p:{pos_w})[{depth}]"), Box::new(actor));

    return agent;
}

fn get_row_evaluator(depth: usize) -> Agent {
    use qubic_engine::ai::RowEvaluator;
    let eval = RowEvaluator::from(1, 5, 12);
    let actor = NegAlpha::new(Box::new(eval), depth as u8);
    let m3_ = Agent::Struct(format!("(1,5,12)[{depth}]"), Box::new(actor));

    return m3_;
}

fn get_position_eval_agent(depth: usize, v: i32, e: i32, s: i32, c: i32) -> Agent {
    use qubic_engine::ai::PositionEvaluator;

    let eval = PositionEvaluator::simpl(v, e, s, c);
    let actor = NegAlpha::new(Box::new(eval), depth as u8);
    let m3_ = Agent::Struct(
        format!("(v:{v:>3},e:{e:>3},s:{s:>3},c:{c:>3})[{depth}]"),
        Box::new(actor),
    );

    return m3_;
}

fn get_position_eval_agent_alpha(
    depth: usize,
    v0: i32,
    v1: i32,
    v2: i32,
    v3: i32,
    e0: i32,
    e1: i32,
    e2: i32,
    e3: i32,
    s0: i32,
    s1: i32,
    c0: i32,
    c1: i32,
) -> Agent {
    let eval = PositionEvaluator::simpl_alpha(v0, v1, v2, v3, s1, s0, e0, e1, e2, e3, c1, c0);
    let actor = NegAlpha::new(Box::new(eval), depth as u8);
    let m3_ = Agent::Struct(
        format!("({depth},{v0},{v1},{v2},{v3},{e0},{e1},{e2},{e3},{s0},{s1},{c0},{c1})[{depth}]"),
        Box::new(actor),
    );

    return m3_;
}

fn compare(a1: &impl GetAction, a2: &impl GetAction) -> (i32, f32, f32) {
    let (r1, r2, flag) = compare_agent(a1, a2, 200, 0.001, false);

    let r1 = r1.floor();
    let r2 = r2.floor();

    if flag && r1 < r2 {
        println!(
            "[a1]({r1:>4}:lose)\nvs\n[a2]({r2:>4}: win) [{:>4}]",
            // a1.name(),
            // a2.name(),
            r1 + r2
        );
        return (1, r1, r2);
    } else if flag {
        println!(
            "[a1]({r1:>4}: win)\nvs\n[a2]({r2:>4}:lose) [{:>4}]",
            // a1.name(),
            // a2.name(),
            r1 + r2
        );
        return (-1, r1, r2);
    } else {
        println!(
            "[a1]({r1:>4}:draw)\nvs\n[a2]({r2:>4}:draw) [{:>4}]",
            // a1.name(),
            // a2.name(),
            r1 + r2
        );
        return (0, r1, r2);
    }
}

fn get_v(b: &Board) -> u32 {
    use qubic_engine::ai::{acum_mask_bundle, apply_mask_bundle, LineEvaluator};
    let (a, _, _, _, _, _) = LineEvaluator::analyze_board(b.black, b.white);
    let stone = (b.black | b.white);
    let float = !((stone << 16) | 0xffff);
    let float = 0xffff_ffff_ffff_0000;
    let ground = ((stone << 16) | 0xffff) ^ stone;

    let a = apply_mask_bundle(a, float);

    return acum_mask_bundle(a);
}

fn beam_search() {
    use qubic_engine::ai::{acum_mask_bundle, apply_mask_bundle, LineEvaluator};
    let root = Board::from(9373139274136299585, 0, qubic_engine::board::Player::Black);
    let width = 1000;
    let mut stack = Vec::new();
    let mut max_val = 0;
    let mut max_board = root.clone();

    stack.push((root, max_val));
    let mut count = 0;
    loop {
        count += 1;
        let mut new_stack = Vec::new();
        let mut hashmap = HashSet::new();
        if stack.len() == 0 {
            break;
        }

        println!("phase: {count}");
        for (b, n) in stack {
            pprint_board(&b);
            println!("val:{n}");
            for action in 0..64 {
                let stone = b.black | b.white;
                if (stone >> action) & 1 == 1 {
                    continue;
                }
                // let next_b = b.next(action);
                let next_b = Board::from(
                    b.black,
                    b.white | (1 << action),
                    qubic_engine::board::Player::Black,
                );

                if next_b.is_win() {
                    continue;
                }
                if next_b.is_draw() {
                    continue;
                }
                let hash = next_b.hash();
                let res = hashmap.get(&hash);
                if res.is_some() {
                    continue;
                }

                let val = get_v(&next_b);
                new_stack.push((next_b.clone(), val));
                hashmap.insert(hash);

                if val > max_val && next_b.white.count_ones() >= next_b.black.count_ones() - 1 {
                    max_val = val;
                    max_board = next_b;
                }
            }
        }

        println!("{}", new_stack.len());

        if new_stack.len() <= width {
            stack = new_stack;
            continue;
        }

        new_stack.sort_by(|a, b| b.1.cmp(&a.1));
        stack = new_stack.drain(0..width).collect();
    }

    println!("max fl1:{max_val}");
    pprint_board(&max_board);
    pprint_u64(max_board.black);
    println!("black:{}", max_board.black);
    pprint_u64(max_board.white);
    println!("white:{}", max_board.white);
}
