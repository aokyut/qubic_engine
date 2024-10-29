use std::{cmp::Ordering, ops::Neg, process::ExitCode, u64};

use proconio::input;
use qubic_engine::{
    ai::{CoEvaluator, NegAlpha, NullEvaluator, NNUE},
    board::{compare_agent, Agent},
    db::BoardDB,
    exp::cal_elo_rate_diff,
    train::{self, create_db, train_with_db},
    utills::half_imcomplete_beta_func,
};
use rand::seq::SliceRandom;
use rand::thread_rng;
use sqlite::Row;

fn main() {
    use qubic_engine::ai::{Evaluator, NegAlpha, RandomEvaluator, RowEvaluator};
    use qubic_engine::board::*;

    // let a1 = Agent::Mcts(50, 500);
    // let a2 = Agent::Mcts(50, 1000);
    // let mut m = NNUE::default();
    // m.load(format!("test_graph"));
    // m.set_inference();

    // train::train(false, true, format!("test_graph"), 3);

    // println!("input db_name");
    // input! {
    //     name: String
    // }
    // create_db(Some("test_graph"), &name, 5);

    // train_with_db(
    //     false,
    //     true,
    //     String::from("test_graph"),
    //     String::from("record.db"),
    //     String::from("record2.db"),
    // );

    let test_val = CoEvaluator::best();
    let mut test_neg1 = NegAlpha::new(Box::new(test_val), 3);
    let test_val = CoEvaluator::best();
    let mut test_neg2 = NegAlpha::new(Box::new(test_val), 3);
    // let test_agent = Agent::Struct(String::from("hoge"), Box::new(test_neg));

    // play_actor(&test_neg1, &test_neg1, false);

    let (c1, c2, t1, t2) = eval_neg(test_neg1, test_neg2, 100, false);

    println!("count:{c2}/{c1} [{:>4}%]", c2 * 100 / c1);
    println!("time:{t2}/{t1} [{:>4}%]", t2 * 100 / t1);
}

pub fn eval_neg(
    mut a1: NegAlpha,
    mut a2: NegAlpha,
    n: usize,
    render: bool,
) -> (i32, i32, u128, u128) {
    use indicatif::{ProgressBar, ProgressStyle};
    let mut score1 = 0.0;
    let mut score2 = 0.0;

    let pb = ProgressBar::new((n * 2) as u64);
    pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) \n {msg}")
            .unwrap()
            .progress_chars("#>-"));

    for i in 0..n {
        let (s1, s2) = play_neg(&mut a1, &mut a2, render);
        // println!("[{}/{}]black: {}, {}", i, n, s1, s2);
        score1 += s1;
        score2 += s2;
        pb.inc(1);
        pb.set_message(format!("[{s1}, {s2}]"));
        // println!("game black: {}, s1:{}, s2:{}", i, s1, s2);
        let (s2, s1) = play_neg(&mut a2, &mut a1, render);
        // println!("[{}/{}]white: {}, {}", i, n, s1, s2);
        score1 += s1;
        score2 += s2;
        pb.inc(1);
        pb.set_message(format!("[{s1}, {s2}]"));
        // println!("game white: {}, s1:{}, s2:{}", i, s1, s2);
    }
    return (a1.c1 + a2.c1, a1.c2 + a2.c2, a1.t1 + a2.t1, a1.t2 + a2.t2);
}

pub fn play_neg(a1: &mut NegAlpha, a2: &mut NegAlpha, render: bool) -> (f32, f32) {
    use qubic_engine::board::*;
    let mut b = Board::new();
    loop {
        if render {
            pprint_board(&b);
        }
        if b.is_black() {
            let action = a1.get_action_count(&b);
            if render {
                println!("action:{action}");
            }
            b = b.next(action);
            if b.is_win() {
                return (1.0, 0.0);
            } else if b.is_draw() {
                return (0.5, 0.5);
            }
        } else {
            let action = a2.get_action_count(&b);
            if render {
                println!("action:{action}");
            }
            b = b.next(action);
            if b.is_win() {
                return (0.0, 1.0);
            } else if b.is_draw() {
                return (0.5, 0.5);
            }
        }
    }
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

fn compare(a1: &Agent, a2: &Agent) -> i32 {
    let (r1, r2, flag) = compare_agent(a1, a2, 200, 0.001, false);

    let r1 = r1.floor();
    let r2 = r2.floor();

    if flag && r1 < r2 {
        println!(
            "[{}]({r1:>4}:lose) vs [{}]({r2:>4}: win) [{:>4}]",
            a1.name(),
            a2.name(),
            r1 + r2
        );
        return 1;
    } else if flag {
        println!(
            "[{}]({r1:>4}: win) vs [{}]({r2:>4}:lose) [{:>4}]",
            a1.name(),
            a2.name(),
            r1 + r2
        );
        return -1;
    } else {
        println!(
            "[{}]({r1:>4}:draw) vs [{}]({r2:>4}:draw) [{:>4}]",
            a1.name(),
            a2.name(),
            r1 + r2
        );
        return 0;
    }
}
