use std::{cmp::Ordering, process::ExitCode, u64};

use proconio::input;
use qubic_engine::{
    ai::{NegAlpha, NullEvaluator, NNUE},
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

    println!("input db_name");
    input! {
        name: String
    }
    create_db(Some("test_graph"), &name, 3);

    // train_with_db(
    //     false,
    //     true,
    //     String::from("test_graph"),
    //     String::from("record.db"),
    //     String::from("record2.db"),
    // );
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
