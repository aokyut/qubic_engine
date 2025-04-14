use proconio::input;
use qubic_engine::ai::{MateNegAlpha, PositionEvaluator};
use qubic_engine::board::get_random;
use qubic_engine::train::{create_db, train_with_db};
use qubic_engine::{
    ai::{CoEvaluator, NegAlpha, NNUE},
    board::{compare_agent, Agent},
};
use rand::rngs::ThreadRng;
use rand::thread_rng;
use rand::Rng;
use std::ops::Neg;
use std::os::unix::thread;
use std::time::{Duration, Instant};

fn main() {
    use qubic_engine::ai::NegAlpha;
    use qubic_engine::board::*;

    // let a1 = Agent::Mcts(50, 5000);
    let m2 = NegAlpha::new(Box::new(CoEvaluator::best()), 2);
    let m3 = NegAlpha::new(Box::new(CoEvaluator::best()), 3);
    let m4 = NegAlpha::new(Box::new(CoEvaluator::best()), 4);
    let m5 = NegAlpha::new(Box::new(CoEvaluator::best()), 5);
    // let m6 = NegAlpha::new(Box::new(CoEvaluator::best()), 6);
    let m7 = NegAlpha::new(Box::new(CoEvaluator::best()), 7);
    let mm3 = MateNegAlpha::new(Box::new(CoEvaluator::best()), 5);
    let mm3 = Agent::Struct(String::from("mm3"), Box::new(mm3));
    let m2 = Agent::Struct(String::from("m2"), Box::new(m2));
    let m3 = Agent::Struct(String::from("m3"), Box::new(m3));
    let m5 = Agent::Struct(String::from("m5"), Box::new(m5));

    // let test = NegAlpha::new(Box::new(PositionEvaluator::simpl_alpha(1, 0, 0, 0, 0, 0)), 3);

    let mut m = NNUE::default();
    m.set_depth(3);
    m.load(format!("mcoe3_8_5_5_ir48_4_d092"));
    m.set_inference();
    // let mut m_ = NNUE::default();
    // m_.set_depth(4);
    // m_.load(format!("coe3_8_5_5_fr12_d092"));
    // m_.set_inference();

    // train::train(false, true, format!("test_graph"), 3);

    // println!("input db_name");
    // input! {
    //     name: String
    // }
    // test_is_win();
    create_db(None, "mcoe3_insertRandom48_4_decay092", 3);

    // let test = Agent::Minimax(3);
    // let agent = Agent::Minimax(5);

    // println!("{mask}");
    // play_actor(&m, &m3, true);

    // explore_best_model();

    // let start = Instant::now();
    // let hoge = get_position_eval_agent_alpha(3, -2, 3, -16, 0, -4, -12, 17, -13, 20, 1, 0, 22);
    // let result = eval_actor(&m5, &mm3, 100, false);
    // let (a, b, c) = compare(&m2, &mm3);
    // println!("{result:#?}");

    // train_with_db(
    //     false,
    //     true,
    //     String::from("mcoe3_8_5_5_ir48_4_d092"),
    //     String::from("mcoe3_insertRandom48_4_decay092"),
    //     String::from("mcoe3_insertRandom48_4_decay092_test"),
    // );

    // qubic_engine::ai::pattern::train_with_db(
    //     false,
    //     false,
    //     String::from("test"),
    //     String::from("coe3_fromRandom12_decay092"),
    //     String::from("coe3_fromRandom12_decay092_test"),
    //     100,
    // );
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
        println!("score:{}, max_v:{}:{}", score, max_v, best.name());

        if max_v < score {
            max_v = score;
            best = tar;
            println!("[{}]swap! best model is {}", score, best.name());
        }
    }

    println!("best_model:{}", best.name());
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

fn compare(a1: &Agent, a2: &Agent) -> (i32, f32, f32) {
    let (r1, r2, flag) = compare_agent(a1, a2, 200, 0.001, false);

    let r1 = r1.floor();
    let r2 = r2.floor();

    if flag && r1 < r2 {
        println!(
            "[{}]({r1:>4}:lose)\nvs\n[{}]({r2:>4}: win) [{:>4}]",
            a1.name(),
            a2.name(),
            r1 + r2
        );
        return (1, r1, r2);
    } else if flag {
        println!(
            "[{}]({r1:>4}: win)\nvs\n[{}]({r2:>4}:lose) [{:>4}]",
            a1.name(),
            a2.name(),
            r1 + r2
        );
        return (-1, r1, r2);
    } else {
        println!(
            "[{}]({r1:>4}:draw)\nvs\n[{}]({r2:>4}:draw) [{:>4}]",
            a1.name(),
            a2.name(),
            r1 + r2
        );
        return (0, r1, r2);
    }
}
