use std::{cmp::Ordering, process::ExitCode, u64};

use qubic_engine::{
    ai::{NullEvaluator, NNUE},
    board::compare_agent,
    board::Agent,
    train,
    utills::half_imcomplete_beta_func,
};
use rand::seq::SliceRandom;
use rand::thread_rng;

fn main() {
    use qubic_engine::ai::{Evaluator, NegAlpha, RowEvaluator};
    use qubic_engine::board::*;

    // let a1 = Agent::Mcts(50, 500);
    // let a2 = Agent::Mcts(50, 1000);
    // let mut m = NNUE::default();
    // m.load(format!("test_graph"));
    // m.set_inference();
    // train::train(false, true, format!("test_graph"));

    // let eval = RowEvaluator::from(4, 6, 9);
    let eval = NullEvaluator::new();
    let actor = NegAlpha::new(Box::new(eval), 3);
    let test_agent = Agent::Struct(format!("(2,5,12)"), Box::new(actor));

    // let eval = RowEvaluator::from(2, 5, 12);
    let eval = NullEvaluator::new();
    let actor = NegAlpha::new(Box::new(eval), 5);
    // let minimax = Agent::Minimax(5);
    let minimax = Agent::Struct(format!("(2,5,12)"), Box::new(actor));

    let res = eval_actor(&test_agent, &minimax, 100, false);
    println!("{res:?}");
    return;

    let mut experiments = vec![(1, 2, 3)];

    for i in 0..10 {
        for j in 0..10 {
            for k in 0..10 {
                experiments.push((i, j, k))
            }
        }
    }

    let mut rng = thread_rng();
    experiments.shuffle(&mut rng);

    let mut agents = vec![];

    for (i, j, k) in experiments {
        let eval = RowEvaluator::from(i, j, k);
        let actor = NegAlpha::new(Box::new(eval), 3);
        let minimax = Agent::Struct(format!("({i},{j},{k})"), Box::new(actor));
        agents.push(minimax);
    }

    for agent in agents {
        let _ = compare(&agent, &test_agent);
    }
}

fn compare(a1: &Agent, a2: &Agent) -> i32 {
    let (r1, r2) = compare_agent(a1, a2, 500, 0.001, false);

    if r1 < r2 {
        println!(
            "[{}](lose:{r1}) vs [{}](win:{r2}) [{}]",
            a1.name(),
            a2.name(),
            r1 + r2
        );
        return 1;
    } else {
        println!(
            "[{}](win:{r1}) vs [{}](lose:{r2}) [{}]",
            a1.name(),
            a2.name(),
            r1 + r2
        );
        return -1;
    }
}
