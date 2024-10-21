use std::u64;

use qubic_engine::{ai::NNUE, train};

fn main() {
    use qubic_engine::board::*;

    // let a1 = Agent::Mcts(50, 500);
    // let a2 = Agent::Mcts(50, 1000);
    // let mut m = NNUE::default();
    // m.load(format!("test_graph"));
    // m.set_inference();
    // let result = eval_actor(&a1, &a2, 50, false);
    // println!("{result:?}")
    // train::train(false, true, format!("test_graph"));

    train::create_db("test_graph", "record.db");
}
