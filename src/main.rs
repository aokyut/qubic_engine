use qubic_engine::{ai::NNUE, train};

fn main() {
    use qubic_engine::board::*;

    // let a1 = Agent::Mcts(50, 500);
    // let a2 = Agent::Minimax(3);
    // let mut m = NNUE::default();
    // m.load(format!("test_graph"));
    // m.set_inference();
    // let result = play_actor(&a2, &m, true);
    // println!("{result:?}")
    train::train(false, true, format!("test_graph"));
}
