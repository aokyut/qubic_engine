use qubic_engine::train;

fn main() {
    use qubic_engine::board::*;

    // let a1 = Agent::Mcts(50, 500);
    // let a2 = Agent::Minimax(3);
    // let (s1, s2) = eval_actor(&a1, &a2, 50, false);
    // println!("s1:{s1}, s2:{s2}");
    train::train(false, false, format!("test_graph"));
}
