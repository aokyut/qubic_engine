use proconio::input;
use qubic_engine::ai::{
    LineEvaluator, MateNegAlpha, MateWrapperActor, NegAlphaF, PositionEvaluator,
    TrainableLineEvaluator,
};
use qubic_engine::board::{count_2row_, get_random, mate_check_horizontal, Board, GetAction};
use qubic_engine::db::BoardDB;
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
    let m3_f32 = NegAlphaF::new(Box::new(CoEvaluator::best()), 3);
    let m4 = NegAlpha::new(Box::new(CoEvaluator::best()), 4);
    let m5 = NegAlpha::new(Box::new(CoEvaluator::best()), 5);
    let m6 = NegAlpha::new(Box::new(CoEvaluator::best()), 6);
    let m7 = NegAlpha::new(Box::new(CoEvaluator::best()), 7);
    let mm3 = MateNegAlpha::new(Box::new(CoEvaluator::best()), 5);
    let mm3 = Agent::Struct(String::from("mm3"), Box::new(mm3));
    let m2 = Agent::Struct(String::from("m2"), Box::new(m2));
    let m3 = Agent::Struct(String::from("m3"), Box::new(m3));
    let m5 = Agent::Struct(String::from("m5"), Box::new(m5));
    let mut l = LineEvaluator::new();
    l.load("wR5_gR_ir1_48.leval".to_string());
    let l3 = wrapping_line_eval(l.clone(), 3);
    let l4 = wrapping_line_eval(l.clone(), 4);
    let l5 = wrapping_line_eval(l.clone(), 5);
    l.load("wR5_gR_ir1_48_t.leval".to_string());
    let mut l5_ = NegAlphaF::new(Box::new(l.clone()), 3);
    l5_.hashmap = false;
    let l5_ = MateWrapperActor::new(Box::new(l5_));
    // let l6 = wrapping_line_eval(l.clone(), 6);
    // let l7 = wrapping_line_eval(l.clone(), 7);
    // let l8 = wrapping_line_eval(l.clone(), 8);
    // let test = NegAlpha::new(Box::new(PositionEvaluator::simpl_alpha(1, 0, 0, 0, 0, 0)), 3);

    // let att = 9361298940875284;
    // let def = 4758053146183082217;
    // let b = Board::from(att, def, Player::Black);
    // pprint_board(&b);
    // let _ = l5_.eval_with_negalpha(&b);

    make_db();

    // play_actor(&l5_, &l3, true);

    // let db = BoardDB::new("mcoe3_insertRandom48_4_decay092", 0);
    // let db_ = BoardDB::new("mcoe3_insertRandom48_4_decay092_", 0);
    // db.concat(db_);

    // explore_best_model();

    // let start = Instant::now();
    // let result = eval_actor(&l5_, &l3, 100, false);
    // println!("time:{}", start.elapsed().as_nanos());
    // println!("{result:#?}");
    // return;
    // let (a, b, c) = compare(&m2, &mm3);

    // exp_count_2row_();
    // println!("{}", 3 * 3 & 1)
    // let att = 2314852547162056300;
    // let def = 360328721413243153;
    // let b = Board::from(att, def, Player::Black);
    // pprint_board(&b);
    // println!("");
    // let a = qubic_engine::board::get_reach_mask_alpha(att, def);
    // let b = qubic_engine::board::get_reach_mask(att, def);
    // pprint_u64(a);
    // println!("");
    // pprint_u64(b);

    // train_with_db(
    //     false,
    //     true,
    //     String::from("wr_coe3_8_5_5_ir48_4_d092"),
    //     String::from("winRate_coe5_genRandom_insertRandom1_48"),
    //     String::from("winRate_coe5_genRandom_insertRandom1_48_test"),
    // );
    // train_line_eval();
    // mpc_for_coe(5, 5);
}

fn mpc_for_coe(long_depth: u8, short_depth: u8) {
    let mut b = Board::new();
    let mut l = LineEvaluator::new();
    l.load("wR5_gR_ir1_48.leval".to_string());
    let long = NegAlphaF::new(Box::new(l.clone()), long_depth);
    let short = NegAlphaF::new(Box::new(l.clone()), short_depth);

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
            b = Board::new();
            is_black = is_black ^ true;
        }

        let action;
        let start = Instant::now();
        let (action1, val, _) = short.eval_with_negalpha_hash(&b);
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

        action = Agent::Random.get_action(&b);

        // let action = m3.get_action(&b);
        b = b.next(action);
        step += 1;
        if step % 10 == 0 {
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
    let mut l = LineEvaluator::new();
    l.load("wR5_gR_ir1_48.leval".to_string());
    let le = NegAlphaF::new(Box::new(l.clone()), 5);

    create_db(Some(le), "le5_genCoe345_insertRandom1_4_48", 5);
}

fn train_line_eval() {
    let mut model = LineEvaluator::new();
    let mut model = TrainableLineEvaluator::from(model, 0.001);
    model.set_param(0b1_0_00_000000_000000_111111_000000_000000);

    qubic_engine::train::train_model_with_db(
        model,
        true,
        true,
        String::from("wR5_gR_ir1_48_pos.leval"),
        String::from("wR5_gR_ir1_48_pos.leval"),
        String::from("winRate_coe5_genRandom_insertRandom1_48"),
        String::from("winRate_coe5_genRandom_insertRandom1_48_test"),
    );
}

fn exp_count_2row_() {
    let mut b_time = 0;
    let mut a_time = 0;
    for i in 0..100000 {
        let mut b = Board::new();
        for j in 0..30 {
            b = b.next(Agent::Random.get_action(&b));
        }
        let (att, def) = b.get_att_def();
        let start = Instant::now();
        let mask_a = qubic_engine::board::get_reach_mask_alpha(att, def);
        a_time += start.elapsed().as_nanos();
        let start = Instant::now();
        let mask_b = qubic_engine::board::get_reach_mask(att, def);
        b_time += start.elapsed().as_nanos();
        assert_eq!(mask_a, mask_b, "[{i}]att:{att}, def:{def}");
    }

    println!("{a_time}, {b_time}");
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
