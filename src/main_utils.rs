use qubic_engine::{
    ai,
    board::{self, Board, GetAction, Player},
    dfpn::{self, proof_number_search, threat_space_search, MateType, ProofNumberSearchStatus},
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::prelude::*;
use std::thread::sleep;
use std::time::{self, Instant};

pub fn read_board_from_json(file_name: &str) -> Vec<(u64, u64)> {
    let deserialized: Vec<(u64, u64)> = serde_json::from_str(&file_name).unwrap();
    return deserialized;
}

pub fn write_board_to_json(v: Vec<(u64, u64)>, file_name: &str) -> std::io::Result<()> {
    let serialized = serde_json::to_string(&v).unwrap();
    let mut file = fs::File::create(file_name)?;
    file.write_all(serialized.as_bytes())?;
    Ok(())
}

pub struct GameRecordIterator<A1: GetAction, A2: GetAction> {
    a1: A1,
    a2: A2,
    b: Board,
}

impl<A1: GetAction, A2: GetAction> GameRecordIterator<A1, A2> {
    pub fn new(a1: A1, a2: A2) -> Self {
        return GameRecordIterator {
            a1,
            a2,
            b: Board::new(),
        };
    }

    pub fn reset(&mut self) -> Board {
        self.b = Board::new();
        return self.b.clone();
    }
}

impl<A1: GetAction, A2: GetAction> Iterator for GameRecordIterator<A1, A2> {
    type Item = Board;

    fn next(&mut self) -> Option<Self::Item> {
        if self.b.is_win() || self.b.is_draw() {
            return None;
        }
        let action = match self.b.player {
            Player::Black => self.a1.get_action(&self.b),
            Player::White => self.a2.get_action(&self.b),
        };
        self.b = self.b.next(action);
        return Some(self.b.clone());
    }
}

pub fn test_pns() {
    use qubic_engine::board::pprint_board;
    let att = 0x000000020043a163;
    let def = 0x0002006004201e88;
    let mut b = Board::from(att, def, Player::Black);
    println!("att:{att:>016x}, def:{def:>016x}");
    let status = proof_number_search(b.clone());

    pprint_board(&b);

    let mut l = ai::line::SimplLineEvaluator::new();
    l.load("simple.json".to_string());

    let mut la = ai::NegAlphaF::new(Box::new(l.clone()), 29);
    la.scout = true;
    la.timelimit = 1000;
    la.min_depth = 7;

    println!("{:#?}", status);
    println!("{:#?}", la.eval_with_negalpha(&b));
}

pub fn generate_problems() {
    println!("att, def, stone, time, size, type, val");
    let mut l = ai::line::SimplLineEvaluator::new();
    l.load("simple.json".to_string());

    let mut la = ai::NegAlphaF::new(Box::new(l.clone()), 29);
    la.scout = true;
    la.timelimit = 10;
    la.min_depth = 5;

    let mut problems = vec![];
    let mut problems_no_tss = vec![];

    loop {
        let po = ai::PlayoutEvaluator::new(ai::PlayoutLevel::Defence4);
        let mcts = ai::mcts::Mcts::new(10000, 3, 10, po);
        let po = ai::PlayoutEvaluator::new(ai::PlayoutLevel::Defence4);
        let mcts2 = ai::mcts::Mcts::new(10000, 3, 10, po);
        let m1 = ai::NegAlpha::new(Box::new(ai::CoEvaluator::best()), 3);
        let m2 = ai::NegAlpha::new(Box::new(ai::CoEvaluator::best()), 3);
        let mut l1 = ai::NegAlphaF::new(Box::new(l.clone()), 5);
        let mut l2 = ai::NegAlphaF::new(Box::new(l.clone()), 5);
        for board in GameRecordIterator::new(board::Agent::Random, board::Agent::Random) {
            if problems.len() == 10000 {
                write_board_to_json(problems, "tss10000.json");
                write_board_to_json(problems_no_tss, "no_tss10000.json");
                return;
            }

            sleep(time::Duration::from_millis(10));
            let (att, def) = board.get_att_def();

            let stone = att | def;
            let stone2 = (stone >> 16) & 0xffff;
            let stone1 = stone & 0xffff;

            assert_eq!(stone2 & stone1, stone2, "att:{att:>016x}, def:{def:016x}");
            // println!("{att:>016x}, {def:>016x}");
            let start = Instant::now();
            let res = dfpn::threat_space_search_alpha((att, def));
            let time = start.elapsed().as_nanos();
            if let Some((a, b)) = res {
                if time < 10_000 || problems.len() >= 10000 {
                    continue;
                }
                println!("{att:>016x}, {def:>016x}, {time}, {}", b.path_size);
                problems.push((att, def));
                break;
            } else {
                if time < 100_000 || problems_no_tss.len() >= 10000 {
                    continue;
                }
                println!("{att:>016x}, {def:>016x}, {time}, no_mate");
                problems_no_tss.push((att, def));
                continue;
            }
            let start = Instant::now();
            let res = proof_number_search(board.clone());
            let time = start.elapsed().as_nanos();

            if res.size <= 1 {
                continue;
            }

            let flag = match res.typ {
                MateType::Two(_) => "two",
                MateType::NoMate => "no",
                MateType::Three(_) => "three",
            };

            if flag == "no" {
                continue;
            }

            println!(
                "{att:>016x}, {def:>016x}, {}, {}.{:>06}, {}, {}, {}",
                (att | def).count_ones(),
                time / 1000_000,
                time % 1000_000,
                res.size,
                flag,
                la.eval_with_negalpha(&board).1,
            );
        }
    }
}
