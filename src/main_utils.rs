use qubic_engine::{
    ai,
    board::{self, Board, GetAction, Player},
    dfpn::{self, proof_number_search, threat_space_search, MateType, ProofNumberSearchStatus},
};
use std::thread::sleep;
use std::time::{self, Instant};

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
        if self.b.is_win() {
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
    let att = 0x00000000000748a9;
    let def = 0x000000000008b746;
    let mut b = Board::from(att, def, Player::Black);
    println!("att:{att:>016x}, def:{def:>016x}");
    let status = proof_number_search(b.clone());

    pprint_board(&b);

    let mut l = ai::line::SimplLineEvaluator::new();
    l.load("simple.json".to_string());

    let mut la = ai::NegAlphaF::new(Box::new(l.clone()), 29);
    la.scout = true;
    la.timelimit = 10000;
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

    loop {
        let po = ai::PlayoutEvaluator::new(ai::PlayoutLevel::Defence4);
        let mcts = ai::mcts::Mcts::new(10000, 3, 10, po);
        let po = ai::PlayoutEvaluator::new(ai::PlayoutLevel::Defence4);
        let mcts2 = ai::mcts::Mcts::new(10000, 3, 10, po);
        let m1 = ai::NegAlpha::new(Box::new(ai::CoEvaluator::best()), 3);
        let m2 = ai::NegAlpha::new(Box::new(ai::CoEvaluator::best()), 3);
        let mut l1 = ai::NegAlphaF::new(Box::new(l.clone()), 5);
        let mut l2 = ai::NegAlphaF::new(Box::new(l.clone()), 5);
        for board in GameRecordIterator::new(mcts, mcts2) {
            sleep(time::Duration::from_millis(10));
            let (att, def) = board.get_att_def();

            let stone = att | def;
            let stone2 = (stone >> 16) & 0xffff;
            let stone1 = stone & 0xffff;

            assert_eq!(stone2 & stone1, stone2, "att:{att:>016x}, def:{def:016x}");
            // println!("{att:>016x}, {def:>016x}");
            let res = threat_space_search((att, def));
            if let Some(_) = res {
                break;
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
