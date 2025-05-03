use crate::board::{get_random, mate_check_horizontal, Board};

use super::board;
use rand::Rng;
use std::{collections::binary_heap::Iter, thread, time};

const K: f32 = 16.0;
const START: f32 = 1500.0;

pub struct Rating {
    agents: Vec<board::Agent>,
    rates: Vec<f32>,
    isfix: Vec<bool>,
    pub temp: f32,
}

fn cal_rate(winner_loser: (f32, f32), temp: f32) -> (f32, f32) {
    let (winner, loser) = winner_loser;
    let point = temp * K / (10.0_f32.powf((winner - loser) / 400.0) + 1.0);
    return (winner + point, loser - point);
}

pub fn cal_elo_rate_diff(win_rate_a_to_b: f32) -> f32 {
    assert!(win_rate_a_to_b != 0.0);
    return -400.0 * (1.0 / win_rate_a_to_b - 1.0).log10();
}

impl Rating {
    pub fn new(in_vec: Vec<board::Agent>) -> Self {
        let size = in_vec.len();
        let mut rates: Vec<f32> = Vec::new();
        let mut isfix: Vec<bool> = Vec::new();
        for _ in 0..size {
            rates.push(START);
            isfix.push(false);
        }
        return Rating {
            agents: in_vec,
            rates: rates,
            isfix: isfix,
            temp: 1.0,
        };
    }

    pub fn from(agents: Vec<board::Agent>, rates: Vec<f32>) -> Self {
        let size = agents.len();
        assert_eq!(agents.len(), rates.len());
        let mut isfix: Vec<bool> = Vec::new();
        for _ in 0..size {
            isfix.push(false);
        }
        return Rating {
            agents: agents,
            rates: rates,
            isfix: isfix,
            temp: 1.0,
        };
    }

    pub fn setfix(&mut self, idx: usize) {
        self.isfix[idx] = true
    }

    pub fn playn(&mut self, n: usize) {
        let wait_time = time::Duration::from_millis(1000);
        for i in 0..n {
            println!("[{}], temp:{}", i, self.temp * K);
            let start = time::Instant::now();
            self.play();
            // self.temp *= 0.998;
            let end = start.elapsed();
            thread::sleep(end.mul_f32(3.0));
        }
    }

    pub fn matching(&mut self) -> (usize, usize) {
        let mut rng = rand::thread_rng();
        let size = self.agents.len();
        loop {
            let idx1 = rng.gen::<usize>() % size;
            let idx2 = (idx1 + 1 + (rng.gen::<usize>() % (size - 1))) % size;
            if !self.isfix[idx1] | !self.isfix[idx2] {
                return (idx1, idx2);
            }
        }
    }

    pub fn play(&mut self) {
        // let mut rng = rand::thread_rng();
        // let size = self.agents.len();
        let (idx1, idx2) = self.matching();
        // let idx1 = rng.gen::<usize>() % size;
        // let idx2 = (idx1 + 1 + (rng.gen::<usize>() % (size - 1))) % size;
        let a1 = &self.agents[idx1];
        let a2 = &self.agents[idx2];
        let rate1 = self.rates[idx1];
        let rate2 = self.rates[idx2];
        println!("{}[{}] vs {}[{}]", a1.name(), rate1, a2.name(), rate2);
        let (s1, s2) = board::eval(&a1, &a2, 1);

        let (new_rate1, new_rate2);
        if s1 == 1.0 {
            (new_rate1, new_rate2) = cal_rate((rate1, rate2), self.temp);
            println!(
                "  {}[{} -> {}({})] win",
                a1.name(),
                rate1,
                new_rate1,
                new_rate1 - rate1
            );
        } else if s1 == 0.0 {
            (new_rate2, new_rate1) = cal_rate((rate2, rate1), self.temp);
            println!(
                "  {}[{} -> {}({})] win",
                a2.name(),
                rate2,
                new_rate2,
                new_rate2 - rate2
            );
        } else {
            let (_new_rate1, _new_rate2) = cal_rate((rate1, rate2), self.temp);
            (new_rate2, new_rate1) = cal_rate((_new_rate2, _new_rate1), self.temp);
            println!("  [{} -> {}({})] draw", rate1, new_rate1, new_rate1 - rate1);
        }

        if !self.isfix[idx1] {
            self.rates[idx1] = new_rate1;
        }
        if !self.isfix[idx2] {
            self.rates[idx2] = new_rate2;
        }
    }

    pub fn print(&self) {
        for i in 0..self.agents.len() {
            println!("{:<15}: {}", self.agents[i].name(), self.rates[i]);
        }
    }
}

pub struct BoardIter {
    pub b: Board,
    mate_end: bool,
}

impl BoardIter {
    pub fn new(mate_end: bool) -> Self {
        return BoardIter {
            b: Board::new(),
            mate_end: mate_end,
        };
    }
}

impl Iterator for BoardIter {
    type Item = Board;
    fn next(&mut self) -> Option<Self::Item> {
        let mate = mate_check_horizontal(&self.b);
        if let Some((flag, _)) = mate {
            if flag && self.mate_end {
                self.b = Board::new();
            }
        }
        if self.b.is_win() {
            self.b = Board::new();
        } else if self.b.is_draw() {
            self.b = Board::new();
        }

        let b = self.b.clone();
        self.b = self.b.next(get_random(&self.b));

        return Some(b);
    }
}
