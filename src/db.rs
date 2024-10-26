use rand::rngs::ThreadRng;
use rand::Rng;
use sqlite::{open, Connection};

use crate::ai::u2vec;
use crate::board::{hash, Board};
use crate::train;
use crate::{
    ml::{create_batch, Tensor},
    train::Transition,
};

pub struct BoardDB {
    conn: Connection,
    batch_size: usize,
    pub batch_num: usize,
    rng: ThreadRng,
    lambda: f32,
}

impl BoardDB {
    pub fn new(s: &str, batch_size: usize) -> Self {
        let conn = open(s).unwrap();

        let query = "
            create table if not exists board_record (
                att integer,
                def integer,
                flag integer,
                val real,
                primary key(att, def)
            )
        ";

        conn.execute(query).unwrap();
        let mut rng = rand::thread_rng();
        let mut db = BoardDB {
            conn: conn,
            batch_size: batch_size,
            batch_num: 0,
            rng: rng,
            lambda: 0.0,
        };
        db.set_batch_num();

        return db;
    }

    pub fn set_lambda(&mut self, lambda: f32) {
        self.lambda = lambda;
    }

    pub fn set_batch_num(&mut self) {
        let count = self.get_count();
        self.batch_num = count / self.batch_size;
    }

    pub fn get_batch_num(&self) -> usize {
        return self.batch_num;
    }

    pub fn add(&self, att: u64, def: u64, flag: i32, val: f32) {
        let h = hash(att, def);
        let att = (h & 0xffff_ffff_ffff_ffff) as u64;
        let def = (h >> 64) as u64;
        let query = format!(
            "
                INSERT INTO board_record(att, def, flag, val)
                VALUES({}, {}, {}, {})
                ON CONFLICT(att, def) DO NOTHING",
            att as i64, def as i64, flag, val
        );
        self.conn.execute(query).unwrap();
    }

    pub fn is_in(&self, board: &Board) -> bool {
        let (att, def) = board.get_att_def();
        let h = hash(att, def);
        let att = (h & 0xffff_ffff_ffff_ffff) as u64;
        let def = (h >> 64) as u64;

        let query = format!(
            "
                SELECT COUNT(*) FROM board_record where att={} and def={}",
            att as i64, def as i64
        );
        let mut count = 0;
        self.conn
            .iterate(query, |pairs| {
                for &(name, value) in pairs.iter() {
                    count = value.unwrap().parse().unwrap();
                }
                true
            })
            .unwrap();
        return count > 0;
    }

    pub fn get(&self, size: usize) -> Vec<(u64, u64, i32, f32)> {
        let query = format!(
            "
                select att, def, flag, val from board_record order by random() limit {}",
            size
        );

        let mut ts = Vec::new();

        self.conn
            .iterate(query, |pairs| {
                let row = pairs.get(0..4).unwrap();
                let att: i64 = row[0].1.unwrap().parse().unwrap();
                let att = att as u64;
                let def: i64 = row[1].1.unwrap().parse().unwrap();
                let def = def as u64;
                let flag: i32 = row[2].1.unwrap().parse().unwrap();
                let val: f32 = row[3].1.unwrap().parse().unwrap();
                ts.push((att, def, flag, val));
                true
            })
            .unwrap();
        return ts;
    }

    pub fn get_count(&self) -> usize {
        let query = "SELECT COUNT(*) FROM board_record";

        let mut count = 0;
        self.conn
            .iterate(query, |pairs| {
                for &(name, value) in pairs.iter() {
                    count = value.unwrap().parse().unwrap();
                }
                true
            })
            .unwrap();
        return count as usize;
    }

    pub fn get_batch(&self) -> Vec<Transition> {
        let query = format!(
            "
                select att, def, flag, val from board_record order by random() limit {}",
            self.batch_size
        );

        let mut ts = Vec::new();

        self.conn
            .iterate(query, |pairs| {
                let row = pairs.get(0..4).unwrap();
                let att: i64 = row[0].1.unwrap().parse().unwrap();
                let att = att as u64;
                let def: i64 = row[1].1.unwrap().parse().unwrap();
                let def = def as u64;
                let flag: i32 = row[2].1.unwrap().parse().unwrap();
                let val: f32 = row[3].1.unwrap().parse().unwrap();
                ts.push(Transition {
                    board: (att as u128) | ((def as u128) << 64),
                    result: flag,
                    t_val: val,
                });
                true
            })
            .unwrap();
        return ts;
    }
}

fn random_rot(b: u128, id: usize) -> u128 {
    let id = id % 8;
    let mut b = b;
    if id < 4 {
        b = Board::hflip(b);
    }
    for i in 0..(id % 4) {
        b = Board::rot(b);
    }
    return b;
}

impl Iterator for BoardDB {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.batch_num == 0 {
            return None;
        } else {
            let mut board = Vec::new();
            let mut result = Vec::new();

            let ts = self.get_batch();
            for t in &ts {
                // pprint_board(&u128_to_b(t.board));
                let res;
                if t.result == 1 {
                    res = 1.0;
                } else if t.result == -1 {
                    res = 0.0;
                } else {
                    res = 0.5;
                }
                // println!("res:{res}, val:{}", t.t_val);
                let rot_b = random_rot(t.board, self.rng.gen());
                board.push(Tensor::new(u2vec(rot_b), vec![128, 1]));
                result.push(Tensor::new(
                    vec![res * train::LAMBDA + (1.0 - train::LAMBDA) * t.t_val],
                    vec![1, 1],
                ));
            }
            let board = create_batch(board);
            let result = create_batch(result);

            self.batch_num -= 1;

            return Some((board, result));
        }
    }
}
