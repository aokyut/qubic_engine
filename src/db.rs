use sqlite::{open, Connection};

use crate::ai::NNUEHash;
use crate::board::Board;
use crate::train;
use crate::{train::Transition, utills::rand::get_random_usize};
use minimum_ml::dataset::{Dataset, Stackable};
use minimum_ml::ml::Tensor;
use std::hash::Hash;
use std::marker::PhantomData;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct WeightedTransition{
    pub board: u128,
    pub result: f32,
    pub val: f32,
    pub weight: f32
}

#[derive(Clone, Stackable)]
pub struct BoardData {
    pub input: Tensor,
    pub label: Tensor,
    pub weight: Tensor,
}

pub struct StepbackBoardDB {
    conn: Connection,
    stepback_alpha: f32,
    lambda: f32,
    weight: HashMap<usize, f32>,
}

impl StepbackBoardDB {
    pub fn new(s: &str, stepback_alpha: f32, use_result_lambda: f32) -> Self {
        let conn = open(s).unwrap();

        let query = "
            create table if not exists stepback_board_record (
                att integer,
                def integer,
                result real,
                backstep integer,
                frontstep integer,
                val real
            )
        ";

        conn.execute(query).unwrap();
        let mut db = StepbackBoardDB {
            conn: conn,
            stepback_alpha: stepback_alpha,
            lambda: use_result_lambda,
            weight: HashMap::new()
        };

        let weight = db.get_weight();
        db.weight = weight;

        return db;
    }

    pub fn get_weight(&self) -> HashMap<usize, f32>{
        let count = self.get_count() as f32;
        let mut hashmap = HashMap::new();
        for i in 0..100{
            let (min, max) = ((i as f32) / 100.0, (i as f32) / 100.0 + 0.01);
            let mut num = 0;
            let s = format!("SELECT COUNT(*) FROM stepback_board_record where val >= {min} and val <= {max}");
            let query = s.as_str();
            self.conn
            .iterate(query, |pairs| {
                for &(name, value) in pairs.iter() {
                    num = value.unwrap().parse().unwrap();
                }
                true
            })
            .unwrap();
            let weight = count / (100.0 * (num + 1) as f32);
            hashmap.insert(i, weight);
        }
        return hashmap;
    }

    pub fn get_count(&self) -> usize {
        let query = "SELECT COUNT(*) FROM stepback_board_record";

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

    pub fn len(&self) -> usize {
        return self.get_count();
    }

    pub fn add(&self, att: u64, def: u64, result: f32, backstep: u64, frontstep: u64, val: f32) {
        let (unique_att, unique_def) = Board::normalize(att, def);

        let query = format!(
            "
            insert into stepback_board_record(att, def, result, backstep, frontstep, val)
            values({}, {}, {}, {}, {}, {})
        ",
            unique_att as i64, unique_def as i64, result, backstep as i64, frontstep as i64, val,
        );

        self.conn.execute(query).unwrap();
    }

    pub fn get_all(&self) -> Vec<WeightedTransition> {
        let query = format!(
            "
                select att, def, result, backstep, frontstep, val from stepback_board_record",
        );

        let mut ts = Vec::new();

        self.conn
            .iterate(query, |pairs| {
                let row = pairs.get(0..6).unwrap();
                let att: i64 = row[0].1.unwrap().parse().unwrap();
                let att = att as u64;
                let def: i64 = row[1].1.unwrap().parse().unwrap();
                let def = def as u64;
                let result: f32 = row[2].1.unwrap().parse().unwrap();
                let backstep = row[3].1.unwrap().parse::<i64>().unwrap() as i32;
                let frontstep = row[4].1.unwrap().parse::<i64>().unwrap() as u64;
                let val: f32 = row[5].1.unwrap().parse().unwrap();

                let scaled_result = (result - 0.5) * self.stepback_alpha.powi(backstep) + 0.5;
                let val = scaled_result * self.lambda + val * (1.0 - self.lambda);

                let weight_id = if val == 1.0 {
                    99
                }else{
                    (val * 100.0).floor() as usize
                };
                let weight = *self.weight.get(&weight_id).unwrap() * (1.0 - (val - 0.5).powi(2) * 2.0);


                ts.push(WeightedTransition {
                    board: (att as u128) | ((def as u128) << 64),
                    result: result,
                    val: val,
                    weight: weight
                });
                true
            })
            .unwrap();
        return ts;
    }
}

pub struct BoardDB {
    conn: Connection,
    batch_size: usize,
    pub batch_num: usize,
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
                val real
            )
        ";

        conn.execute(query).unwrap();
        let mut db = BoardDB {
            conn: conn,
            batch_size: batch_size,
            batch_num: 0,
            lambda: 0.0,
        };

        if batch_size == 0 {
            let count = db.get_count();
            db.batch_size = count;
        }

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

    pub fn len(&self) -> usize {
        self.get_count()
    }

    pub fn begine(&self) {
        let _ = self.conn.execute("begine");
    }

    pub fn end(&self) {
        let _ = self.conn.execute("end");
    }

    pub fn add(&self, att: u64, def: u64, flag: i32, val: f32) {
        let query = format!(
            "
                insert into board_record(att, def, flag, val)
                values({}, {}, {}, {})",
            att as i64, def as i64, flag, val
        );

        self.conn.execute(query).unwrap();
    }

    pub fn concat(&self, other: BoardDB) {
        let query = format!(
            "
                select att, def, flag, val from board_record",
        );

        // let mut ts = Vec::new();
        let mut count = 0;

        self.begine();
        println!("merge BoardDB");
        other
            .conn
            .iterate(query, |pairs| {
                if count % 1000 == 0 {
                    println!("{count}");
                }
                count += 1;
                let row = pairs.get(0..4).unwrap();
                let att: i64 = row[0].1.unwrap().parse().unwrap();
                let att = att as u64;
                let def: i64 = row[1].1.unwrap().parse().unwrap();
                let def = def as u64;
                let flag: i32 = row[2].1.unwrap().parse().unwrap();
                let val: f32 = row[3].1.unwrap().parse().unwrap();
                self.add(att, def, flag, val);
                true
            })
            .unwrap();
        self.end();
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
                    result: (flag as f32) * 0.5 + 0.5,
                    val: val,
                });
                true
            })
            .unwrap();
        return ts;
    }

    pub fn get_all(&self) -> Vec<WeightedTransition> {
        let query = format!(
            "
                select att, def, flag, val from board_record",
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
                ts.push(WeightedTransition {
                    board: (att as u128) | ((def as u128) << 64),
                    result: (flag as f32) * 0.5 + 0.5,
                    val: val,
                    weight: 1.0
                });
                true
            })
            .unwrap();
        return ts;
    }
}

pub fn random_rot(b: u128, id: usize) -> u128 {
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

// BoardDataset: Wrapper around BoardDB that implements Dataset with NNUEHash support
// Caches all data in memory for fast random access
pub struct BoardDataset<H: NNUEHash = crate::ai::BundleHash> {
    data: Vec<WeightedTransition>,
    _hash: PhantomData<H>,
}

impl<H: NNUEHash> BoardDataset<H> {
    pub fn new(db_path: &str, batch_size: usize) -> Self {
        let db = BoardDB::new(db_path, batch_size);
        println!("Loading dataset into memory...");
        let data = db.get_all();
        println!("Loaded {} records", data.len());
        BoardDataset {
            data,
            _hash: PhantomData,
        }
    }

    pub fn from_stepback_db(db_path: &str, stepback_alpha: f32, lambda: f32) -> Self {
        let db = StepbackBoardDB::new(db_path, stepback_alpha, lambda);
        let data = db.get_all();
        BoardDataset {
            data,
            _hash: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl<H: NNUEHash> Dataset for BoardDataset<H> {
    type Item = BoardData;

    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, index: usize) -> Self::Item {
        let t = &self.data[index];

        let res = t.result;
        let rot_b = random_rot(t.board, get_random_usize());
        let input = Tensor::new(H::to_input_vec(rot_b), vec![H::feature_count()]);
        let label = Tensor::new(
            vec![t.val],
            vec![1],
        );
        let weight = Tensor::new(
            vec![t.weight],
            vec![1],
        );

        BoardData { input, label, weight }
    }
}
