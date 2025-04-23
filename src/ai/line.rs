use crate::board::Board;

use super::{acum_mask_bundle, acum_or, apply_mask_bundle, EvaluatorF, LineEvaluator, Trainable};
use anyhow::{Ok, Result};
use serde::{Deserialize, Serialize};

const WFL3_WIDTH: usize = 32;
const WGL3_WIDTH: usize = 16;
const WL2_WIDTH: usize = 64;
const WGL1_WIDTH: usize = 96;
const WFL1_WIDTH: usize = 106;

#[derive(Clone, Serialize, Deserialize)]
pub struct SimplLineEvaluator {
    pub wfl3: Vec<f32>,
    pub wgl3: Vec<f32>,
    pub wfl2: Vec<f32>,
    pub wgl2: Vec<f32>,
    pub wfl1: Vec<f32>,
    pub wgl1: Vec<f32>,
    pub wt3nw: Vec<f32>,
    pub wt3nb: Vec<f32>,
    pub bias: f32,
}

impl SimplLineEvaluator {
    pub fn new() -> Self {
        return SimplLineEvaluator {
            wfl3: vec![0.0; WFL3_WIDTH * WFL3_WIDTH],
            wgl3: vec![0.0; WGL3_WIDTH * WGL3_WIDTH],
            wfl2: vec![0.0; 64 * 64],
            wgl2: vec![0.0; 64 * 64],
            wfl1: vec![0.0; WFL1_WIDTH * WFL1_WIDTH],
            wgl1: vec![0.0; WGL1_WIDTH * WGL1_WIDTH],
            wt3nw: vec![0.0; 12],
            wt3nb: vec![0.0; 12],
            bias: 0.0,
        };
    }

    pub fn get_counts(
        &self,
        b: &Board,
    ) -> (
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    ) {
        let (att, def) = b.get_att_def();
        let (a1, a2, a3, _, _, _) = LineEvaluator::analyze_board(att, def);
        let (d1, d2, d3, _, _, _) = LineEvaluator::analyze_board(def, att);
        let stone = att | def;
        let ground = !stone & (stone << 16 | 0xffff);
        let float = !stone ^ ground;
        let a1_float = acum_mask_bundle(apply_mask_bundle(a1, float)) as usize;
        let a1_ground = acum_mask_bundle(apply_mask_bundle(a1, ground)) as usize;
        let a2_float = acum_mask_bundle(apply_mask_bundle(a2, float)) as usize;
        let a2_ground = acum_mask_bundle(apply_mask_bundle(a2, ground)) as usize;
        // let a3_float = acum_mask_bundle(apply_mask_bundle(a3, float)) as usize;
        let a3_float = (acum_or(a3) & float).count_ones() as usize;
        // let a3_ground = acum_mask_bundle(apply_mask_bundle(a3, ground)) as usize;
        let a3_ground = (acum_or(a3) & ground).count_ones() as usize;
        let d1_float = acum_mask_bundle(apply_mask_bundle(d1, float)) as usize;
        let d1_ground = acum_mask_bundle(apply_mask_bundle(d1, ground)) as usize;
        let d2_float = acum_mask_bundle(apply_mask_bundle(d2, float)) as usize;
        let d2_ground = acum_mask_bundle(apply_mask_bundle(d2, ground)) as usize;
        // let d3_float = acum_mask_bundle(apply_mask_bundle(d3, float)) as usize;
        let d3_float = (acum_or(d3) & float).count_ones() as usize;
        // let d3_ground = acum_mask_bundle(apply_mask_bundle(d3, ground)) as usize;
        let d3_ground = (acum_or(d3) & ground).count_ones() as usize;

        let l3_mask = acum_or(d3) | acum_or(a3);
        let trap_3_num = (l3_mask & (!l3_mask << 16) & 0x0000_ffff_0000_0000).count_ones() as usize;

        return (
            a1_float, a2_float, a3_float, a1_ground, a2_ground, a3_ground, d1_float, d2_float,
            d3_float, d1_ground, d2_ground, d3_ground, trap_3_num,
        );
    }

    pub fn evaluate_board(&self, b: &Board) -> f32 {
        let (af1, af2, af3, ag1, ag2, ag3, df1, df2, df3, dg1, dg2, dg3, tn3) = self.get_counts(&b);
        let (att, def) = b.get_att_def();
        let is_black = (att.count_ones() + def.count_ones()) % 2 == 0;
        let mut val = 0.0;

        if is_black {
            val += self.wt3nb[tn3];
        } else {
            val += self.wt3nw[tn3];
        }
        val += self.wfl1[af1 * WFL1_WIDTH + df1]
            + self.wfl2[af2 * WL2_WIDTH + df2]
            + self.wfl3[af3 * WFL3_WIDTH + df3]
            + self.wgl1[ag1 * WGL1_WIDTH + dg1]
            + self.wgl2[ag2 * WL2_WIDTH + dg2]
            + self.wgl3[ag3 * WGL3_WIDTH + dg3]
            // + self.wdfl1[96 - df1 + af1]
            // + self.wdfl2[64 - df2 + af2]
            // + self.wdfl3[32 - df3 + af3]
            // + self.wdgl1[96 + ag1 - dg1]
            // + self.wdgl2[64 + ag2 - dg2]
            // + self.wdgl3[32 + ag3 - dg3]
            + self.bias;
        return 1.0 / (1.0 + (-val).exp());
    }

    pub fn save(&self, name: String) -> Result<()> {
        use anyhow::Context;
        use std::fs::File;
        use std::io::{BufWriter, Write};

        let data_str = serde_json::to_string(self)?;

        let file = File::create(name)?;
        let mut buff_writer: BufWriter<File> = BufWriter::new(file);

        buff_writer
            .write(data_str.as_bytes())
            .context("write error")?;
        buff_writer.flush().context("flush error")?;

        Ok(())
    }
    pub fn load(&mut self, name: String) -> Result<()> {
        use anyhow::Context;
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(name)?;
        let buff_reader: BufReader<File> = BufReader::new(file);

        let mut lines = Vec::new();

        for line in buff_reader.lines() {
            // if process here, can save memory
            lines.push(line.context("read error")?);
        }
        let data_str = lines.join("\n");
        let mut src: SimplLineEvaluator = serde_json::from_str(&data_str)?;

        std::mem::swap(self, &mut src);

        Ok(())
    }
}

impl EvaluatorF for SimplLineEvaluator {
    fn eval_func_f32(&self, b: &Board) -> f32 {
        return self.evaluate_board(b);
    }
}

#[derive(Clone)]
pub struct TrainableSLE {
    main: SimplLineEvaluator,
    v: SimplLineEvaluator,
    m: SimplLineEvaluator,
    lr: f32,
}

impl TrainableSLE {
    pub fn new(lr: f32) -> Self {
        TrainableSLE {
            main: SimplLineEvaluator::new(),
            v: SimplLineEvaluator::new(),
            m: SimplLineEvaluator::new(),
            lr: lr,
        }
    }

    pub fn from(e: SimplLineEvaluator, lr: f32) -> Self {
        TrainableSLE {
            main: e,
            v: SimplLineEvaluator::new(),
            m: SimplLineEvaluator::new(),
            lr: lr,
        }
    }
}

impl Trainable for TrainableSLE {
    fn update(&mut self, b: &Board, delta: f32) {
        let (a1, a2, a3, a1_, a2_, a3_, d1, d2, d3, d1_, d2_, d3_, trap_3_num) =
            self.main.get_counts(b);
        let is_black = (b.black.count_ones() + b.white.count_ones()) % 2 == 0;
        // とりあえずsgd
        let val = self.main.evaluate_board(b);
        let dv = val * (1.0 - val);
        let delta = self.lr * delta * dv;
        self.main.wfl1[a1 * WFL1_WIDTH + d1] += delta;
        self.main.wfl2[a2 * WL2_WIDTH + d2] += delta;
        self.main.wfl3[a3 * WFL3_WIDTH + d3] += delta;
        self.main.wgl1[a1_ * WGL1_WIDTH + d1_] += delta;
        self.main.wgl2[a2_ * WL2_WIDTH + d2_] += delta;
        self.main.wgl3[a3_ * WGL3_WIDTH + d3_] += delta;
        // self.main.wdfl1[96 + a1 - d1] += delta;
        // self.main.wdfl2[64 + a2 - d2] += delta;
        // self.main.wdfl3[32 + a3 - d3] += delta;
        // self.main.wdgl1[96 + a1_ - d1_] += delta;
        // self.main.wdgl2[64 + a2_ - d2_] += delta;
        // self.main.wdgl3[32 + a3_ - d3_] += delta;
        if is_black {
            self.main.wt3nb[trap_3_num] += delta;
        } else {
            self.main.wt3nw[trap_3_num] += delta;
        }
        self.main.bias += delta;
    }

    fn get_val(&self, b: &Board) -> f32 {
        self.main.evaluate_board(b)
    }

    fn save(&self, file: String) -> Result<()> {
        self.main.save(file)
    }
    fn load(&mut self, file: String) -> Result<()> {
        self.main.load(file)
    }
    fn eval(&mut self) {}
    fn train(&mut self) {}
}

impl EvaluatorF for TrainableSLE {
    fn eval_func_f32(&self, b: &Board) -> f32 {
        return self.main.evaluate_board(b).clamp(0.0, 1.0);
    }
}
