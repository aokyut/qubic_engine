use crate::board::Board;
use super::{
    acum_mask_bundle, acum_or, apply_mask_bundle, EvaluatorF, LineEvaluator, LineMaskBundle,
    Trainable,
};
use anyhow::{Ok, Result};
use serde::{Deserialize, Serialize};

const FEATURE_SIZE: usize = 39; // 13 original + 26 overlap features
const HIDDEN1_SIZE: usize = 64;
const HIDDEN2_SIZE: usize = 32;
const EPS: f32 = 1e-7;

#[derive(Clone, Serialize, Deserialize)]
pub struct Linear {
    weight: Vec<f32>,
    bias: Vec<f32>,
    v_weight: Vec<f32>,
    v_bias: Vec<f32>,
    m_weight: Vec<f32>,
    m_bias: Vec<f32>,
    activation: bool,
    width: usize,
    height: usize,
}

impl Linear {
    pub fn new(width: usize, height: usize, activation: bool) -> Self {
        use minimum_ml::ml::xiver_vec;
        let weight = xiver_vec(width, height * width);
        let bias = xiver_vec(1, height);

        Linear {
            weight,
            width,
            height,
            bias,
            activation,
            v_weight: vec![0.0; width * height],
            v_bias: vec![0.0; height],
            m_weight: vec![0.0; width * height],
            m_bias: vec![0.0; height],
        }
    }

    pub fn affine(&self, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; self.height];
        for i in 0..self.height {
            let offset = i * self.width;
            for j in 0..self.width {
                out[i] += self.weight[offset + j] * x[j];
            }
            out[i] += self.bias[i];
        }
        out
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut x = self.affine(x);
        if self.activation {
            for i in 0..self.height {
                x[i] = x[i].max(0.0); // ReLU
            }
        }
        x
    }

    pub fn backward(
        &mut self,
        input: &[f32],
        out: &[f32],
        dout: &[f32],
        beta1: f32,
        beta2: f32,
    ) -> Vec<f32> {
        let mut daffine = vec![0.0; self.height];
        if self.activation {
            for i in 0..self.height {
                if out[i] > 0.0 {
                    daffine[i] = dout[i];
                }
            }
        } else {
            daffine = dout.to_vec();
        }

        // Update bias gradients
        for i in 0..self.height {
            self.v_bias[i] = beta1 * self.v_bias[i] + (1.0 - beta1) * daffine[i];
            self.m_bias[i] = beta2 * self.m_bias[i] + (1.0 - beta2) * daffine[i].powi(2);
        }

        // Update weight gradients
        let mut dweight = vec![0.0; self.weight.len()];
        for i in 0..self.height {
            let offset = i * self.width;
            for j in 0..self.width {
                dweight[offset + j] = daffine[i] * input[j];
            }
        }
        for i in 0..self.weight.len() {
            self.v_weight[i] = beta1 * self.v_weight[i] + (1.0 - beta1) * dweight[i];
            self.m_weight[i] = beta2 * self.m_weight[i] + (1.0 - beta2) * dweight[i].powi(2);
        }

        // Compute dinput
        let mut dinput = vec![0.0; self.width];
        for i in 0..self.height {
            if daffine[i] == 0.0 {
                continue;
            }
            let offset = i * self.width;
            for j in 0..self.width {
                dinput[j] += self.weight[offset + j] * daffine[i];
            }
        }
        dinput
    }

    pub fn update(&mut self, lr: f32) {
        for i in 0..self.height {
            self.bias[i] += lr * self.v_bias[i] / (self.m_bias[i] + EPS).sqrt();
        }
        for i in 0..self.weight.len() {
            self.weight[i] += lr * self.v_weight[i] / (self.m_weight[i] + EPS).sqrt();
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct NeuralLineEvaluator {
    l1: Linear,
    l2: Linear,
    l3: Linear,
}

impl NeuralLineEvaluator {
    pub fn new() -> Self {
        NeuralLineEvaluator {
            l1: Linear::new(FEATURE_SIZE, HIDDEN1_SIZE, true),
            l2: Linear::new(HIDDEN1_SIZE, HIDDEN2_SIZE, true),
            l3: Linear::new(HIDDEN2_SIZE, 1, false),
        }
    }

    /// Extract features with overlap information
    pub fn extract_features(b: &Board) -> Vec<f32> {
        let (att, def) = b.get_att_def();
        let (a1, a2, a3, _, _, _) = LineEvaluator::analyze_board(att, def);
        let (d1, d2, d3, _, _, _) = LineEvaluator::analyze_board(def, att);
        
        let stone = att | def;
        let ground = !stone & (stone << 16 | 0xffff);
        let float = !stone ^ ground;
        
        // Original features (13)
        let a1_float = acum_mask_bundle(apply_mask_bundle(a1, float)) as f32;
        let a1_ground = acum_mask_bundle(apply_mask_bundle(a1, ground)) as f32;
        let a2_float = acum_mask_bundle(apply_mask_bundle(a2, float)) as f32;
        let a2_ground = acum_mask_bundle(apply_mask_bundle(a2, ground)) as f32;
        let a3_float = (acum_or(a3) & float).count_ones() as f32;
        let a3_ground = (acum_or(a3) & ground).count_ones() as f32;
        
        let d1_float = acum_mask_bundle(apply_mask_bundle(d1, float)) as f32;
        let d1_ground = acum_mask_bundle(apply_mask_bundle(d1, ground)) as f32;
        let d2_float = acum_mask_bundle(apply_mask_bundle(d2, float)) as f32;
        let d2_ground = acum_mask_bundle(apply_mask_bundle(d2, ground)) as f32;
        let d3_float = (acum_or(d3) & float).count_ones() as f32;
        let d3_ground = (acum_or(d3) & ground).count_ones() as f32;
        
        let l3_mask = acum_or(d3) | acum_or(a3);
        let trap_3_num = (l3_mask & (!l3_mask << 16) & 0x0000_ffff_0000_0000).count_ones() as f32;
        
        // NEW: Overlap features (26 = 13 choose 2, but we'll use meaningful ones)
        // Overlap between attacker's different line types
        let a1_a2_overlap = Self::mask_bundle_overlap(a1, a2) as f32;
        let a1_a3_overlap = Self::mask_bundle_overlap_or(a1, a3) as f32;
        let a2_a3_overlap = Self::mask_bundle_overlap_or(a2, a3) as f32;
        
        // Overlap between defender's different line types
        let d1_d2_overlap = Self::mask_bundle_overlap(d1, d2) as f32;
        let d1_d3_overlap = Self::mask_bundle_overlap_or(d1, d3) as f32;
        let d2_d3_overlap = Self::mask_bundle_overlap_or(d2, d3) as f32;
        
        // Cross overlaps (attacker vs defender)
        let a1_d1_overlap = Self::mask_bundle_overlap(a1, d1) as f32;
        let a1_d2_overlap = Self::mask_bundle_overlap(a1, d2) as f32;
        let a1_d3_overlap = Self::mask_bundle_overlap_or(a1, d3) as f32;
        
        let a2_d1_overlap = Self::mask_bundle_overlap(a2, d1) as f32;
        let a2_d2_overlap = Self::mask_bundle_overlap(a2, d2) as f32;
        let a2_d3_overlap = Self::mask_bundle_overlap_or(a2, d3) as f32;
        
        let a3_d1_overlap = Self::mask_bundle_overlap_or(a3, d1) as f32;
        let a3_d2_overlap = Self::mask_bundle_overlap_or(a3, d2) as f32;
        let a3_d3_overlap = ((acum_or(a3) & acum_or(d3)).count_ones()) as f32;
        
        // Float/Ground interaction overlaps
        let a_float_ground_ratio = if a1_ground + a2_ground + a3_ground > 0.0 {
            (a1_float + a2_float + a3_float) / (a1_ground + a2_ground + a3_ground + 1.0)
        } else {
            0.0
        };
        let d_float_ground_ratio = if d1_ground + d2_ground + d3_ground > 0.0 {
            (d1_float + d2_float + d3_float) / (d1_ground + d2_ground + d3_ground + 1.0)
        } else {
            0.0
        };
        
        // Threat density features
        let a_total = a1_float + a1_ground + a2_float + a2_ground + a3_float + a3_ground;
        let d_total = d1_float + d1_ground + d2_float + d2_ground + d3_float + d3_ground;
        let threat_ratio = if d_total > 0.0 { a_total / (d_total + 1.0) } else { a_total };
        
        // Advanced features
        let a3_urgency = a3_float + a3_ground;
        let d3_urgency = d3_float + d3_ground;
        let multiple_threats = if a3_urgency >= 2.0 { 1.0 } else { 0.0 };
        let must_defend = if d3_urgency >= 1.0 { 1.0 } else { 0.0 };
        
        // Weighted sum features
        let a_weighted = a1_float + a1_ground + 2.0 * (a2_float + a2_ground) + 4.0 * (a3_float + a3_ground);
        let d_weighted = d1_float + d1_ground + 2.0 * (d2_float + d2_ground) + 4.0 * (d3_float + d3_ground);
        
        vec![
            // Original 13 features
            a1_float, a2_float, a3_float, a1_ground, a2_ground, a3_ground,
            d1_float, d2_float, d3_float, d1_ground, d2_ground, d3_ground,
            trap_3_num,
            // Overlap features (26 more)
            a1_a2_overlap, a1_a3_overlap, a2_a3_overlap,
            d1_d2_overlap, d1_d3_overlap, d2_d3_overlap,
            a1_d1_overlap, a1_d2_overlap, a1_d3_overlap,
            a2_d1_overlap, a2_d2_overlap, a2_d3_overlap,
            a3_d1_overlap, a3_d2_overlap, a3_d3_overlap,
            a_float_ground_ratio, d_float_ground_ratio,
            threat_ratio, a3_urgency, d3_urgency,
            multiple_threats, must_defend,
            a_weighted, d_weighted,
            // Board state features
            (att.count_ones() as f32), (def.count_ones() as f32),
        ]
    }
    
    /// Count overlapping bits between two mask bundles
    fn mask_bundle_overlap(mb1: LineMaskBundle, mb2: LineMaskBundle) -> usize {
        let (x1, y1, z1, xy1, xy1_, yz1, yz1_, xz1, xz1_, xyz11, xyz21, xyz31, xyz41) = mb1;
        let (x2, y2, z2, xy2, xy2_, yz2, yz2_, xz2, xz2_, xyz12, xyz22, xyz32, xyz42) = mb2;
        
        (x1 & x2).count_ones() as usize +
        (y1 & y2).count_ones() as usize +
        (z1 & z2).count_ones() as usize +
        (xy1 & xy2).count_ones() as usize +
        (xy1_ & xy2_).count_ones() as usize +
        (yz1 & yz2).count_ones() as usize +
        (yz1_ & yz2_).count_ones() as usize +
        (xz1 & xz2).count_ones() as usize +
        (xz1_ & xz2_).count_ones() as usize +
        (xyz11 & xyz12).count_ones() as usize +
        (xyz21 & xyz22).count_ones() as usize +
        (xyz31 & xyz32).count_ones() as usize +
        (xyz41 & xyz42).count_ones() as usize
    }
    
    /// Count overlapping bits between mask bundle and OR-ed mask
    fn mask_bundle_overlap_or(mb: LineMaskBundle, mask: LineMaskBundle) -> usize {
        let mask_or = acum_or(mask);
        let (x, y, z, xy, xy_, yz, yz_, xz, xz_, xyz1, xyz2, xyz3, xyz4) = mb;
        
        (x & mask_or).count_ones() as usize +
        (y & mask_or).count_ones() as usize +
        (z & mask_or).count_ones() as usize +
        (xy & mask_or).count_ones() as usize +
        (xy_ & mask_or).count_ones() as usize +
        (yz & mask_or).count_ones() as usize +
        (yz_ & mask_or).count_ones() as usize +
        (xz & mask_or).count_ones() as usize +
        (xz_ & mask_or).count_ones() as usize +
        (xyz1 & mask_or).count_ones() as usize +
        (xyz2 & mask_or).count_ones() as usize +
        (xyz3 & mask_or).count_ones() as usize +
        (xyz4 & mask_or).count_ones() as usize
    }

    pub fn evaluate_board(&self, b: &Board) -> f32 {
        let features = Self::extract_features(b);
        let h1 = self.l1.forward(&features);
        let h2 = self.l2.forward(&h1);
        let output = self.l3.forward(&h2);
        
        // Sigmoid activation
        1.0 / (1.0 + (-output[0]).exp())
    }

    pub fn save(&self, name: String) -> Result<()> {
        use std::fs::File;
        use std::io::{BufWriter, Write};

        let data_str = serde_json::to_string(self)?;
        let file = File::create(name)?;
        let mut buff_writer = BufWriter::new(file);
        buff_writer.write_all(data_str.as_bytes())?;
        buff_writer.flush()?;
        Ok(())
    }

    pub fn load(&mut self, name: String) -> Result<()> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(name)?;
        let buff_reader = BufReader::new(file);
        let mut lines = Vec::new();

        for line in buff_reader.lines() {
            lines.push(line?);
        }
        let data_str = lines.join("\n");
        let mut src: NeuralLineEvaluator = serde_json::from_str(&data_str)?;
        std::mem::swap(self, &mut src);
        Ok(())
    }
}

impl EvaluatorF for NeuralLineEvaluator {
    fn eval_func_f32(&self, b: &Board) -> f32 {
        self.evaluate_board(b)
    }
}

#[derive(Clone)]
pub struct TrainableNLE {
    main: NeuralLineEvaluator,
    lr: f32,
    beta1: f32,
    beta2: f32,
    // Store intermediate values for backprop
    last_features: Vec<f32>,
    last_h1: Vec<f32>,
    last_h1_pre: Vec<f32>,
    last_h2: Vec<f32>,
    last_h2_pre: Vec<f32>,
}

impl TrainableNLE {
    pub fn new(lr: f32) -> Self {
        TrainableNLE {
            main: NeuralLineEvaluator::new(),
            lr,
            beta1: 0.9,
            beta2: 0.999,
            last_features: vec![],
            last_h1: vec![],
            last_h1_pre: vec![],
            last_h2: vec![],
            last_h2_pre: vec![],
        }
    }

    pub fn from(e: NeuralLineEvaluator, lr: f32) -> Self {
        TrainableNLE {
            main: e,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            last_features: vec![],
            last_h1: vec![],
            last_h1_pre: vec![],
            last_h2: vec![],
            last_h2_pre: vec![],
        }
    }
}

impl Trainable for TrainableNLE {
    fn update(&mut self, b: &Board, delta: f32) {
        // Forward pass with caching
        let features = NeuralLineEvaluator::extract_features(b);
        self.last_features = features.clone();
        
        self.last_h1_pre = self.main.l1.affine(&features);
        self.last_h1 = self.last_h1_pre.iter().map(|&x| x.max(0.0)).collect();
        
        self.last_h2_pre = self.main.l2.affine(&self.last_h1);
        self.last_h2 = self.last_h2_pre.iter().map(|&x| x.max(0.0)).collect();
        
        let output_pre = self.main.l3.affine(&self.last_h2);
        let output = 1.0 / (1.0 + (-output_pre[0]).exp());
        
        // Backward pass
        // dL/doutput = delta * sigmoid'(output) = delta * output * (1 - output)
        let doutput = delta * output * (1.0 - output);
        let dout = vec![doutput];
        
        // Layer 3 backward
        let dh2 = self.main.l3.backward(&self.last_h2, &output_pre, &dout, self.beta1, self.beta2);
        
        // Layer 2 backward
        let dh1 = self.main.l2.backward(&self.last_h1, &self.last_h2_pre, &dh2, self.beta1, self.beta2);
        
        // Layer 1 backward
        let _ = self.main.l1.backward(&self.last_features, &self.last_h1_pre, &dh1, self.beta1, self.beta2);
        
        // Update weights
        self.main.l3.update(self.lr);
        self.main.l2.update(self.lr);
        self.main.l1.update(self.lr);
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

impl EvaluatorF for TrainableNLE {
    fn eval_func_f32(&self, b: &Board) -> f32 {
        self.main.evaluate_board(b).clamp(0.001, 0.999)
    }
}
