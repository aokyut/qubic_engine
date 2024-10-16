use super::xiver_vec;
use super::Node;
use super::Tensor;
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Linear {
    pub w: Tensor,
    height: usize,
    width: usize,
    pub b: Tensor,
    pub w_grad: Option<Tensor>,
    pub b_grad: Option<Tensor>,
    ignore_grad: bool,
}

impl Linear {
    pub fn new(w: Tensor, b: Tensor) -> Self {
        assert_eq!(w.shape.len(), 2);
        assert_eq!(b.shape.len(), 1);
        assert_eq!(b.shape[0], w.shape[0]);

        let height = w.shape[0];
        let width = w.shape[1];
        return Self {
            w: w,
            b: b,
            height: height,
            width: width,
            w_grad: None,
            b_grad: None,
            ignore_grad: false,
        };
    }

    pub fn auto(input_size: usize, output_size: usize) -> Self {
        let weight = xiver_vec(output_size, output_size * input_size);
        let weight = Tensor::new(weight, vec![output_size, input_size]);
        let b = Tensor::zeros(vec![output_size]);
        return Linear::new(weight, b);
    }

    pub fn set_ignore(&mut self) {
        self.ignore_grad = true;
    }
}

impl Node for Linear {
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        let input = inputs[0];
        let shape_size = input.shape.len();
        let n = input.shape[shape_size - 2];
        let m = *input.shape.last().unwrap();
        let step = n * m;
        let batch = input.data.len() / step;

        let mut w_grad = Tensor::zeros_like(&self.w);
        let mut b_grad = Tensor::zeros_like(&self.b);
        let mut input_grad = Tensor::zeros_like(&input);

        let grad_step = self.height * m;

        for h in 0..batch {
            let offset_input = h * step;
            let offset_grad = h * grad_step;
            for i in 0..self.height {
                for j in 0..self.width {
                    for k in 0..m {
                        w_grad.data[i * self.width + j] += grad.data[offset_grad + i * m + k]
                            * input.data[offset_input + j * m + k];
                        input_grad.data[offset_input + j * m + k] +=
                            self.w.data[i * self.width + j] * grad.data[offset_grad + i * m + k]
                    }
                }

                for k in 0..m {
                    b_grad.data[i] += grad.data[offset_grad + i * m + k];
                }
            }
        }
        if let Some(_w_grad) = self.w_grad.as_mut() {
            *_w_grad += w_grad;
        } else {
            self.w_grad = Some(w_grad);
        }
        if let Some(_b_grad) = self.b_grad.as_mut() {
            *_b_grad += b_grad;
        } else {
            self.b_grad = Some(b_grad);
        }

        return vec![input_grad];
    }

    fn call(&self, input: Vec<Tensor>) -> Tensor {
        // println!("input:{:?}", input);
        assert_eq!(input.len(), 1);
        let input = &input[0];
        let shape_size = input.shape.len();
        assert!(shape_size > 1);
        let n = input.shape[shape_size - 2];
        let m = *input.shape.last().unwrap();
        let step = n * m;
        let batch = input.data.len() / step;

        let mut ans_shape = input.shape.clone();
        ans_shape[shape_size - 2] = self.height;
        let mut ans = Tensor::zeros(ans_shape);
        let ans_step = self.height * m;

        for h in 0..batch {
            let offset_input = h * step;
            let offset_ans = h * ans_step;
            for i in 0..self.height {
                let bi = self.b.data[i];
                for k in 0..self.width {
                    let wik = self.w.data[i * self.width + k];
                    for j in 0..m {
                        ans.data[offset_ans + i * m + j] +=
                            input.data[offset_input + k * m + j] * wik;
                    }
                }
                for j in 0..m {
                    ans.data[offset_ans + i * m + j] += bi;
                }
            }
        }
        // println!("output:{:?}", ans);
        return ans;
    }
    fn no_grad(&self) -> bool {
        self.ignore_grad
    }

    fn has_params(&self) -> bool {
        !self.ignore_grad
    }

    fn pull_grad(&self) -> Option<Vec<&Tensor>> {
        return Some(vec![
            self.w_grad.as_ref().unwrap(),
            self.b_grad.as_ref().unwrap(),
        ]);
    }

    fn apply_update(&mut self, update: Vec<Tensor>) {
        for i in 0..self.w.data.len() {
            self.w.data[i] += update[0].data[i];
        }

        for i in 0..self.b.data.len() {
            self.b.data[i] += update[1].data[i];
        }

        self.w_grad = None;
        self.b_grad = None;
    }

    fn print(&self) {
        println!("w:{:?}, b:{:?}", self.w, self.b)
    }

    fn save_param(&self, _file: std::path::PathBuf) -> Result<()> {
        use anyhow::Context;
        use std::fs::File;
        use std::io::{BufWriter, Write};

        let data_str = serde_json::to_string(self).unwrap();

        let file = File::create(_file).unwrap();
        let mut buff_writer: BufWriter<File> = BufWriter::new(file);

        buff_writer
            .write(data_str.as_bytes())
            .context("write error")?;
        buff_writer.flush().context("flush error")?;
        Ok(())
    }

    fn load_param(&mut self, _file: std::path::PathBuf) -> Result<()> {
        use anyhow::Context;
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(_file).unwrap();
        let buff_reader: BufReader<File> = BufReader::new(file);

        let mut lines = Vec::new();

        for line in buff_reader.lines() {
            // if process here, can save memory
            lines.push(line.context("read error")?);
        }
        let data_str = lines.join("\n");
        let mut src: Linear = serde_json::from_str(&data_str).unwrap();

        std::mem::swap(&mut self.w, &mut src.w);
        std::mem::swap(&mut self.b, &mut src.b);

        Ok(())
    }
}
