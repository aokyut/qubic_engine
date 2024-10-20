pub mod funcs;
pub mod ops;
pub mod optim;
pub mod params;

use anyhow::Result;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::fs;
use std::ops::{Add, AddAssign, Sub};
use std::path::{Path, PathBuf};

fn xiver_vec(n: usize, size: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let sigma = (1.0 / n as f32).sqrt();
    let normal = Normal::new(0.0, sigma).unwrap();
    let mut ans = Vec::new();

    for _ in 0..size {
        ans.push(normal.sample(&mut rng));
    }

    return ans;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn from_shape(shape: Vec<usize>) -> Self {
        let mut size = 1;
        for s in shape.iter() {
            size *= s;
        }
        let mut data = Tensor::create_random_array(size);
        return Tensor {
            data: data,
            shape: shape,
        };
    }

    pub fn zeros_like(tensor: &Tensor) -> Self {
        let mut size = tensor.data.len();
        let mut data = vec![0.0; size];

        return Tensor {
            data: data,
            shape: tensor.shape.clone(),
        };
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let mut size = 1;
        for s in shape.iter() {
            size *= s;
        }
        let mut data = vec![0.0; size];

        return Tensor {
            data: data,
            shape: shape,
        };
    }

    pub fn ones_like(tensor: &Tensor) -> Self {
        let mut size = tensor.data.len();
        let mut data = vec![1.0; size];

        return Tensor {
            data: data,
            shape: tensor.shape.clone(),
        };
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let mut size = 1;
        for s in shape.iter() {
            size *= s;
        }
        let mut data = vec![1.0; size];

        return Tensor {
            data: data,
            shape: shape,
        };
    }

    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let mut size: usize = 1;
        for s in shape.iter() {
            size *= s;
        }
        assert_eq!(size, data.len());
        return Tensor {
            data: data,
            shape: shape,
        };
    }

    fn create_random_array(size: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let mut data = Vec::new();
        for i in 0..size {
            data.push(rng.gen());
        }

        return data;
    }

    fn null() -> Self {
        return Tensor {
            data: Vec::new(),
            shape: Vec::new(),
        };
    }

    pub fn get_item(&self) -> Option<f32> {
        if self.data.len() == 1 {
            return Some(self.data[0]);
        } else {
            return None;
        }
    }
}

pub fn create_batch(tensors: Vec<Tensor>) -> Tensor {
    let size = tensors[0].data.len();
    let mut batch_data = Vec::new();
    let mut shape = tensors[0].shape.clone();
    shape.insert(0, tensors.len());

    for tensor in tensors {
        assert_eq!(tensor.data.len(), size);
        for &f in tensor.data.iter() {
            batch_data.push(f);
        }
    }

    return Tensor::new(batch_data, shape);
}

impl Add for Tensor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.data.len(), rhs.data.len());

        let mut out_data = self.data.clone();
        for i in 0..out_data.len() {
            out_data[i] += rhs.data[i];
        }

        return Tensor::new(out_data, self.shape.clone());
    }
}

impl AddAssign for Tensor {
    fn add_assign(&mut self, rhs: Self) {
        assert_eq!(self.data.len(), rhs.data.len());

        for i in 0..self.data.len() {
            self.data[i] += rhs.data[i];
        }
    }
}

pub trait Node {
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, output: &Tensor) -> Vec<Tensor>;
    fn call(&self, input: Vec<Tensor>) -> Tensor;
    fn no_grad(&self) -> bool {
        false
    }
    fn has_params(&self) -> bool {
        false
    }
    fn apply_update(&mut self, _update: Vec<Tensor>) {
        return;
    }
    fn load_param(&mut self, _file: PathBuf) -> Result<()> {
        Ok(())
    }
    fn save_param(&self, _file: PathBuf) -> Result<()> {
        Ok(())
    }
    fn pull_grad(&self) -> Option<Vec<&Tensor>> {
        return None;
    }
    fn print(&self) {}
}

pub trait Optimizer {
    fn optimize(&mut self, tar_id: usize, grads: Vec<&Tensor>) -> Vec<Tensor>;
}

pub struct Placeholder {}

impl Node for Placeholder {
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        return vec![];
    }
    fn call(&self, input: Vec<Tensor>) -> Tensor {
        return Tensor::null();
    }
    fn no_grad(&self) -> bool {
        true
    }
}

impl Placeholder {
    pub fn new() -> Self {
        return Placeholder {};
    }
}

pub struct Graph {
    pub layers: Vec<Box<dyn Node>>,
    pub optimizer: Option<Box<dyn Optimizer>>,
    pub flows: Vec<Option<Tensor>>,
    pub backflows: Vec<Option<Tensor>>,
    placeholder: Option<Vec<usize>>,
    parameters: Vec<usize>,
    inputs: Vec<Vec<usize>>,
    output: Vec<usize>,
    pub target: usize,
}

impl Graph {
    pub fn new() -> Self {
        return Graph {
            layers: Vec::new(),
            flows: Vec::new(),
            optimizer: None,
            backflows: Vec::new(),
            parameters: Vec::new(),
            placeholder: None,
            inputs: Vec::new(),
            output: Vec::new(),
            target: 0,
        };
    }

    pub fn backward(&mut self) {
        let mut stack: Vec<usize> = vec![self.target];
        self.backflows[self.target] = Some(Tensor::ones_like(
            &self.flows[self.target].as_ref().unwrap(),
        ));

        while stack.len() > 0 {
            let tar = stack.pop().unwrap();

            let input_ids = &self.inputs[tar];
            let mut input_vecs = Vec::new();
            if self.layers[tar].no_grad() {
                continue;
            }
            for input_id in input_ids.iter() {
                input_vecs.push(self.flows[*input_id].as_ref().unwrap());
                stack.push(*input_id);
            }
            let mut input_grads = self.layers[tar].backward(
                self.backflows[tar].as_ref().unwrap(),
                input_vecs,
                self.flows[tar].as_ref().unwrap(),
            );

            for i in 0..input_ids.len() {
                let input_id = input_ids[i];
                let swap = std::mem::replace(&mut input_grads[i], Tensor::null());
                self.backflows[input_id] = Some(swap);
            }
        }
    }

    pub fn inference(&self, mut input_vec: Vec<Tensor>) -> Tensor {
        let placeholder = self.placeholder.as_ref().unwrap();
        assert_eq!(placeholder.len(), input_vec.len());
        let mut flows = vec![None; self.layers.len()];

        // println!("placeholder:{placeholder:?}");
        for i in 0..placeholder.len() {
            let id = placeholder[i];

            let input = std::mem::replace(&mut input_vec[i], Tensor::null());
            flows[id] = Some(input);
        }

        let mut stack: Vec<usize> = vec![self.target];
        while stack.len() > 0 {
            let tar = stack.pop().unwrap();
            stack.push(tar);
            let input_ids = &self.inputs[tar];
            let mut full = true;
            for input_id in input_ids.iter() {
                if flows[*input_id].is_none() {
                    stack.push(*input_id);
                    full = false;
                }
            }
            if !full {
                continue;
            }
            stack.pop();
            let mut inputs = Vec::new();
            for input_id in input_ids.iter() {
                let input = flows[*input_id].clone();
                inputs.push(input.unwrap());
            }
            let out = self.layers[tar].call(inputs);
            flows[tar] = Some(out);
        }
        let output = flows[self.target].clone().unwrap();
        // println!("{flows:?}");
        return output.clone();
    }

    pub fn forward(&mut self, mut input_vec: Vec<Tensor>) -> Tensor {
        // println!("[g]input:{:?}", input_vec);
        return self.forward_(self.placeholder.as_ref().unwrap().clone(), input_vec);
    }

    pub fn forward_(&mut self, placeholder: Vec<usize>, mut input_vec: Vec<Tensor>) -> Tensor {
        assert_eq!(placeholder.len(), input_vec.len());
        for i in 0..placeholder.len() {
            let id = placeholder[i];

            let input = std::mem::replace(&mut input_vec[id], Tensor::null());
            self.flows[id] = Some(input);
        }

        let mut stack: Vec<usize> = vec![self.target];
        while stack.len() > 0 {
            let tar = stack.pop().unwrap();
            stack.push(tar);
            let input_ids = &self.inputs[tar];
            let mut full = true;
            for input_id in input_ids.iter() {
                if self.flows[*input_id].is_none() {
                    stack.push(*input_id);
                    full = false;
                }
            }
            if !full {
                continue;
            }
            stack.pop();
            let mut inputs = Vec::new();
            for input_id in input_ids.iter() {
                let input = self.flows[*input_id].clone();
                inputs.push(input.unwrap());
            }
            let out = self.layers[tar].call(inputs);
            self.flows[tar] = Some(out);
        }
        let output = self.flows[self.target].clone().unwrap();
        return output;
    }

    pub fn optimize(&mut self) {
        if let Some(optimizer) = self.optimizer.as_mut() {
            //TODO
            for &id in self.parameters.iter() {
                let grads = self.layers[id].pull_grad().unwrap();
                let update = optimizer.optimize(id, grads);
                self.layers[id].apply_update(update);
            }
            return;
        }
    }

    pub fn push_placeholder(&mut self) -> usize {
        let placeholder = Placeholder {};
        self.layers.push(Box::new(placeholder));
        self.flows.push(None);
        self.backflows.push(None);
        self.inputs.push(Vec::new());

        let id = self.inputs.len() - 1;
        self.output.push(id);

        return id;
    }

    pub fn add_layer(&mut self, inputs: Vec<usize>, node: Box<dyn Node>) -> usize {
        let has_param = node.has_params();
        self.layers.push(node);
        self.flows.push(None);
        self.backflows.push(None);

        let id = self.layers.len() - 1;
        self.output.push(id);
        if has_param {
            self.parameters.push(id);
        }

        for i in inputs.iter() {
            self.output[*i] = id;
        }

        self.inputs.push(inputs);

        return id;
    }

    pub fn reset(&mut self) {
        for i in 0..self.flows.len() {
            self.flows[i] = None;
            self.backflows[i] = None
        }
    }

    pub fn set_target(&mut self, id: usize) {
        self.target = id;
    }

    pub fn set_placeholder(&mut self, placeholder: Vec<usize>) {
        self.placeholder = Some(placeholder);
    }

    pub fn save(&self, file: String) {
        let mut path = PathBuf::new();
        path.push(file);

        fs::create_dir_all(path.clone());

        for i in 0..self.layers.len() {
            path.push(format!("{}.json", i));

            self.layers[i].save_param(path.clone());
            path = path.parent().unwrap().to_path_buf();
        }
    }

    pub fn load(&mut self, file: String) {
        let mut path = PathBuf::new();
        path.push(file);

        for i in 0..self.layers.len() {
            path.push(format!("{}.json", i));

            self.layers[i].load_param(path.clone());
            path = path.parent().unwrap().to_path_buf();
        }
    }
}
