use std::iter::FlatMap;
use std::process::exit;
use std::ptr::null;

use crate::activations;
use crate::activations::mwrap;
use crate::activations::Activation;

use crate::constants::FloatPrecision;

use crate::layers::Layer;

use crate::math::mulm;
use crate::math::naive_mulm;
use crate::math::subm;
use crate::math::tmulm;
use crate::math::DMatrix;

pub struct NeuralNetwork2 {
    layer0: Layer,
    layer1: Layer,
    pub error: DMatrix,
    delta: DMatrix,
    fdnet: DMatrix,
}

impl NeuralNetwork2 {
    pub fn new(layer0: Layer, layer1: Layer, output_size: usize) -> Self {
        Self {
            layer0,
            layer1,
            error: DMatrix::new(vec![0.; output_size], (output_size, 1)),
            delta: DMatrix::new(vec![0.; output_size], (output_size, 1)),
            fdnet: DMatrix::new(vec![0.; output_size], (output_size, 1)),
        }
    }

    pub fn predict(&mut self, input: &DMatrix) -> &DMatrix {
        self.layer0.forward(input);
        self.layer1.forward(&self.layer0.out);
        &self.layer1.out
    }

    pub fn train(&mut self, input: &DMatrix, label: &DMatrix) {
        self.layer0.forward(input);
        self.layer1.forward(&self.layer0.out);

        subm(label, &self.layer1.out, &mut self.error); // dE

        self.layer1.backward(&self.layer0.out, &self.error);

        tmulm(&self.layer1.weights, &self.layer1.delta, &mut self.layer0.delta);
        self.layer0.backward(&input, &self.delta);
    }
}
