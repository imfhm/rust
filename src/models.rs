use std::iter::FlatMap;

use crate::activations::mwrap;
use crate::activations::Activation;
use crate::constants::FloatPrecision;
use crate::layers::FlattenLayer;
use crate::layers::Layer;
use crate::layers::DenseLayer;
use crate::math::add;
use crate::math::dot;
use crate::math::Matrix;
use crate::math::scale;
use crate::math::sub;

pub struct NeuralNetwork {
    input_layer:FlattenLayer,
    layers:Vec<DenseLayer>,
    input_size: usize,
    output_size: usize,
    layer_count: usize,
}

impl NeuralNetwork {
    pub fn new() -> Self {
        Self {
            input_layer: FlattenLayer::new(),
            layers: Vec::new(),
            input_size: 0,
            output_size: 0,
            layer_count: 0
        }
    }
    /// Forward pass to predict an output given inputs
    pub fn predict(&self, input: Matrix) -> Matrix {
        let mut out = input;
        for i in 0..self.layer_count {
            out = self.layers[i].predict(out);
        }
        out
    }

    // Train the network on a single sample
    pub fn train(&mut self, input: Matrix, target: Matrix) {
        let mut nets_and_outs = Vec::new();
        let mut out = input.clone();
        for i in 0..self.layer_count {
            let sum = dot(self.layers[i].weights.clone(), out);
            let net = add(sum, self.layers[i].bias.clone());
            out = mwrap(self.layers[i].activation.f, net.clone());
            nets_and_outs.push((net, out.clone()));
        }

        let mut delta = sub(target, out);
        //let mut delta = scale(2. / (out.shape.0 as FloatPrecision), abserror); // initially MSE
        
        for i in (1..self.layer_count).rev() {
            let (net, out) = &nets_and_outs[i];
            let (dw, db) = self.layers[i].backprop((*net).clone(), &delta);
            delta = db.clone();
            self.layers[i].weights = add(self.layers[i].weights.clone(), dw); 
            self.layers[i].bias = add(self.layers[i].weights.clone(), db); 
        }

        let (dw, db) = self.layers[0].backprop(input, &delta);
        self.layers[0].weights = add(self.layers[0].weights.clone(), dw);
        self.layers[0].bias = add(self.layers[0].bias.clone(), db);

    }

    pub fn add_dense_layer(&mut self, input_size: usize, output_size: usize, activation: Activation, rate: FloatPrecision) {
        self.layers.push(DenseLayer::new(input_size, output_size, activation, rate));
        self.layer_count += 1;
    }
}

