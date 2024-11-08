use std::iter::FlatMap;
use std::process::exit;
use std::ptr::null;

use crate::activations;
use crate::activations::mwrap;
use crate::activations::Activation;

use crate::constants::FloatPrecision;

use crate::layers::Layer;

use crate::math::max;
use crate::math::mulm;
use crate::math::naive_mulm;
use crate::math::ssubm;
use crate::math::subm;
use crate::math::mtmulm;
use crate::math::DMatrix;

// Someday...
/*pub struct NeuralNetwork {
    layers: Vec<Layer>,
    pub error: DMatrix,
    delta: DMatrix,
    fdnet: DMatrix,
}

impl NeuralNetwork {
    pub fn new(output_size: usize) -> Self {
        Self {
            layers: Vec::new(),
            error: DMatrix::new(vec![0.; output_size], (output_size, 1)),
            delta: DMatrix::new(vec![0.; output_size], (output_size, 1)),
            fdnet: DMatrix::new(vec![0.; output_size], (output_size, 1)),
        }
    }

    pub fn predict(&mut self, input: &DMatrix) -> &DMatrix {
        self.layers[0].forward(input);
        let i = 0;
        for layer in self.layers.iter() {
            layer.forward(&self.layers[i].out);
            i += 1;
        }
        &self.layers[self.layers.len() - 1].out
    }

    pub fn train(&mut self, input: &DMatrix, label: &DMatrix) {
        self.layer0.forward(input);
        self.layer1.forward(&self.layer0.out);

        subm(label, &self.layer1.out, &mut self.layer1.delta); // dE
        subm(label, &self.layer1.out, &mut self.error); // dE

        self.layer1.backward(&self.layer0.out);

        tmulm(&self.layer1.weights, &self.layer1.delta, &mut self.layer0.delta);
        self.layer0.backward(&input);
    }
}*/

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

        ssubm(1. / label.data.len() as FloatPrecision, label, &self.layer1.out, &mut self.layer1.delta); // dE

        //println!("label: {}", label);
        //println!("out:{}", &self.layer1.out);
        //println!("delta:{}", &self.layer1.delta);
        
        ssubm(1. / label.data.len() as FloatPrecision, label, &self.layer1.out, &mut self.error); // dE

        self.layer1.backward(&self.layer0.out);

        mtmulm(&self.layer1.weights, &self.layer1.delta, &mut self.layer0.delta);
        self.layer0.backward(&input);
    }
    pub fn get_error(&self) -> FloatPrecision {
        self.layer1.delta.abs()
    }
}

pub struct NeuralNetwork3 {
    layer0: Layer,
    layer1: Layer,
    layer2: Layer,
    pub error: DMatrix,
    delta: DMatrix,
    fdnet: DMatrix,
}

impl NeuralNetwork3 {
    pub fn new(layer0: Layer, layer1: Layer, layer2:Layer, output_size: usize) -> Self {
        Self {
            layer0,
            layer1,
            layer2,
            error: DMatrix::new(vec![0.; output_size], (output_size, 1)),
            delta: DMatrix::new(vec![0.; output_size], (output_size, 1)),
            fdnet: DMatrix::new(vec![0.; output_size], (output_size, 1)),
        }
    }

    pub fn predict(&mut self, input: &DMatrix) -> &DMatrix {
        self.layer0.forward(input);
        self.layer1.forward(&self.layer0.out);
        self.layer2.forward(&self.layer1.out);
        &self.layer2.out
    }

    pub fn train(&mut self, input: &DMatrix, label: &DMatrix) {
        self.layer0.forward(input);
        self.layer1.forward(&self.layer0.out);
        self.layer2.forward(&self.layer1.out);

        ssubm(1. / label.data.len() as FloatPrecision, label, &self.layer2.out, &mut self.layer2.delta); // dE = y-t, delta = -dE
        ssubm(1. / label.data.len() as FloatPrecision, label, &self.layer2.out, &mut self.error); // dE

        self.layer2.backward(&self.layer1.out);

        mtmulm(&self.layer2.weights, &self.layer2.delta, &mut self.layer1.delta);
        self.layer1.backward(&self.layer0.out);

        mtmulm(&self.layer1.weights, &self.layer1.delta, &mut self.layer0.delta);
        self.layer0.backward(&input);

    }

    pub fn get_error(&self) -> FloatPrecision {
        self.layer2.delta.abs()
    }
}

pub struct NeuralNetwork5 {
    layer0: Layer,
    layer1: Layer,
    layer2: Layer,
    layer3: Layer,
    layer4: Layer,
    pub error: DMatrix,
    delta: DMatrix,
    fdnet: DMatrix,
}

impl NeuralNetwork5 {
    pub fn new(layer0: Layer, layer1: Layer, layer2:Layer, layer3: Layer, layer4: Layer, output_size: usize) -> Self {
        Self {
            layer0,
            layer1,
            layer2,
            layer3,
            layer4,
            error: DMatrix::new(vec![0.; output_size], (output_size, 1)),
            delta: DMatrix::new(vec![0.; output_size], (output_size, 1)),
            fdnet: DMatrix::new(vec![0.; output_size], (output_size, 1)),
        }
    }

    pub fn predict(&mut self, input: &DMatrix) -> &DMatrix {
        self.layer0.forward(input);
        self.layer1.forward(&self.layer0.out);
        self.layer2.forward(&self.layer1.out);
        self.layer3.forward(&self.layer2.out);
        self.layer4.forward(&self.layer3.out);
        &self.layer4.out
    }

    pub fn train(&mut self, input: &DMatrix, label: &DMatrix) {
        self.layer0.forward(input);
        self.layer1.forward(&self.layer0.out);
        self.layer2.forward(&self.layer1.out);
        self.layer3.forward(&self.layer2.out);
        self.layer4.forward(&self.layer3.out);

        ssubm(1. / label.data.len() as FloatPrecision, label, &self.layer4.out, &mut self.layer4.delta); // dE = y-t, delta = -dE
        ssubm(1. / label.data.len() as FloatPrecision, label, &self.layer4.out, &mut self.error); // dE

        self.layer4.backward(&self.layer3.out);

        mtmulm(&self.layer4.weights, &self.layer4.delta, &mut self.layer3.delta);
        self.layer3.backward(&self.layer2.out);

        mtmulm(&self.layer3.weights, &self.layer3.delta, &mut self.layer2.delta);
        self.layer2.backward(&self.layer1.out);

        mtmulm(&self.layer2.weights, &self.layer2.delta, &mut self.layer1.delta);
        self.layer1.backward(&self.layer0.out);

        mtmulm(&self.layer1.weights, &self.layer1.delta, &mut self.layer0.delta);
        self.layer0.backward(&input);
    }

    pub fn get_error(&self) -> FloatPrecision {
        self.layer4.delta.abs()
    }
}
