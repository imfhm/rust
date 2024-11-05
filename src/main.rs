#![allow(warnings)]

use std::env;
use std::fs;
use std::io;
use std::io::Write;
use std::process::exit;

mod activations;
mod constants;
mod layers;
mod load;
mod math;
mod models;

#[macro_use]
mod macros;

use constants::FloatPrecision;
use load::Mnist;
use math::argmax;
use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::layers::Layer;

use crate::math::DMatrix;

fn one_hot(i: usize) -> Vec<FloatPrecision> {
    let mut data = vec![0.; 10];
    data[i] = 1.;
    data
}

/*fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    println!("Reading training data ....");
    let mnist = Mnist::new("C:/users/antga/documents/uni/neuralnets/MNIST/");
    let mut training_data = Vec::new();
    for i in 0..mnist.train_data.len() {
        let image_data: Vec<FloatPrecision> = mnist.train_data[i]
            .iter()
            .map(|&x| (x as FloatPrecision) / 255.)
            .collect();
        let label_data = one_hot(mnist.train_labels[i] as usize);

        let image = DMatrix::new(image_data, (784, 1));
        let label = DMatrix::new(label_data, (10, 1));
        training_data.push((image, label));
    }

    let mut rng = thread_rng();
    training_data.shuffle(&mut rng);

    let mut nn = models::NeuralNetwork2::new(
        Layer::new(784, 32, activations::SIGMOID, 0.1),
        Layer::new(32, 10, activations::SIGMOID, 0.1),
        10,
    );

    println!("Starting to train ...");
    let mut counter = 0;
    for (image, label) in &training_data {
        nn.train(image, label);
        counter += 1;
        if counter > 600 {
            print!("|");
            io::stdout().flush();
            counter = 0;
        }
    }

    println!("\nReading test data ...");
    let mut test_data = Vec::new();
    for i in 0..mnist.test_data.len() {
        let image_data: Vec<FloatPrecision> = mnist.test_data[i]
            .iter()
            .map(|&x| (x as FloatPrecision) / 255.)
            .collect();
        let label_data = one_hot(mnist.test_labels[i] as usize);

        let image = DMatrix::new(image_data, (784, 1));
        let label = DMatrix::new(label_data, (10, 1));
        test_data.push((image, label));
    }

    let mut rng = thread_rng();
    test_data.shuffle(&mut rng);

    println!("Starting to test ...");
    let mut counter = 0;
    let mut sum = 0.;
    for (image, label) in &test_data {
        let prediction = nn.predict(image);
        let target = label;

        let pred = argmax(&prediction.data);
        let targ = argmax(&target.data);
        if pred == targ {
            sum += 1.;
        }

        counter += 1;
        if counter > 100 {
            print!("|");
            io::stdout().flush();
            counter = 0;
        }
    }
    println!(
        "Percentage correct: {}",
        sum * 100. / (test_data.len() as FloatPrecision)
    );
}*/

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let training_data = [
        (mat!([0., 0.], 2, 1), mat!([0.0], 1, 1)),
        (mat!([1.0, 0.0], 2, 1), mat!([1.0], 1 ,1)),
        (mat!([0.0, 1.0], 2, 1), mat!([1.0], 1 ,1)),
        (mat!([1.0, 1.0], 2, 1), mat!([1.0], 1 ,1)),
    ];
    let test_data = [
        (mat!([0.0, 0.0], 2, 1), mat!([0.0], 1, 1)),
        (mat!([1.0, 0.0], 2, 1), mat!([1.0], 1 ,1)),
        (mat!([0.0, 1.0], 2, 1), mat!([1.0], 1 ,1)),
        (mat!([1.0, 1.0], 2, 1), mat!([1.0], 1 ,1)),
    ];

    let mut nn = models::NeuralNetwork2::new(
        Layer::new(2, 2, activations::SIGMOID, 0.1),
        Layer::new(2, 1, activations::SIGMOID, 0.1),
        1
    );

    // Training loop
    for _ in 0..10000 {
        for (inputs, target) in &training_data {
            nn.train(inputs, target);
        }
    }

    // Testing the neural network
    for (inputs, target) in &test_data {
        let prediction = nn.predict(inputs);
        println!(
            "Input: {:?} -> Prediction: {:.3} (Target: {})",
            inputs.clone(), prediction, target
        );
    }

}
