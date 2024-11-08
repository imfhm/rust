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
mod plot;

#[macro_use]
mod macros;

use constants::FloatPrecision;
use plotters::data::float::FloatPrettyPrinter;
use rand::Rng;

use crate::load::read_floats;
use load::loading;
use load::Mnist;
use math::max;
use math::min;
use plot::plot2;

use math::argmax;
use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::layers::Layer;

use crate::math::DMatrix;

use crate::plot::plot;

fn one_hot(i: usize) -> Vec<FloatPrecision> {
    let mut data = vec![0.; 10];
    data[i] = 1.;
    data
}

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let path = "C:/users/antga/documents/uni/neuralnets/Hhwayli.dat";
    let heights: Vec<FloatPrecision> = match read_floats(path) {
        Err(_) => panic!("Could not read file at {}.", path),
        Ok(value) => value,
    };
    let steps: Vec<FloatPrecision> = (0..heights.len())
        .map(|x| 0.05 * (x as FloatPrecision))
        .collect();
    let n = heights.len() as FloatPrecision;
    let mean = n * 0.05 / 2.;
    let var = heights.iter().map(|x| (x - mean) * (x - mean)).sum::<FloatPrecision>() / (n - 1.);

    let mut inputs: Vec<DMatrix> = steps
        .iter()
        .map(|&x| DMatrix::new(vec![(x - mean) / var], (1, 1)))
        .collect();
    let labels: Vec<DMatrix> = heights
        .iter()
        .map(|&x| DMatrix::new(vec![x], (1, 1)))
        .collect();

    let imin = 100;
    let imax = 200;
    let wsteps = &steps[imin..imax];
    let wheights = &heights[imin..imax];
    let xmin = steps[imin];
    let xmax = steps[imax - 1];
    let min = min(&wheights.to_vec());
    let max = max(&wheights.to_vec());

    plot(
        "Heights",
        "C:/users/antga/documents/uni/neuralnets/rust/plots/plot0.png",
        &steps,
        &heights,
        (1000, 400),
        (xmin, xmax),
        (min, max),
    );

    let mut nn = models::NeuralNetwork5::new(
        Layer::new(1, 128, activations::LEAKYRELU, 0.0001),
        Layer::new(128, 256, activations::LEAKYRELU, 0.0001),
        Layer::new(256, 512, activations::LEAKYRELU, 0.0001),
        Layer::new(512, 512, activations::LEAKYRELU, 0.0001),
        Layer::new(512, 1, activations::LINEAR, 0.0001),
        1,
    );
    let mut rng = thread_rng();
    inputs.shuffle(&mut rng);
    for i in 0..inputs.len() {
        nn.train(&inputs[i], &labels[i]);
        loading(i, inputs.len(), 10);
    }

    let xs: Vec<FloatPrecision> = (0..8192 * 2)
        .map(|x| (0.025 * (x as FloatPrecision) - 8192. * 0.025) / (8192. * 0.05))
        .collect();
    let mut lys: Vec<FloatPrecision> = (0..8192 * 2 - 1)
        .map(|i| {
            if i % 2 == 0 {
                heights[i / 2]
            } else {
                (heights[(i - 1) / 2] + heights[(i + 1) / 2]) / 2.
            }
        })
        .collect();
    lys.push(heights[heights.len() - 1]);

    let mut pys: Vec<FloatPrecision> = Vec::new();
    for i in 0..8192 * 2 {
        let input = DMatrix::new(vec![xs[i]], (1, 1));
        let prediction = nn.predict(&input);
        pys.push(prediction.data[0]);
    }

    plot2(
        "Heights vs. Predictions",
        "C:/users/antga/documents/uni/neuralnets/rust/plots/plot1.png",
        &xs,
        &lys,
        &xs,
        &pys,
        (1000, 400),
        (-1., 1.),
        (min, max),
    );
}

fn main2() {
    env::set_var("RUST_BACKTRACE", "1");

    println!("Reading training data ....");
    let mnist = Mnist::new("C:/users/antga/documents/uni/neuralnets/MNIST/");
    let mut training_data = Vec::new();
    for i in 0..mnist.train_data.len() {
        let image_data: Vec<FloatPrecision> = mnist.train_data[i]
            .iter()
            .map(|&x| ((x as FloatPrecision) - 128.) / 255.)
            .collect();
        let label_data = one_hot(mnist.train_labels[i] as usize);

        let image = DMatrix::new(image_data, (784, 1));
        let label = DMatrix::new(label_data, (10, 1));
        training_data.push((image, label));
    }

    let mut rng = thread_rng();

    let mut nn = models::NeuralNetwork2::new(
        Layer::new(784, 32, activations::SIGMOID, 0.1),
        Layer::new(32, 10, activations::SIGMOID, 0.1),
        //Layer::new(10, 10, activations::SIGMOID, 0.1),
        10,
    );

    let mut err: Vec<FloatPrecision> = Vec::new();
    let n = training_data.len();
    let ticks = (0..n).map(|x| x as f64).collect::<Vec<FloatPrecision>>();
    println!("Starting to train ...");
    for i in 0..60000 / 32 {
        training_data.shuffle(&mut rng);
        let randn = rng.gen_range(0..60000 / 32 - 32);
        for (image, label) in &training_data[randn..randn + 32] {
            nn.train(image, label);
            err.push(nn.error.abs());
        }
        loading(i, 60000 / 32, 10);
    }
    
    let min = min(&err);
    let max = max(&err);
    plot(
        "Errors",
        &format!(
            "C:/Users/antga/documents/uni/neuralnets/rust/plots/err{}.png",
            0
        ),
    &ticks[0..5000].to_vec(),
    &err[0..5000].to_vec(),
        (1000, 400),
        (0., 5000 as f64),
        (min, max),
    );
    println!("\nReading test data ...");
    let mut test_data = Vec::new();
    for i in 0..mnist.test_data.len() {
        let random_number = rng.gen_range(0..=1875 - 32);

        let image_data: Vec<FloatPrecision> = mnist.test_data[i]
            .iter()
            .map(|&x| ((x as FloatPrecision) - 128.) / 255.)
            .collect();
        let label_data = one_hot(mnist.test_labels[i] as usize);

        let image = DMatrix::new(image_data, (784, 1));
        let label = DMatrix::new(label_data, (10, 1));
        test_data.push((image, label));
    }

    let mut rng = thread_rng();
    test_data.shuffle(&mut rng);

    println!("Starting to test ...");
    let mut sum = 0.;
    for i in 0..test_data.len() {
        let (image, label) = &test_data[i];
        let prediction = nn.predict(image);
        let target = label;

        let pred = argmax(&prediction.data);
        let targ = argmax(&target.data);
        if pred == targ {
            sum += 1.;
        }
        loading(i, test_data.len(), 10);
    }

    println!(
        "\nPercentage correct: {}",
        sum * 100. / (test_data.len() as FloatPrecision)
    );
}

/*fn main() {
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
*/
