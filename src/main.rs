#![allow(warnings)]

mod layers;
mod activations;
mod constants;
mod math;
mod models;

#[macro_use]
mod macros;

use math::SMatrix;

use crate::math::Matrix;
use crate::math::dot;
use crate::math::add;

fn test<'a, const RC: usize> (a: SMatrix<'a, RC>) {
    println!("{:p}", a.data);
}
fn main() {
    let A = mat!([0., 1., 2.], 3, 1);
    let B = mat!([0.; 3], 3, 1);
    let C = mat!(A+B, 3, 1);
    
    //let B = Matrix {data: vec![vec![1.,2.,3.];3],shape:(3,3)};
    //let C = dot(&A,&B);
    //let D = dot(&A,&B);

    /*// Sample data: simple OR problem
    let training_data = [
        (Matrix::new(vec![0.0, 0.0], (2, 1)), Matrix::new(vec![0.0], (1, 1))),
        (Matrix::new(vec![1.0, 0.0], (2, 1)), Matrix::new(vec![1.0], (1 ,1))),
        (Matrix::new(vec![0.0, 1.0], (2, 1)), Matrix::new(vec![1.0], (1 ,1))),
        (Matrix::new(vec![1.0, 1.0], (2, 1)), Matrix::new(vec![1.0], (1 ,1))),
    ];
    let test_data = [
        (Matrix::new(vec![0.0, 0.0], (2, 1)), Matrix::new(vec![0.0], (1, 1))),
        (Matrix::new(vec![1.0, 0.0], (2, 1)), Matrix::new(vec![1.0], (1 ,1))),
        (Matrix::new(vec![0.0, 1.0], (2, 1)), Matrix::new(vec![1.0], (1 ,1))),
        (Matrix::new(vec![1.0, 1.0], (2, 1)), Matrix::new(vec![1.0], (1 ,1))),
    ];

    let mut nn = models::NeuralNetwork::new();
    nn.add_dense_layer(2, 1, activations::SIGMOID, 0.1);

    // Training loop
    for _ in 0..10000 {
        for (inputs, target) in &training_data {
            nn.train((*inputs).clone(), (*target).clone());
        }
    }

    // Testing the neural network
    for (inputs, target) in test_data {
        let prediction = nn.predict(inputs.clone());
        println!(
            "Input: {:?} -> Prediction: {:.3} (Target: {})",
            inputs.clone(), prediction, target
        );
    }*/

}