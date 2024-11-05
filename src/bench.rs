#![feature(test)]

extern crate test;
mod constants;
mod math;

use crate::math::DMatrix;
use crate::math::SMatrix;
use test::Bencher;

// Results: For small matrices (<= 10_000 elements), statically sized is a bit faster.
// For larger matrices, we get a stack overflow for statically sized matrices.
#[bench]
fn bench_statically_sized(b: &mut Bencher) {
    let mut A = SMatrix::new([1.; 1000 * 1000], (1000, 1000));
    let mut B = SMatrix::new([1.; 1000 * 1000], (1000, 1000));
    b.iter(|| { &A + &B;});
}

#[bench]
fn bench_dyn_sized(b: &mut Bencher) {
    let A = DMatrix::new(vec![1.; 1000 * 1000], (1000, 1000));
    let B = DMatrix::new(vec![2.; 1000 * 1000], (1000, 1000));
    b.iter(|| { &A + &B });
}
