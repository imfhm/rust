// macro_rules! mat {
//     ($arr:expr, $n:expr, $m:expr) => {
//         SMatrix::new($arr, ($n, $m))
//     };
// }

macro_rules! mat {
    ($arr:expr, $n:expr, $m:expr) => {
        DMatrix::new(Vec::from($arr), ($n, $m))
    }
}
