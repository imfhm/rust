macro_rules! mat {
    ($arr:expr, $n:expr, $m:expr) => {
        SMatrix::new(&$arr, ($n, $m))
    };
}