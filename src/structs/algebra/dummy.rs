#![macro_use]

macro_rules! vector_space_impl(
    ($t: ident, $point: expr, $dimension: expr, $($compN: ident),+) => { }
);

macro_rules! rotation_group_impl(
    ($t: ident, $point: ident, $vector: ident) => { }
);

macro_rules! euclidean_space_impl(
    ($t: ident, $vector: ident) => { }
);

macro_rules! matrix_group_approx_impl(
    ($t: ident, $($compN: ident),+) => { }
);

macro_rules! direct_isometry_group_impl(
    ($t: ident, $point: ident, $vector: ident, $rotation: ident) => { }
);

macro_rules! similarity_group_impl(
    ($t: ident, $point: ident, $vector: ident, $rotation: ident) => { }
);
