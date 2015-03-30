//! Traits giving structural informations on linear algebra objects or the space they live in.

use std::{f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, isize, usize};
use std::slice::{Iter, IterMut};
use std::ops::{Add, Sub, Mul, Div, Rem, Index, IndexMut};
use num::{Float, Zero, One};
use traits::operations::{RMul, LMul, Axpy, Transpose, Inv, Absolute};
use traits::geometry::{Dot, Norm, Orig};

/// Basic integral numeric trait.
pub trait BaseNum: Copy + Zero + One +
                   Add<Self, Output = Self> + Sub<Self, Output = Self> +
                   Mul<Self, Output = Self> + Div<Self, Output = Self> +
                   Rem<Self, Output = Self> + PartialEq +
                   Absolute<AbsoluteValueType = Self> +
                   Axpy<Self> {
}

/// Basic floating-point number numeric trait.
pub trait BaseFloat: Float + Cast<f64> + BaseNum {
    /// Archimedes' constant.
    fn pi() -> Self;
    /// 2.0 * pi.
    fn two_pi() -> Self;
    /// pi / 2.0.
    fn frac_pi_2() -> Self;
    /// pi / 3.0.
    fn frac_pi_3() -> Self;
    /// pi / 4.0.
    fn frac_pi_4() -> Self;
    /// pi / 6.0.
    fn frac_pi_6() -> Self;
    /// pi / 8.0.
    fn frac_pi_8() -> Self;
    /// 1.0 / pi.
    fn frac_1_pi() -> Self;
    /// 2.0 / pi.
    fn frac_2_pi() -> Self;
    /// 2.0 / sqrt(pi).
    fn frac_2_sqrt_pi() -> Self;

    /// Euler's number.
    fn e() -> Self;
    /// log2(e).
    fn log2_e() -> Self;
    /// log10(e).
    fn log10_e() -> Self;
    /// ln(2.0).
    fn ln_2() -> Self;
    /// ln(10.0).
    fn ln_10() -> Self;
}

/// Traits of objects which can be created from an object of type `T`.
pub trait Cast<T> {
    /// Converts an element of type `T` to an element of type `Self`.
    fn from(t: T) -> Self;
}

/// Trait of matrices.
///
/// A matrix has rows and columns and are able to multiply them.
pub trait Mat: Row + Col + RMul<<Self as Row>::Row> + LMul<<Self as Col>::Column> + Index<(usize, usize)> {
    type ScalarType = Self::Output;
}

impl<M> Mat for M
where M: Row + Col + RMul<<M as Row>::Row> + LMul<<M as Col>::Column> + Index<(usize, usize)>,
      M::Output : Sized {
    type ScalarType = M::Output;
}

/// Trait implemented by square matrices.
pub trait SquareMat: Diag + Mat<Row = <Self as Diag>::DiagonalType, Column = <Self as Diag>::DiagonalType> +
                     Mul<Self, Output = Self> + Eye + Transpose + Diag + Dim + One {
}

impl<M> SquareMat for M
    where M: Diag + Mat<Row = <M as Diag>::DiagonalType, Column = <M as Diag>::DiagonalType> +
             Mul<M, Output = M> + Eye + Transpose + Diag + Inv + Dim + One {
}

/// Trait for constructing the identity matrix
pub trait Eye {
    /// Return the identity matrix of specified dimension
    fn new_identity(dim: usize) -> Self;
}

/// Types that have maximum and minimum value.
pub trait Bounded {
    /// The minimum value.
    #[inline]
    fn min_value() -> Self;
    /// The maximum value.
    #[inline]
    fn max_value() -> Self;
}

// FIXME: return an iterator instead
/// Traits of objects which can form a basis (typically vectors).
pub trait Basis {
    /// Iterates through the canonical basis of the space in which this object lives.
    fn canonical_basis<F: FnMut(Self) -> bool>(F);

    /// Iterates through a basis of the subspace orthogonal to `self`.
    fn orthonormal_subspace_basis<F: FnMut(Self) -> bool>(&Self, F);

    /// Gets the ith element of the canonical basis.
    fn canonical_basis_element(i: usize) -> Option<Self>;
}

/// Trait to access rows of a matrix or a vector.
pub trait Row {
    type Row;

    /// The number of column of `self`.
    fn nrows(&self) -> usize;
    /// Reads the `i`-th row of `self`.
    fn row(&self, i: usize) -> Self::Row;
    /// Writes the `i`-th row of `self`.
    fn set_row(&mut self, i: usize, row: Self::Row);

    // FIXME: add iterators on rows: this could be a very good way to generalize _and_ optimize
    // a lot of operations.
}

/// Trait to access columns of a matrix or vector.
pub trait Col {
    type Column;

    /// The number of column of this matrix or vector.
    fn ncols(&self) -> usize;

    /// Reads the `i`-th column of `self`.
    fn col(&self, i: usize) -> Self::Column;

    /// Writes the `i`-th column of `self`.
    fn set_col(&mut self, i: usize, Self::Column);

    // FIXME: add iterators on columns: this could be a very good way to generalize _and_ optimize
    // a lot of operations.
}

/// Trait to access part of a column of a matrix
pub trait ColSlice {
    type ColSlice;

    /// Returns a view to a slice of a column of a matrix.
    fn col_slice(&self, col_id: usize, row_start: usize, row_end: usize) -> Self::ColSlice;
}

/// Trait to access part of a row of a matrix
pub trait RowSlice {
    type RowSlice;

    /// Returns a view to a slice of a row of a matrix.
    fn row_slice(&self, row_id: usize, col_start: usize, col_end: usize) -> Self::RowSlice;
}

/// Trait of objects having a spacial dimension known at compile time.
pub trait Dim {
    /// The dimension of the object.
    fn dim(unused_mut: Option<Self>) -> usize;
}

/// Trait to get the diagonal of square matrices.
pub trait Diag {
    type DiagonalType;

    /// Creates a new matrix with the given diagonal.
    fn from_diag(diag: &Self::DiagonalType) -> Self;

    /// Sets the diagonal of this matrix.
    fn set_diag(&mut self, diag: &Self::DiagonalType);

    /// The diagonal of this matrix.
    fn diag(&self) -> Self::DiagonalType;
}

/// The shape of an indexable object.
pub trait Shape<I>: Index<I> {
    /// Returns the shape of an indexable object.
    fn shape(&self) -> I;
}

/// This is a workaround of current Rust limitations.
///
/// It exists because the `I` trait cannot be used to express write access.
/// Thus, this is the same as the `I` trait but without the syntactic sugar and with a method
/// to write to a specific index.
pub trait Indexable<I>: Shape<I> + IndexMut<I> {
    /// Swaps the `i`-th element of `self` with its `j`-th element.
    fn swap(&mut self, i: I, j: I);

    /// Reads the `i`-th element of `self`.
    ///
    /// `i` is not checked.
    unsafe fn unsafe_at(&self, i: I) -> Self::Output;
    /// Writes to the `i`-th element of `self`.
    ///
    /// `i` is not checked.
    unsafe fn unsafe_set(&mut self, i: I, Self::Output);
}

/// This is a workaround of current Rust limitations.
///
/// Traits of objects which can be iterated through like a vector.
pub trait Iterable {
    type Item;

    /// Gets a vector-like read-only iterator.
    fn iter<'l>(&'l self) -> Iter<'l, Self::Item>;
}

/// This is a workaround of current Rust limitations.
///
/// Traits of mutable objects which can be iterated through like a vector.
pub trait IterableMut {
    type Item;

    /// Gets a vector-like read-write iterator.
    fn iter_mut<'l>(&'l mut self) -> IterMut<'l, Self::Item>;
}

/*
 * Vec related traits.
 */
/// Trait grouping most common operations on vectors.
pub trait NumVec: Dim                                                 +
                  Sub<Self, Output = Self>                            +
                  Add<Self, Output = Self>                            +
                  Mul<<Self as NumVec>::ScalarType, Output = Self>    +
                  Div<<Self as NumVec>::ScalarType, Output = Self>    + 
                  Index<usize, Output = <Self as NumVec>::ScalarType> +
                  Zero + PartialEq + Dot + Axpy<<Self as NumVec>::ScalarType> {
    type ScalarType;
}

/// Trait of vector with components implementing the `BaseFloat` trait.
pub trait FloatVec: NumVec +
                    Norm<NormType = <Self as NumVec>::ScalarType> +
                    Basis {
}

/*
 * Pnt related traits.
 */
/// Trait that relates a point of an affine space to a vector of the associated vector space.
pub trait PntAsVec {
    type VectorType;

    /// Converts this point to its associated vector.
    fn to_vec(self) -> Self::VectorType;

    /// Converts a reference to this point to a reference to its associated vector.
    fn as_vec(&self) -> &Self::VectorType;

    // NOTE: this is used in some places to overcome some limitations untill the trait reform is
    // done on rustc.
    /// Sets the coordinates of this point to match those of a given vector.
    fn set_coords(&mut self, coords: Self::VectorType);
}

/// Trait grouping most common operations on points.
// XXX: the vector space element `V` should be an associated type. Though this would prevent V from
// having bounds (they are not supported yet). So, for now, we will just use a type parameter.
pub trait NumPnt:
          Copy +
          PntAsVec + // FIXME: this is weird. The inheritence should be the other way round!
          Dim +
          Orig +
          PartialEq +
          Axpy<<Self as NumPnt>::ScalarType> +
          Sub<Self, Output = <Self as PntAsVec>::VectorType> +
          Mul<<Self as NumPnt>::ScalarType, Output = Self> +
          Div<<Self as NumPnt>::ScalarType, Output = Self> +
          Add<<Self as PntAsVec>::VectorType, Output = Self> +
          Index<usize, Output = <Self as NumPnt>::ScalarType> { // FIXME: + Sub<V, Self>
    type ScalarType;
}

/// Trait of points with components implementing the `BaseFloat` trait.
pub trait FloatPnt: NumPnt + Sized
    where Self::ScalarType: BaseFloat,
          Self::VectorType: Norm<NormType = <Self as NumPnt>::ScalarType> {
    /// Computes the square distance between two points.
    #[inline]
    fn sqdist(&self, other: &Self) -> Self::ScalarType {
        (*self - *other).sqnorm()
    }

    /// Computes the distance between two points.
    #[inline]
    fn dist(&self, other: &Self) -> Self::ScalarType {
        (*self - *other).norm()
    }
}

/*
 *
 *
 * Some implementations for builtin types.
 *
 *
 */
// Bounded
macro_rules! impl_bounded(
    ($n: ty, $min: expr, $max: expr) => {
        impl Bounded for $n {
            #[inline]
            fn min_value() -> $n {
                $min
            }

            #[inline]
            fn max_value() -> $n {
                $max
            }
        }
    }
);

impl_bounded!(f32, f32::MIN, f32::MAX);
impl_bounded!(f64, f64::MIN, f64::MAX);
impl_bounded!(i8, i8::MIN, i8::MAX);
impl_bounded!(i16, i16::MIN, i16::MAX);
impl_bounded!(i32, i32::MIN, i32::MAX);
impl_bounded!(i64, i64::MIN, i64::MAX);
impl_bounded!(isize, isize::MIN, isize::MAX);
impl_bounded!(u8, u8::MIN, u8::MAX);
impl_bounded!(u16, u16::MIN, u16::MAX);
impl_bounded!(u32, u32::MIN, u32::MAX);
impl_bounded!(u64, u64::MIN, u64::MAX);
impl_bounded!(usize, usize::MIN, usize::MAX);


// BaseFloat
macro_rules! impl_base_float(
    ($n: ident) => {
        impl BaseFloat for $n {
            /// Archimedes' constant.
            fn pi() -> $n {
                $n::consts::PI
            }

            /// 2.0 * pi.
            fn two_pi() -> $n {
                2.0 * $n::consts::PI
            }

            /// pi / 2.0.
            fn frac_pi_2() -> $n {
                $n::consts::FRAC_PI_2
            }

            /// pi / 3.0.
            fn frac_pi_3() -> $n {
                $n::consts::FRAC_PI_3
            }

            /// pi / 4.0.
            fn frac_pi_4() -> $n {
                $n::consts::FRAC_PI_4
            }

            /// pi / 6.0.
            fn frac_pi_6() -> $n {
                $n::consts::FRAC_PI_6
            }

            /// pi / 8.0.
            fn frac_pi_8() -> $n {
                $n::consts::FRAC_PI_8
            }

            /// 1.0 / pi.
            fn frac_1_pi() -> $n {
                $n::consts::FRAC_1_PI
            }

            /// 2.0 / pi.
            fn frac_2_pi() -> $n {
                $n::consts::FRAC_2_PI
            }

            /// 2.0 / sqrt(pi).
            fn frac_2_sqrt_pi() -> $n {
                $n::consts::FRAC_2_SQRT_PI
            }


            /// Euler's number.
            fn e() -> $n {
                $n::consts::E
            }

            /// log2(e).
            fn log2_e() -> $n {
                $n::consts::LOG2_E
            }

            /// log10(e).
            fn log10_e() -> $n {
                $n::consts::LOG10_E
            }

            /// ln(2.0).
            fn ln_2() -> $n {
                $n::consts::LN_2
            }

            /// ln(10.0).
            fn ln_10() -> $n {
                $n::consts::LN_10
            }
        }
    }
);

impl BaseNum for i8 { }
impl BaseNum for i16 { }
impl BaseNum for i32 { }
impl BaseNum for i64 { }
impl BaseNum for isize { }
impl BaseNum for u8 { }
impl BaseNum for u16 { }
impl BaseNum for u32 { }
impl BaseNum for u64 { }
impl BaseNum for usize { }
impl BaseNum for f32 { }
impl BaseNum for f64 { }

impl_base_float!(f32);
impl_base_float!(f64);
