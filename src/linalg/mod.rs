//! [Reexported at the root of this crate.] Factorization of real matrices.

pub mod balancing;
mod bidiagonal;
mod cholesky;
mod convolution;
mod determinant;
mod full_piv_lu;
pub mod givens;
mod hessenberg;
pub mod householder;
mod inverse;
mod lu;
mod permutation_sequence;
mod qr;
mod schur;
mod solve;
mod svd;
mod symmetric_eigen;
mod symmetric_tridiagonal;

//// FIXME: Not complete enough for publishing.
//// This handles only cases where each eigenvalue has multiplicity one.
// mod eigen;

pub use self::{
    bidiagonal::*, cholesky::*, convolution::*, full_piv_lu::*, hessenberg::*, lu::*,
    permutation_sequence::*, qr::*, schur::*, svd::*, symmetric_eigen::*, symmetric_tridiagonal::*,
};
