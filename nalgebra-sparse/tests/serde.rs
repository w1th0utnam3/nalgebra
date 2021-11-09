#![cfg(feature = "serde-serialize")]
//! Serialization tests
#[cfg(any(not(feature = "proptest-support"), not(feature = "compare")))]
compile_error!("Tests must be run with features `proptest-support` and `compare`");

#[macro_use]
pub mod common;

use nalgebra_sparse::coo::CooMatrix;
use nalgebra_sparse::csc::CscMatrix;
use nalgebra_sparse::csr::CsrMatrix;
use nalgebra_sparse::pattern::SparsityPattern;

use nalgebra::{Matrix5, Vector5};
use nalgebra_sparse::factorization::{CscCholesky, CscSymbolicCholesky};
use proptest::prelude::*;
use serde::{Deserialize, Serialize};

use crate::common::{csc_strategy, csr_strategy};

fn json_roundtrip<T: Serialize + for<'a> Deserialize<'a>>(value: &T) -> T {
    let serialized = serde_json::to_string(value).unwrap();
    println!("{}", serialized);
    let deserialized: T = serde_json::from_str(&serialized).unwrap();
    deserialized
}

/*
#[test]
fn pattern_roundtrip() {
    {
        // A pattern with zero explicitly stored entries
        let pattern =
            SparsityPattern::try_from_offsets_and_indices(3, 2, vec![0, 0, 0, 0], Vec::new())
                .unwrap();

        assert_eq!(json_roundtrip(&pattern), pattern);
    }

    {
        // Arbitrary pattern
        let offsets = vec![0, 2, 2, 5];
        let indices = vec![0, 5, 1, 2, 3];
        let pattern =
            SparsityPattern::try_from_offsets_and_indices(3, 6, offsets.clone(), indices.clone())
                .unwrap();

        assert_eq!(json_roundtrip(&pattern), pattern);
    }
}

#[test]
#[rustfmt::skip]
fn pattern_deserialize_invalid() {
    assert!(serde_json::from_str::<SparsityPattern>(r#"{"major_dim":3,"minor_dim":6,"major_offsets":[0,2,2,5],"minor_indices":[0,5,1,2,3]}"#).is_ok());
    assert!(serde_json::from_str::<SparsityPattern>(r#"{"major_dim":0,"minor_dim":0,"major_offsets":[],"minor_indices":[]}"#).is_err());
    assert!(serde_json::from_str::<SparsityPattern>(r#"{"major_dim":3,"minor_dim":6,"major_offsets":[0, 3, 5],"minor_indices":[0, 1, 2, 3, 5]}"#).is_err());
    assert!(serde_json::from_str::<SparsityPattern>(r#"{"major_dim":3,"minor_dim":6,"major_offsets":[1, 2, 2, 5],"minor_indices":[0, 5, 1, 2, 3]}"#).is_err());
    assert!(serde_json::from_str::<SparsityPattern>(r#"{"major_dim":3,"minor_dim":6,"major_offsets":[0, 2, 2, 4],"minor_indices":[0, 5, 1, 2, 3]}"#).is_err());
    assert!(serde_json::from_str::<SparsityPattern>(r#"{"major_dim":3,"minor_dim":6,"major_offsets":[0, 2, 2],"minor_indices":[0, 5, 1, 2, 3]}"#).is_err());
    assert!(serde_json::from_str::<SparsityPattern>(r#"{"major_dim":3,"minor_dim":6,"major_offsets":[0, 3, 2, 5],"minor_indices":[0, 1, 2, 3, 4]}"#).is_err());
    assert!(serde_json::from_str::<SparsityPattern>(r#"{"major_dim":3,"minor_dim":6,"major_offsets":[0, 2, 2, 5],"minor_indices":[0, 2, 3, 1, 4]}"#).is_err());
    assert!(serde_json::from_str::<SparsityPattern>(r#"{"major_dim":3,"minor_dim":6,"major_offsets":[0, 2, 2, 5],"minor_indices":[0, 6, 1, 2, 3]}"#).is_err());
    assert!(serde_json::from_str::<SparsityPattern>(r#"{"major_dim":3,"minor_dim":6,"major_offsets":[0, 2, 2, 5],"minor_indices":[0, 5, 2, 2, 3]}"#).is_err());
    assert!(serde_json::from_str::<SparsityPattern>(r#"{"major_dim":3,"minor_dim":6,"major_offsets":[0, 10, 2, 5],"minor_indices":[0, 5, 1, 2, 3]}"#).is_err());
}

#[test]
fn coo_roundtrip() {
    {
        // A COO matrix without entries
        let matrix =
            CooMatrix::<i32>::try_from_triplets(3, 2, Vec::new(), Vec::new(), Vec::new()).unwrap();

        assert_eq!(json_roundtrip(&matrix), matrix);
    }

    {
        // Arbitrary COO matrix, no duplicates
        let i = vec![0, 1, 0, 0, 2];
        let j = vec![0, 2, 1, 3, 3];
        let v = vec![2, 3, 7, 3, 1];
        let matrix =
            CooMatrix::<i32>::try_from_triplets(3, 5, i.clone(), j.clone(), v.clone()).unwrap();

        assert_eq!(json_roundtrip(&matrix), matrix);
    }
}

#[test]
fn coo_deserialize_invalid() {
    // Valid matrix: {"nrows":3,"ncols":5,"row_indices":[0,1,0,0,2],"col_indices":[0,2,1,3,3],"values":[2,3,7,3,1]}
    assert!(serde_json::from_str::<CooMatrix<i32>>(r#"{"nrows":3,"ncols":5,"row_indices":[0,1,0,0,2],"col_indices":[0,2,1,3,3],"values":[2,3,7,3,1]}"#).is_ok());
    assert!(serde_json::from_str::<CooMatrix<i32>>(r#"{"nrows":0,"ncols":0,"row_indices":[0,1,0,0,2],"col_indices":[0,2,1,3,3],"values":[2,3,7,3,4]}"#).is_err());
    assert!(serde_json::from_str::<CooMatrix<i32>>(r#"{"nrows":-3,"ncols":5,"row_indices":[0,1,0,0,2],"col_indices":[0,2,1,3,3],"values":[2,3,7,3,4]}"#).is_err());
    assert!(serde_json::from_str::<CooMatrix<i32>>(r#"{"nrows":3,"ncols":5,"row_indices":[0,1,0,0,2],"col_indices":[0,2,1,3,3],"values":[2,3,7,3]}"#).is_err());
    assert!(serde_json::from_str::<CooMatrix<i32>>(r#"{"nrows":3,"ncols":5,"row_indices":[0,1,0,0,2],"col_indices":[0,2,1,3,3],"values":[2,3,7,3,4,5]}"#).is_err());
    assert!(serde_json::from_str::<CooMatrix<i32>>(r#"{"nrows":3,"ncols":5,"row_indices":[0,1,0,0,2],"col_indices":[0,2,1,8,3],"values":[2,3,7,3,4]}"#).is_err());
    assert!(serde_json::from_str::<CooMatrix<i32>>(r#"{"nrows":3,"ncols":5,"row_indices":[0,1,0,0],"col_indices":[0,2,1,8,3],"values":[2,3,7,3,4]}"#).is_err());
    assert!(serde_json::from_str::<CooMatrix<i32>>(r#"{"nrows":3,"ncols":5,"row_indices":[0,10,0,0,2],"col_indices":[0,2,1,3,3],"values":[2,3,7,3,4]}"#).is_err());
    assert!(serde_json::from_str::<CooMatrix<i32>>(r#"{"nrows":3,"ncols":5,"row_indices":[0,1,0,0,2],"col_indices":[0,2,1,30,3],"values":[2,3,7,3,4]}"#).is_err());
}

#[test]
fn coo_deserialize_duplicates() {
    assert_eq!(
        serde_json::from_str::<CooMatrix<i32>>(
            r#"{"nrows":3,"ncols":5,"row_indices":[0,1,0,0,2,0,1],"col_indices":[0,2,1,3,3,0,2],"values":[2,3,7,3,1,5,6]}"#
        ).unwrap(),
        CooMatrix::<i32>::try_from_triplets(
            3,
            5,
            vec![0, 1, 0, 0, 2, 0, 1],
            vec![0, 2, 1, 3, 3, 0, 2],
            vec![2, 3, 7, 3, 1, 5, 6]
        )
        .unwrap()
    );
}

#[test]
fn csc_roundtrip() {
    {
        // A CSC matrix with zero explicitly stored entries
        let offsets = vec![0, 0, 0, 0];
        let indices = vec![];
        let values = Vec::<i32>::new();
        let matrix = CscMatrix::try_from_csc_data(2, 3, offsets, indices, values).unwrap();

        assert_eq!(json_roundtrip(&matrix), matrix);
    }

    {
        // An arbitrary CSC matrix
        let offsets = vec![0, 2, 2, 5];
        let indices = vec![0, 5, 1, 2, 3];
        let values = vec![0, 1, 2, 3, 4];
        let matrix =
            CscMatrix::try_from_csc_data(6, 3, offsets.clone(), indices.clone(), values.clone())
                .unwrap();

        assert_eq!(json_roundtrip(&matrix), matrix);
    }
}

#[test]
fn csc_deserialize_invalid() {
    // Valid matrix: {"nrows":6,"ncols":3,"col_offsets":[0,2,2,5],"row_indices":[0,5,1,2,3],"values":[0,1,2,3,4]}
    assert!(serde_json::from_str::<CscMatrix<i32>>(r#"{"nrows":6,"ncols":3,"col_offsets":[0,2,2,5],"row_indices":[0,5,1,2,3],"values":[0,1,2,3,4]}"#).is_ok());
    assert!(serde_json::from_str::<CscMatrix<i32>>(r#"{"nrows":0,"ncols":0,"col_offsets":[0,2,2,5],"row_indices":[0,5,1,2,3],"values":[0,1,2,3,4]}"#).is_err());
    assert!(serde_json::from_str::<CscMatrix<i32>>(r#"{"nrows":-6,"ncols":3,"col_offsets":[0,2,2,5],"row_indices":[0,5,1,2,3],"values":[0,1,2,3,4]}"#).is_err());
    assert!(serde_json::from_str::<CscMatrix<i32>>(r#"{"nrows":6,"ncols":3,"col_offsets":[0,2,2,5],"row_indices":[0,5,1,2,3],"values":[0,1,2,3]}"#).is_err());
    assert!(serde_json::from_str::<CscMatrix<i32>>(r#"{"nrows":6,"ncols":3,"col_offsets":[0,2,2,5],"row_indices":[0,5,1,2,3],"values":[0,1,2,3,4,5]}"#).is_err());
    assert!(serde_json::from_str::<CscMatrix<i32>>(r#"{"nrows":6,"ncols":3,"col_offsets":[0,2,2,5],"row_indices":[0,5,1,8,3],"values":[0,1,2,3,4]}"#).is_err());
    assert!(serde_json::from_str::<CscMatrix<i32>>(r#"{"nrows":6,"ncols":3,"col_offsets":[0,2,2,5],"row_indices":[0,5,1,2,3,1,1],"values":[0,1,2,3,4]}"#).is_err());
    assert!(serde_json::from_str::<CscMatrix<i32>>(r#"{"nrows":6,"ncols":3,"col_offsets":[0,10,2,5],"row_indices":[0,5,1,2,3],"values":[0,1,2,3,4]}"#).is_err());
    assert!(serde_json::from_str::<CscMatrix<i32>>(r#"{"nrows":3,"ncols":6,"row_offsets":[0,2,2,5],"col_indices":[0,5,1,2,3],"values":[0,1,2,3,4]}"#).is_err());
}

#[test]
fn csr_roundtrip() {
    {
        // A CSR matrix with zero explicitly stored entries
        let offsets = vec![0, 0, 0, 0];
        let indices = vec![];
        let values = Vec::<i32>::new();
        let matrix = CsrMatrix::try_from_csr_data(3, 2, offsets, indices, values).unwrap();

        assert_eq!(json_roundtrip(&matrix), matrix);
    }

    {
        // An arbitrary CSR matrix
        let offsets = vec![0, 2, 2, 5];
        let indices = vec![0, 5, 1, 2, 3];
        let values = vec![0, 1, 2, 3, 4];
        let matrix =
            CsrMatrix::try_from_csr_data(3, 6, offsets.clone(), indices.clone(), values.clone())
                .unwrap();

        assert_eq!(json_roundtrip(&matrix), matrix);
    }
}

#[test]
fn csr_deserialize_invalid() {
    // Valid matrix: {"nrows":3,"ncols":6,"row_offsets":[0,2,2,5],"col_indices":[0,5,1,2,3],"values":[0,1,2,3,4]}
    assert!(serde_json::from_str::<CsrMatrix<i32>>(r#"{"nrows":3,"ncols":6,"row_offsets":[0,2,2,5],"col_indices":[0,5,1,2,3],"values":[0,1,2,3,4]}"#).is_ok());
    assert!(serde_json::from_str::<CsrMatrix<i32>>(r#"{"nrows":0,"ncols":0,"row_offsets":[0,2,2,5],"col_indices":[0,5,1,2,3],"values":[0,1,2,3,4]}"#).is_err());
    assert!(serde_json::from_str::<CsrMatrix<i32>>(r#"{"nrows":-3,"ncols":6,"row_offsets":[0,2,2,5],"col_indices":[0,5,1,2,3],"values":[0,1,2,3,4]}"#).is_err());
    assert!(serde_json::from_str::<CsrMatrix<i32>>(r#"{"nrows":3,"ncols":6,"row_offsets":[0,2,2,5],"col_indices":[0,5,1,2,3],"values":[0,1,2,3]}"#).is_err());
    assert!(serde_json::from_str::<CsrMatrix<i32>>(r#"{"nrows":3,"ncols":6,"row_offsets":[0,2,2,5],"col_indices":[0,5,1,2,3],"values":[0,1,2,3,4,5]}"#).is_err());
    assert!(serde_json::from_str::<CsrMatrix<i32>>(r#"{"nrows":3,"ncols":6,"row_offsets":[0,2,2,5],"col_indices":[0,5,1,8,3],"values":[0,1,2,3,4]}"#).is_err());
    assert!(serde_json::from_str::<CsrMatrix<i32>>(r#"{"nrows":3,"ncols":6,"row_offsets":[0,2,2,5],"col_indices":[0,5,1,2,3,1,1],"values":[0,1,2,3,4]}"#).is_err());
    assert!(serde_json::from_str::<CsrMatrix<i32>>(r#"{"nrows":3,"ncols":6,"row_offsets":[0,10,2,5],"col_indices":[0,5,1,2,3],"values":[0,1,2,3,4]}"#).is_err());
    assert!(serde_json::from_str::<CsrMatrix<i32>>(r#"{"nrows":6,"ncols":3,"col_offsets":[0,2,2,5],"row_indices":[0,5,1,2,3],"values":[0,1,2,3,4]}"#).is_err());
}

proptest! {
    #[test]
    fn csc_roundtrip_proptest(csc in csc_strategy()) {
        prop_assert_eq!(json_roundtrip(&csc), csc);
    }

    #[test]
    fn csr_roundtrip_proptest(csr in csr_strategy()) {
        prop_assert_eq!(json_roundtrip(&csr), csr);
    }
}
*/

#[test]
fn csc_cholesky_roundtrip() {
    let mut a = Matrix5::new(
        40.0, 0.0, 0.0, 0.0, 0.0, 2.0, 60.0, 0.0, 0.0, 0.0, 1.0, 0.0, 11.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 50.0, 0.0, 1.0, 0.0, 0.0, 4.0, 10.0,
    );
    a.fill_upper_triangle_with_lower_triangle();
    test_cholesky(&CscMatrix::from(&a));

    let a = Matrix5::from_diagonal(&Vector5::new(40.0, 60.0, 11.0, 50.0, 10.0));
    test_cholesky(&CscMatrix::from(&a));

    let mut a = Matrix5::new(
        40.0, 0.0, 0.0, 0.0, 0.0, 2.0, 60.0, 0.0, 0.0, 0.0, 1.0, 0.0, 11.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 4.0, 10.0,
    );
    a.fill_upper_triangle_with_lower_triangle();
    test_cholesky(&CscMatrix::from(&a));

    let mut a = Matrix5::new(
        2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0,
    );
    a.fill_upper_triangle_with_lower_triangle();
    test_cholesky(&CscMatrix::from(&a));
}

fn test_cholesky(csc: &CscMatrix<f64>) {
    let sym_chol_csc = CscSymbolicCholesky::factor(csc.pattern().clone());
    let sym_chol_csc_deserialized = json_roundtrip(&sym_chol_csc);
    assert_eq!(
        sym_chol_csc, sym_chol_csc_deserialized,
        "symbolic factorization has to match"
    );

    let chol_csc = CscCholesky::factor(&csc).unwrap();
    let chol_csc_deserialized = json_roundtrip(&chol_csc);
    assert_eq!(
        chol_csc.symbolic_factorization(),
        chol_csc_deserialized.symbolic_factorization(),
        "symbolic part of full factorization has to match"
    );
    assert_eq!(
        chol_csc.l().pattern(),
        chol_csc_deserialized.l().pattern(),
        "pattern of l factor has to match"
    );
    assert_eq!(
        chol_csc.l().values(),
        chol_csc_deserialized.l().values(),
        "values of l factor have to match"
    );
}
