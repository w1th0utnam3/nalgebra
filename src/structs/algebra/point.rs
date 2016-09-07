#![macro_use]

macro_rules! use_euclidean_space_modules(
    () => {
        use algebra::general::{Field, Real};
        use algebra::linear::{AffineSpace, EuclideanSpace};
        use algebra::cmp::ApproxEq as AlgebraApproxEq;
    }
);


macro_rules! euclidean_space_impl(
    ($t: ident, $vector: ident) => {
        impl<N> AffineSpace for $t<N>
            where N: Copy + Zero + Neg<Output = N> + Add<N, Output = N> +
                     Sub<N, Output = N> + AlgebraApproxEq + Field {
            type Translation = $vector<N>;
            
            #[inline]
            fn translate_by(&self, vector: &Self::Translation) -> Self {
                *self + *vector
            }

            #[inline]
            fn subtract(&self, other: &Self) -> Self::Translation {
                *self - *other
            }
        }

        impl<N: Real> EuclideanSpace for $t<N> {
            type Vector = $vector<N>;
            type Real   = N;
        }
    }
);
