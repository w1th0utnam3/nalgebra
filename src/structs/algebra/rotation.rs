#![macro_use]

macro_rules! use_rotation_group_modules(
    () => {
        use algebra::general::{Magma, Group, Loop, Monoid, Quasigroup, Semigroup,
                               Real, Recip, Multiplicative};
        use algebra::linear::{Transformation, Similarity, Isometry, DirectIsometry,
                              OrthogonalGroup};
        use algebra::linear::Rotation as AlgebraRotation;
        use algebra::general::Identity as AlgebraIdentity;
        use algebra::cmp::ApproxEq as AlgebraApproxEq;

        use structs::Identity;
    }
);

macro_rules! rotation_group_impl(
    ($t: ident, $point: ident, $vector: ident) => {
        impl<N: Copy + AlgebraApproxEq<Eps = N>> AlgebraApproxEq for $t<N> {
            type Eps = N;

            #[inline]
            fn default_epsilon() -> N {
                <N as AlgebraApproxEq>::default_epsilon()
            }

            #[inline]
            fn approx_eq_eps(&self, other: &$t<N>, epsilon: &N) -> bool {
                AlgebraApproxEq::approx_eq_eps(&self.submatrix, &other.submatrix, &epsilon)
            }
        }


        /*
         *
         * Algebraic structures.
         *
         */
        impl<N: BaseNum> AlgebraIdentity<Multiplicative> for $t<N> {
            #[inline]
            fn id() -> Self {
                ::one()
            }
        }

        impl<N: Copy> Recip for $t<N> {
            type Result = $t<N>;

            #[inline]
            fn recip(mut self) -> $t<N> {
                self.inverse_mut();

                self
            }
        }

        // FIXME: in the end, we will keep only Real (instead of BaseNum + Real).
        impl<N: BaseNum + Real> Group<Multiplicative>      for $t<N> { }
        impl<N: BaseNum + Real> Loop<Multiplicative>       for $t<N> { }
        impl<N: BaseNum + Real> Monoid<Multiplicative>     for $t<N> { }
        impl<N: BaseNum + Real> Quasigroup<Multiplicative> for $t<N> { }
        impl<N: BaseNum + Real> Semigroup<Multiplicative>  for $t<N> { }
        impl<N: BaseNum + Real> Magma<Multiplicative>      for $t<N> {
            #[inline]
            fn operate(self, lhs: Self) -> Self {
                self * lhs
            }
        }

        /*
         *
         * Transformation groups.
         *
         */
        impl<N: BaseNum + Real> Transformation<$point<N>> for $t<N> {
            #[inline]
            fn transform_point(&self, pt: &$point<N>) -> $point<N> {
                *self * *pt
            }

            #[inline]
            fn transform_vector(&self, v: &$vector<N>) -> $vector<N> {
                *self * *v
            }

            #[inline]
            fn inverse_transform_point(&self, pt: &$point<N>) -> $point<N> {
                *pt * *self
            }

            #[inline]
            fn inverse_transform_vector(&self, v: &$vector<N>) -> $vector<N> {
                *v * *self
            }
        }

        impl<N: BaseNum + Real> Similarity<$point<N>> for $t<N> {
            type Translation = Identity;
            type Rotation    = $t<N>;

            #[inline]
            fn translation(&self) -> Identity {
                Identity::new()
            }

            #[inline]
            fn rotation(&self) -> $t<N> {
                *self
            }

            #[inline]
            fn scaling_factor(&self) -> N {
                ::one()
            }
        }

        impl<N: BaseNum + Real> Isometry<$point<N>> for $t<N> { }
        impl<N: BaseNum + Real> DirectIsometry<$point<N>> for $t<N> { }
        impl<N: BaseNum + Real> OrthogonalGroup<$point<N>> for $t<N> { }
        impl<N: BaseNum + Real> AlgebraRotation<$point<N>> for $t<N> { }
    }
);
