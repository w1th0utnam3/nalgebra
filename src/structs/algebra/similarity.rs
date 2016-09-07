#![macro_use]

macro_rules! use_similarity_group_modules(
    () => {
        use algebra::general::{Magma, Group, Loop, Monoid, Quasigroup, Semigroup,
                               Real, Recip, Multiplicative};
        use algebra::linear::Similarity;
        use algebra::linear::Transformation as AlgebraTransformation;
        use algebra::general::Identity as AlgebraIdentity;
        use algebra::cmp::ApproxEq as AlgebraApproxEq;
    }
);

macro_rules! similarity_group_impl(
    ($t: ident, $point: ident, $vector: ident, $rotation: ident) => {

        impl<N: Copy + AlgebraApproxEq<Eps = N>> AlgebraApproxEq for $t<N> {
            type Eps = N;

            #[inline]
            fn default_epsilon() -> N {
                <N as AlgebraApproxEq>::default_epsilon()
            }

            #[inline]
            fn approx_eq_eps(&self, other: &$t<N>, epsilon: &N) -> bool {
                AlgebraApproxEq::approx_eq_eps(&self.scale, &other.scale, &epsilon) &&
                AlgebraApproxEq::approx_eq_eps(&self.isometry, &other.isometry, &epsilon)
            }
        }


        /*
         *
         * Algebraic structures.
         *
         */
        impl<N: BaseFloat> AlgebraIdentity<Multiplicative> for $t<N> {
            #[inline]
            fn id() -> Self {
                ::one()
            }
        }

        impl<N: BaseFloat> Recip for $t<N> {
            type Result = $t<N>;

            #[inline]
            fn recip(mut self) -> $t<N> {
                self.inverse_mut();

                self
            }
        }

        impl<N: BaseFloat + Real> Magma<Multiplicative> for $t<N> {
            #[inline]
            fn operate(self, rhs: $t<N>) -> $t<N> {
                self * rhs
            }
        }

        // FIXME: in the end, we will keep only Real (instead of BaseFloat + Real).
        impl<N: BaseFloat + Real> Group<Multiplicative>      for $t<N> { }
        impl<N: BaseFloat + Real> Loop<Multiplicative>       for $t<N> { }
        impl<N: BaseFloat + Real> Monoid<Multiplicative>     for $t<N> { }
        impl<N: BaseFloat + Real> Quasigroup<Multiplicative> for $t<N> { }
        impl<N: BaseFloat + Real> Semigroup<Multiplicative>  for $t<N> { }

        /*
         *
         * Matrix groups.
         *
         */
        impl<N: BaseFloat + Real> AlgebraTransformation<$point<N>> for $t<N> {
            #[inline]
            fn transform_point(&self, pt: &$point<N>) -> $point<N> {
                self.transform(pt)
            }

            #[inline]
            fn transform_vector(&self, v: &$vector<N>) -> $vector<N> {
                self.isometry.transform_vector(&(*v * self.scale))
            }

            #[inline]
            fn inverse_transform_point(&self, pt: &$point<N>) -> $point<N> {
                self.inverse_transform(pt)
            }

            #[inline]
            fn inverse_transform_vector(&self, v: &$vector<N>) -> $vector<N> {
                self.isometry.inverse_transform_vector(v) / self.scale
            }
        }

        impl<N: BaseFloat + Real> Similarity<$point<N>> for $t<N> {
            type Translation = $vector<N>;
            type Rotation    = $rotation<N>;

            #[inline]
            fn translation(&self) -> $vector<N> {
                self.isometry.translation()
            }

            #[inline]
            fn rotation(&self) -> $rotation<N> {
                self.isometry.rotation()
            }

            #[inline]
            fn scaling_factor(&self) -> N {
                self.scale
            }
        }
    }
);
