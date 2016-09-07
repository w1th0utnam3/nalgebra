#![macro_use]

macro_rules! use_vector_space_modules(
    () => {
        use algebra::general::{Magma, Field, Real, RingCommutative, GroupAbelian,
                               Group, Loop, Monoid, Quasigroup, Semigroup, Module,
                               Additive, Multiplicative, Recip};
        use algebra::linear::{VectorSpace, NormedSpace, InnerSpace, FiniteDimVectorSpace,
                              Similarity, Isometry, DirectIsometry};
        use algebra::general::Identity as AlgebraIdentity;
        use algebra::linear::Translation as AlgebraTranslation;
        use algebra::linear::Transformation as AlgebraTransformation;
        use algebra::cmp::ApproxEq as AlgebraApproxEq;
        use structs::Identity;
    }
);

macro_rules! vector_space_impl(
    ($t: ident, $point: ident, $dimension: expr, $($compN: ident),+) => {
        impl<N: AlgebraApproxEq> AlgebraApproxEq for $t<N> {
            type Eps = N::Eps;

            #[inline]
            fn default_epsilon() -> N::Eps {
                N::default_epsilon()
            }

            #[inline]
            fn approx_eq_eps(&self, other: &$t<N>, epsilon: &N::Eps) -> bool {
                $(AlgebraApproxEq::approx_eq_eps(&self.$compN, &other.$compN, &epsilon))&&+
            }
        }

        /*
         *
         * Algebraic structures.
         *
         */
        impl<N: Copy + AlgebraIdentity<Additive>> AlgebraIdentity<Additive> for $t<N> {
            #[inline]
            fn id() -> Self {
                Repeat::repeat(AlgebraIdentity::id())
            }
        }

        impl<N: Clone + Add<Output = N>> Magma<Additive> for $t<N> {
            fn operate(self, other: $t<N>) -> $t<N> {
                self + other
            }
        }

        product_space_inherit_structure!($t, GroupAbelian<Additive>);
        product_space_inherit_structure!($t, Group<Additive>);
        product_space_inherit_structure!($t, Loop<Additive>);
        product_space_inherit_structure!($t, Monoid<Additive>);
        product_space_inherit_structure!($t, Quasigroup<Additive>);
        product_space_inherit_structure!($t, Semigroup<Additive>);

        // Seen as a translation, this is a multiplicative abelian group where the multiplication
        // is the addition.
        impl<N: Zero> AlgebraIdentity<Multiplicative> for $t<N> {
            #[inline]
            fn id() -> Self {
                ::zero()
            }
        }

        impl<N: Copy + Neg<Output = N>> Recip for $t<N> {
            type Result = $t<N>;

            #[inline]
            fn recip(self) -> $t<N> {
                -self
            }
        }

        impl<N: Clone + Add<Output = N>> Magma<Multiplicative> for $t<N> {
            fn operate(self, other: $t<N>) -> $t<N> {
                self + other
            }
        }

        product_space_inherit_structure!($t, GroupAbelian<Multiplicative>);
        product_space_inherit_structure!($t, Group<Multiplicative>);
        product_space_inherit_structure!($t, Loop<Multiplicative>);
        product_space_inherit_structure!($t, Monoid<Multiplicative>);
        product_space_inherit_structure!($t, Quasigroup<Multiplicative>);
        product_space_inherit_structure!($t, Semigroup<Multiplicative>);

        /*
         *
         * Transformation groups.
         *
         */
        impl<N: BaseNum + Real> AlgebraTransformation<$point<N>> for $t<N> {
            #[inline]
            fn transform_point(&self, pt: &$point<N>) -> $point<N> {
                *pt + *self
            }

            #[inline]
            fn transform_vector(&self, v: &$t<N>) -> $t<N> {
                *v + *self
            }

            #[inline]
            fn inverse_transform_point(&self, pt: &$point<N>) -> $point<N> {
                *pt - *self
            }

            #[inline]
            fn inverse_transform_vector(&self, v: &$t<N>) -> $t<N> {
                *v - *self
            }
        }

        impl<N: BaseNum + Real> Similarity<$point<N>> for $t<N> {
            type Translation = $t<N>;
            type Rotation    = Identity;

            #[inline]
            fn translation(&self) -> $t<N> {
                *self
            }

            #[inline]
            fn rotation(&self) -> Identity {
                Identity::new()
            }

            #[inline]
            fn scaling_factor(&self) -> N {
                ::one()
            }
        }

        impl<N: BaseNum + Real> Isometry<$point<N>> for $t<N> { }
        impl<N: BaseNum + Real> DirectIsometry<$point<N>> for $t<N> { }
        impl<N: BaseNum + Real> AlgebraTranslation<$point<N>> for $t<N> { }


        /*
         *
         * Vector space.
         *
         */
        impl<N> Module for $t<N> where N: Copy + Zero + Neg<Output = N> + Add<N, Output = N> +
                                          AlgebraApproxEq + RingCommutative {
            type Ring = N;
        }

        impl<N> VectorSpace for $t<N> where N: Copy + Zero + Neg<Output = N> + Add<N, Output = N> +
                                            AlgebraApproxEq + Field {
            type Field = N;
        }

        impl<N> FiniteDimVectorSpace for $t<N> where N: Copy + Zero + One + Neg<Output = N> +
                                                        Add<N, Output = N> + AlgebraApproxEq + Field {
            #[inline]
            fn dimension() -> usize {
                $dimension
            }

            #[inline]
            fn canonical_basis<F: FnOnce(&[$t<N>])>(f: F) {
                let basis = [
                    $($t::$compN()),*
                ];

                f(&basis[..])
            }

            #[inline]
            fn component(&self, i: usize) -> N {
                self[i]
            }

            #[inline]
            unsafe fn component_unchecked(&self, i: usize)  -> N {
                self.at_fast(i)
            }
        }

        impl<N: Real> NormedSpace for $t<N> {
            #[inline]
            fn norm_squared(&self) -> N {
                self.inner_product(self)
            }

            #[inline]
            fn norm(&self) -> N {
                self.norm_squared().sqrt()
            }

            #[inline]
            fn normalize(&self) -> Self {
                *self / self.norm()
            }

            #[inline]
            fn normalize_mut(&mut self) -> N {
                let n = self.norm();
                *self /= n;

                n
            }

            #[inline]
            fn try_normalize(&self, min_norm: &N) -> Option<Self> {
                let n = self.norm();

                if n <= *min_norm {
                    None
                }
                else {
                    Some(*self / n)
                }
            }

            #[inline]
            fn try_normalize_mut(&mut self, min_norm: &N) -> Option<N> {
                let n = self.norm();

                if n <= *min_norm {
                    None
                }
                else {
                    *self /= n;
                    Some(n)
                }
            }
        }

        impl<N: Real> InnerSpace for $t<N> {
            type Real = N;

            #[inline]
            fn inner_product(&self, other: &Self) -> N {
                fold_add!($(self.$compN * other.$compN ),+)
            }
        }
    }
);

macro_rules! product_space_inherit_structure(
    ($t: ident, $marker: ident<$operator: ident>) => {
        impl<N> $marker<$operator> for $t<N>
            where N: Copy + Zero + Neg<Output = N> + Add<N, Output = N> + AlgebraApproxEq +
                     $marker<$operator>
                 { }
    }
);
