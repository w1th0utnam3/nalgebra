use alga::general::{Magma, Group, Loop, Monoid, Quasigroup, Semigroup, Op, Inverse};
use alga::linear::{EuclideanSpace, Transformation, Similarity, Isometry,
                   DirectIsometry, OrthogonalGroup, Translation, Rotation};
use alga::cmp::ApproxEq;
use alga::general::Identity as AlgebraIdentity;

use structs::Identity;

impl<O: Op> AlgebraIdentity<O> for Identity {
    #[inline]
    fn id() -> Identity {
        Identity::new()
    }
}

impl ApproxEq for Identity {
    type Eps = ();

    #[inline]
    fn default_epsilon() -> () {
        ()
    }

    #[inline]
    fn approx_eq_eps(&self, _: &Identity, _: &()) -> bool {
        true
    }
}

/*
 *
 * Algebraic structures.
 *
 */
impl<O: Op> Group<O>      for Identity { }
impl<O: Op> Loop<O>       for Identity { }
impl<O: Op> Monoid<O>     for Identity { }
impl<O: Op> Quasigroup<O> for Identity { }
impl<O: Op> Semigroup<O>  for Identity { }
impl<O: Op> Magma<O>      for Identity {
    #[inline]
    fn operate(self, _: Self) -> Identity {
        self
    }
}

impl<O: Op> Inverse<O> for Identity {
    #[inline]
    fn inv(self) -> Self {
        self
    }
}

/*
 *
 * Matrix groups.
 *
 */
impl<P: EuclideanSpace> Transformation<P> for Identity {
    #[inline]
    fn transform_point(&self, pt: &P) -> P {
        pt.clone()
    }

    #[inline]
    fn transform_vector(&self, v: &P::Vector) -> P::Vector {
        v.clone()
    }

    #[inline]
    fn inverse_transform_point(&self, pt: &P) -> P {
        pt.clone()
    }

    #[inline]
    fn inverse_transform_vector(&self, v: &P::Vector) -> P::Vector {
        v.clone()
    }
}

impl<P: EuclideanSpace> Similarity<P> for Identity {
    type Translation = Identity;
    type Rotation    = Identity;

    #[inline]
    fn translation(&self) -> Identity {
        Identity::new()
    }

    #[inline]
    fn rotation(&self) -> Identity {
        Identity::new()
    }

    #[inline]
    fn scaling_factor(&self) -> P::Real {
        ::one()
    }
}

impl<P: EuclideanSpace> Isometry<P> for Identity { }
impl<P: EuclideanSpace> DirectIsometry<P> for Identity { }
impl<P: EuclideanSpace> OrthogonalGroup<P> for Identity { }
impl<P: EuclideanSpace> Rotation<P> for Identity { }
impl<P: EuclideanSpace> Translation<P> for Identity { }
