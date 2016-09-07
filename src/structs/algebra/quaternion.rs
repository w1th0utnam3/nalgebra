use std::ops::Neg;

use algebra::general::{Magma, Group, Loop, Monoid, Quasigroup, Semigroup, Real, Recip,
                       Multiplicative, Additive};
use algebra::linear::{Transformation, Similarity, Isometry, DirectIsometry,
                      OrthogonalGroup};
use algebra::linear::Rotation as AlgebraRotation;
use algebra::general::Identity as AlgebraIdentity;
use algebra::cmp::ApproxEq as AlgebraApproxEq;

use structs::{Identity, Quaternion, UnitQuaternion, Point3, Vector3};
use traits::{BaseNum, BaseFloat, Inverse};

/*
 *
 * Implementations for Quaternion.
 *
 */
impl<N: Copy + AlgebraApproxEq<Eps = N>> AlgebraApproxEq for Quaternion<N> {
    type Eps = N;

    #[inline]
    fn default_epsilon() -> N {
        <N as AlgebraApproxEq>::default_epsilon()
    }

    #[inline]
    fn approx_eq_eps(&self, other: &Quaternion<N>, epsilon: &N) -> bool {
        AlgebraApproxEq::approx_eq_eps(&self.w, &other.w, &epsilon) &&
        AlgebraApproxEq::approx_eq_eps(self.vector(), other.vector(), &epsilon)
    }
}


impl<N: BaseNum> AlgebraIdentity<Multiplicative> for Quaternion<N> {
    #[inline]
    fn id() -> Quaternion<N> {
        ::one()
    }
}

impl<N: BaseNum> AlgebraIdentity<Additive> for Quaternion<N> {
    #[inline]
    fn id() -> Quaternion<N> {
        ::zero()
    }
}

impl<N: BaseFloat> Recip for Quaternion<N> {
    type Result = Quaternion<N>;

    #[inline]
    fn recip(mut self) -> Quaternion<N> {
        self.inverse_mut();

        self
    }
}

// FIXME: in the end, we will keep only Real (instead of BaseNum + Real).
impl<N: BaseFloat + Real> Group<Multiplicative>      for Quaternion<N> { }
impl<N: BaseFloat + Real> Loop<Multiplicative>       for Quaternion<N> { }
impl<N: BaseFloat + Real> Monoid<Multiplicative>     for Quaternion<N> { }
impl<N: BaseFloat + Real> Quasigroup<Multiplicative> for Quaternion<N> { }
impl<N: BaseFloat + Real> Semigroup<Multiplicative>  for Quaternion<N> { }
impl<N: BaseFloat + Real> Magma<Multiplicative>      for Quaternion<N> {
    #[inline]
    fn operate(self, lhs: Quaternion<N>) -> Quaternion<N> {
        self * lhs
    }
}

impl<N: BaseNum + Real> Group<Additive>      for Quaternion<N> { }
impl<N: BaseNum + Real> Loop<Additive>       for Quaternion<N> { }
impl<N: BaseNum + Real> Monoid<Additive>     for Quaternion<N> { }
impl<N: BaseNum + Real> Quasigroup<Additive> for Quaternion<N> { }
impl<N: BaseNum + Real> Semigroup<Additive>  for Quaternion<N> { }
impl<N: BaseNum + Real> Magma<Additive>      for Quaternion<N> {
    #[inline]
    fn operate(self, lhs: Quaternion<N>) -> Quaternion<N> {
        self + lhs
    }
}


/*
 *
 * Implementations for UnitQuaternion.
 *
 */
impl<N: Copy + AlgebraApproxEq<Eps = N>> AlgebraApproxEq for UnitQuaternion<N> {
    type Eps = N;

    #[inline]
    fn default_epsilon() -> N {
        <N as AlgebraApproxEq>::default_epsilon()
    }

    #[inline]
    fn approx_eq_eps(&self, other: &UnitQuaternion<N>, epsilon: &N) -> bool {
        self.as_ref().approx_eq_eps(other.as_ref(), epsilon)
    }
}


impl<N: BaseNum> AlgebraIdentity<Multiplicative> for UnitQuaternion<N> {
    #[inline]
    fn id() -> UnitQuaternion<N> {
        ::one()
    }
}

impl<N: BaseNum + Neg<Output = N>> Recip for UnitQuaternion<N> {
    type Result = UnitQuaternion<N>;

    #[inline]
    fn recip(mut self) -> UnitQuaternion<N> {
        self.inverse_mut();

        self
    }
}

// FIXME: in the end, we will keep only Real (instead of BaseNum + Real).
impl<N: BaseNum + Real> Group<Multiplicative>      for UnitQuaternion<N> { }
impl<N: BaseNum + Real> Loop<Multiplicative>       for UnitQuaternion<N> { }
impl<N: BaseNum + Real> Monoid<Multiplicative>     for UnitQuaternion<N> { }
impl<N: BaseNum + Real> Quasigroup<Multiplicative> for UnitQuaternion<N> { }
impl<N: BaseNum + Real> Semigroup<Multiplicative>  for UnitQuaternion<N> { }
impl<N: BaseNum + Real> Magma<Multiplicative>      for UnitQuaternion<N> {
    #[inline]
    fn operate(self, lhs: UnitQuaternion<N>) -> UnitQuaternion<N> {
        self * lhs
    }
}
impl<N: BaseNum + Real> Transformation<Point3<N>> for UnitQuaternion<N> {
    #[inline]
    fn transform_point(&self, pt: &Point3<N>) -> Point3<N> {
        *self * *pt
    }

    #[inline]
    fn transform_vector(&self, v: &Vector3<N>) -> Vector3<N> {
        *self * *v
    }

    #[inline]
    fn inverse_transform_point(&self, pt: &Point3<N>) -> Point3<N> {
        *pt * *self
    }

    #[inline]
    fn inverse_transform_vector(&self, v: &Vector3<N>) -> Vector3<N> {
        *v * *self
    }
}

impl<N: BaseNum + Real> Similarity<Point3<N>> for UnitQuaternion<N> {
    type Translation = Identity;
    type Rotation    = UnitQuaternion<N>;

    #[inline]
    fn translation(&self) -> Identity {
        Identity::new()
    }

    #[inline]
    fn rotation(&self) -> UnitQuaternion<N> {
        *self
    }

    #[inline]
    fn scaling_factor(&self) -> N {
        ::one()
    }
}

impl<N: BaseNum + Real> Isometry<Point3<N>>        for UnitQuaternion<N> { }
impl<N: BaseNum + Real> DirectIsometry<Point3<N>>  for UnitQuaternion<N> { }
impl<N: BaseNum + Real> OrthogonalGroup<Point3<N>> for UnitQuaternion<N> { }
impl<N: BaseNum + Real> AlgebraRotation<Point3<N>> for UnitQuaternion<N> { }
