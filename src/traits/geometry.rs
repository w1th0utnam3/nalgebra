//! Traits of operations having a well-known or explicit geometric meaning.

use std::ops::Neg;
use num::Float;
use traits::structure::{BaseFloat, Mat};

/// Trait of object which represent a translation, and to wich new translation
/// can be appended.
pub trait Translation {
    type TranslationType;

    // FIXME: add a "from translation: translantion(V) -> Self ?
    /// Gets the translation associated with this object.
    fn translation(&self) -> Self::TranslationType;

    /// Gets the inverse translation associated with this object.
    fn inv_translation(&self) -> Self::TranslationType;

    /// Appends a translation to this object.
    fn append_translation_mut(&mut self, &Self::TranslationType);

    /// Appends the translation `amount` to a copy of `t`.
    fn append_translation(&self, amount: &Self::TranslationType) -> Self;

    /// Prepends a translation to this object.
    fn prepend_translation_mut(&mut self, &Self::TranslationType);

    /// Prepends the translation `amount` to a copy of `t`.
    fn prepend_translation(&self, amount: &Self::TranslationType) -> Self;

    /// Sets the translation.
    fn set_translation(&mut self, Self::TranslationType);
}

/// Trait of objects able to translate other objects. This is typically
/// implemented by vectors to translate points.
pub trait Translate<V> {
    /// Apply a translation to an object.
    fn translate(&self, &V) -> V;

    /// Apply an inverse translation to an object.
    fn inv_translate(&self, &V) -> V;
}

/// Trait of object which can represent a rotation, and to which new rotations can be appended. A
/// rotation is assumed to be an isometry without translation and without reflexion.
pub trait Rotation {
    type RotationType;

    /// Gets the rotation associated with `self`.
    fn rotation(&self) -> Self::RotationType;

    /// Gets the inverse rotation associated with `self`.
    fn inv_rotation(&self) -> Self::RotationType;

    /// Appends a rotation to this object.
    fn append_rotation_mut(&mut self, &Self::RotationType);

    /// Appends the rotation `amount` to a copy of `t`.
    fn append_rotation(&self, amount: &Self::RotationType) -> Self;

    /// Prepends a rotation to this object.
    fn prepend_rotation_mut(&mut self, &Self::RotationType);

    /// Prepends the rotation `amount` to a copy of `t`.
    fn prepend_rotation(&self, amount: &Self::RotationType) -> Self;

    /// Sets the rotation of `self`.
    fn set_rotation(&mut self, Self::RotationType);
}

/// Trait of objects able to rotate other objects.
///
/// This is typically implemented by matrices which rotate vectors.
pub trait Rotate<V> {
    /// Applies a rotation to `v`.
    fn rotate(&self, v: &V) -> V;

    /// Applies an inverse rotation to `v`.
    fn inv_rotate(&self, v: &V) -> V;
}

/// Various composition of rotation and translation.
///
/// Utilities to make rotations with regard to a point different than the origin.  All those
/// operations are the composition of rotations and translations.
///
/// Those operations are automatically implemented in term of the `Rotation` and `Translation`
/// traits.
pub trait RotationWithTranslation: Rotation + Translation + Sized
    where Self::TranslationType: Neg<Output = <Self as Translation>::TranslationType> + Copy {
    /// Applies a rotation centered on a specific point.
    ///
    /// # Arguments
    ///   * `t` - the object to be rotated.
    ///   * `amount` - the rotation to apply.
    ///   * `point` - the center of rotation.
    #[inline]
    fn append_rotation_wrt_point(&self, amount: &Self::RotationType, center: &Self::TranslationType) -> Self {
        let mut res = Translation::append_translation(self, &-*center);

        res.append_rotation_mut(amount);
        res.append_translation_mut(center);

        res
    }

    /// Rotates `self` using a specific center of rotation.
    ///
    /// The rotation is applied in-place.
    ///
    /// # Arguments
    ///   * `amount` - the rotation to be applied
    ///   * `center` - the new center of rotation
    #[inline]
    fn append_rotation_wrt_point_mut(&mut self,
                                     amount: &Self::RotationType,
                                     center: &Self::TranslationType) {
        self.append_translation_mut(&-*center);
        self.append_rotation_mut(amount);
        self.append_translation_mut(center);
    }

    /// Applies a rotation centered on the translation of `m`.
    /// 
    /// # Arguments
    ///   * `t` - the object to be rotated.
    ///   * `amount` - the rotation to apply.
    #[inline]
    fn append_rotation_wrt_center(&self, amount: &Self::RotationType) -> Self {
        RotationWithTranslation::append_rotation_wrt_point(self, amount, &self.translation())
    }

    /// Applies a rotation centered on the translation of `m`.
    ///
    /// The rotation os applied on-place.
    ///
    /// # Arguments
    ///   * `amount` - the rotation to apply.
    #[inline]
    fn append_rotation_wrt_center_mut(&mut self, amount: &Self::RotationType) {
        let center = self.translation();
        self.append_rotation_wrt_point_mut(amount, &center)
    }
}

impl<M> RotationWithTranslation for M
where M: Rotation + Translation,
      M::TranslationType: Neg<Output = <M as Translation>::TranslationType> + Copy {
}

/// Trait of transformation having a rotation extractable as a rotation matrix. This can typically
/// be implemented by quaternions to convert them to a rotation matrix.
pub trait RotationMatrix : Rotation {
    /// The output rotation matrix type.
    type RotationMatrixType: Mat + Rotation;

    /// Gets the rotation matrix represented by `self`.
    fn to_rot_mat(&self) -> Self::RotationMatrixType;
}

/// Composition of a rotation and an absolute value.
///
/// The operation is accessible using the `RotationMatrix`, `Absolute`, and `RMul` traits, but
/// doing so is not easy in generic code as it can be a cause of type over-parametrization.
pub trait AbsoluteRotate<V> {
    /// This is the same as:
    ///
    /// ```.ignore
    ///     self.rotation_matrix().absolute().rmul(v)
    /// ```
    fn absolute_rotate(&self, v: &V) -> V;
}

/// Trait of object which represent a transformation, and to which new transformations can
/// be appended.
///
/// A transformation is assumed to be an isometry without reflexion.
pub trait Transformation {
    type TransformationType;

    /// Gets the transformation of `self`.
    fn transformation(&self) -> Self::TransformationType;

    /// Gets the inverse transformation of `self`.
    fn inv_transformation(&self) -> Self::TransformationType;

    /// Appends a transformation to this object.
    fn append_transformation_mut(&mut self, &Self::TransformationType);

    /// Appends the transformation `amount` to a copy of `t`.
    fn append_transformation(&self, amount: &Self::TransformationType) -> Self;

    /// Prepends a transformation to this object.
    fn prepend_transformation_mut(&mut self, &Self::TransformationType);

    /// Prepends the transformation `amount` to a copy of `t`.
    fn prepend_transformation(&self, amount: &Self::TransformationType) -> Self;

    /// Sets the transformation of `self`.
    fn set_transformation(&mut self, Self::TransformationType);
}

/// Trait of objects able to transform other objects.
///
/// This is typically implemented by matrices which transform vectors.
pub trait Transform<V> {
    /// Applies a transformation to `v`.
    fn transform(&self, &V) -> V;

    /// Applies an inverse transformation to `v`.
    fn inv_transform(&self, &V) -> V;
}

/// Traits of objects having a dot product.
pub trait Dot {
    type DotProductType;

    /// Computes the dot (inner) product of two vectors.
    #[inline]
    fn dot(&self, other: &Self) -> Self::DotProductType;
}

/// Traits of objects having an euclidian norm.
pub trait Norm {
    type NormType: BaseFloat;

    /// Computes the norm of `self`.
    #[inline]
    fn norm(&self) -> Self::NormType {
        self.sqnorm().sqrt()
    }

    /// Computes the squared norm of `self`.
    ///
    /// This is usually faster than computing the norm itself.
    fn sqnorm(&self) -> Self::NormType;

    /// Gets the normalized version of a copy of `v`.
    fn normalize(&self) -> Self;

    /// Normalizes `self`.
    fn normalize_mut(&mut self) -> Self::NormType;
}

/**
 * Trait of elements having a cross product.
 */
pub trait Cross {
    /// The cross product output.
    type CrossProductType;

    /// Computes the cross product between two elements (usually vectors).
    fn cross(&self, other: &Self) -> Self::CrossProductType;
}

/**
 * Trait of elements having a cross product operation which can be expressed as a matrix.
 */
pub trait CrossMatrix {
    type CrossMatrixFormType;

    /// The matrix associated to any cross product with this vector. I.e. `v.cross(anything)` =
    /// `v.cross_matrix().rmul(anything)`.
    fn cross_matrix(&self) -> Self::CrossMatrixFormType;
}

/// Traits of objects which can be put in homogeneous coordinates form.
pub trait ToHomogeneous {
    type HomogeneousFormType;

    /// Gets the homogeneous coordinates form of this object.
    fn to_homogeneous(&self) -> Self::HomogeneousFormType;
}

/// Traits of objects which can be build from an homogeneous coordinate form.
pub trait FromHomogeneous<U> {
    /// Builds an object from its homogeneous coordinate form.
    ///
    /// Note that this this is not required that `from` is the inverse of `to_homogeneous`.
    /// Typically, `from` will remove some informations unrecoverable by `to_homogeneous`.
    fn from(&U) -> Self;
}

/// Trait of vectors able to sample a unit sphere.
///
/// The number of sample must be sufficient to approximate a sphere using a support mapping
/// function.
pub trait UniformSphereSample {
    /// Iterate through the samples.
    fn sample<F: FnMut(Self)>(F);
}

/// The zero element of a vector space, seen as an element of its embeding affine space.
// XXX: once associated types are suported, move this to the `AnyPnt` trait.
pub trait Orig {
    /// The trivial origin.
    fn orig() -> Self;
    /// Returns true if this points is exactly the trivial origin.
    fn is_orig(&self) -> bool;
}
