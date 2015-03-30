//! nalgebra trait implementation for primitive types.

#![allow(missing_docs)]
#![allow(non_camel_case_types)]

use traits::structure::Cast;

macro_rules! cast_impl(
    ($t:ident, $from: ident) => (
        impl Cast<$from> for $t {
            #[inline(always)]
            fn from(t: $from) -> $t {
                t as $t
            }
        }
    )
);

cast_impl!(f64, f64);
cast_impl!(f64, f32);
cast_impl!(f64, i64);
cast_impl!(f64, i32);
cast_impl!(f64, i16);
cast_impl!(f64, i8);
cast_impl!(f64, u64);
cast_impl!(f64, u32);
cast_impl!(f64, u16);
cast_impl!(f64, u8);
cast_impl!(f64, isize);
cast_impl!(f64, usize);

cast_impl!(f32, f64);
cast_impl!(f32, f32);
cast_impl!(f32, i64);
cast_impl!(f32, i32);
cast_impl!(f32, i16);
cast_impl!(f32, i8);
cast_impl!(f32, u64);
cast_impl!(f32, u32);
cast_impl!(f32, u16);
cast_impl!(f32, u8);
cast_impl!(f32, isize);
cast_impl!(f32, usize);

cast_impl!(i64, f64);
cast_impl!(i64, f32);
cast_impl!(i64, i64);
cast_impl!(i64, i32);
cast_impl!(i64, i16);
cast_impl!(i64, i8);
cast_impl!(i64, u64);
cast_impl!(i64, u32);
cast_impl!(i64, u16);
cast_impl!(i64, u8);
cast_impl!(i64, isize);
cast_impl!(i64, usize);

cast_impl!(i32, f64);
cast_impl!(i32, f32);
cast_impl!(i32, i64);
cast_impl!(i32, i32);
cast_impl!(i32, i16);
cast_impl!(i32, i8);
cast_impl!(i32, u64);
cast_impl!(i32, u32);
cast_impl!(i32, u16);
cast_impl!(i32, u8);
cast_impl!(i32, isize);
cast_impl!(i32, usize);

cast_impl!(i16, f64);
cast_impl!(i16, f32);
cast_impl!(i16, i64);
cast_impl!(i16, i32);
cast_impl!(i16, i16);
cast_impl!(i16, i8);
cast_impl!(i16, u64);
cast_impl!(i16, u32);
cast_impl!(i16, u16);
cast_impl!(i16, u8);
cast_impl!(i16, isize);
cast_impl!(i16, usize);

cast_impl!(i8, f64);
cast_impl!(i8, f32);
cast_impl!(i8, i64);
cast_impl!(i8, i32);
cast_impl!(i8, i16);
cast_impl!(i8, i8);
cast_impl!(i8, u64);
cast_impl!(i8, u32);
cast_impl!(i8, u16);
cast_impl!(i8, u8);
cast_impl!(i8, isize);
cast_impl!(i8, usize);

cast_impl!(u64, f64);
cast_impl!(u64, f32);
cast_impl!(u64, i64);
cast_impl!(u64, i32);
cast_impl!(u64, i16);
cast_impl!(u64, i8);
cast_impl!(u64, u64);
cast_impl!(u64, u32);
cast_impl!(u64, u16);
cast_impl!(u64, u8);
cast_impl!(u64, isize);
cast_impl!(u64, usize);

cast_impl!(u32, f64);
cast_impl!(u32, f32);
cast_impl!(u32, i64);
cast_impl!(u32, i32);
cast_impl!(u32, i16);
cast_impl!(u32, i8);
cast_impl!(u32, u64);
cast_impl!(u32, u32);
cast_impl!(u32, u16);
cast_impl!(u32, u8);
cast_impl!(u32, isize);
cast_impl!(u32, usize);

cast_impl!(u16, f64);
cast_impl!(u16, f32);
cast_impl!(u16, i64);
cast_impl!(u16, i32);
cast_impl!(u16, i16);
cast_impl!(u16, i8);
cast_impl!(u16, u64);
cast_impl!(u16, u32);
cast_impl!(u16, u16);
cast_impl!(u16, u8);
cast_impl!(u16, isize);
cast_impl!(u16, usize);

cast_impl!(u8, f64);
cast_impl!(u8, f32);
cast_impl!(u8, i64);
cast_impl!(u8, i32);
cast_impl!(u8, i16);
cast_impl!(u8, i8);
cast_impl!(u8, u64);
cast_impl!(u8, u32);
cast_impl!(u8, u16);
cast_impl!(u8, u8);
cast_impl!(u8, isize);
cast_impl!(u8, usize);

cast_impl!(usize, f64);
cast_impl!(usize, f32);
cast_impl!(usize, i64);
cast_impl!(usize, i32);
cast_impl!(usize, i16);
cast_impl!(usize, i8);
cast_impl!(usize, u64);
cast_impl!(usize, u32);
cast_impl!(usize, u16);
cast_impl!(usize, u8);
cast_impl!(usize, isize);
cast_impl!(usize, usize);

cast_impl!(isize, f64);
cast_impl!(isize, f32);
cast_impl!(isize, i64);
cast_impl!(isize, i32);
cast_impl!(isize, i16);
cast_impl!(isize, i8);
cast_impl!(isize, u64);
cast_impl!(isize, u32);
cast_impl!(isize, u16);
cast_impl!(isize, u8);
cast_impl!(isize, isize);
cast_impl!(isize, usize);
