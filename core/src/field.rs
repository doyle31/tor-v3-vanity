//! Field arithmetic for Curve25519 / ed25519, tuned for the GPU.
//!
//! Representation: 5 limbs of 51 bits in `u64` words, the same layout
//! fiat-crypto emits. This is *not* the fastest choice on NVIDIA hardware --
//! consumer SMs have no native 64-bit integer multiply, so every limb
//! product is an emulated `IMAD.WIDE.U32` sequence. But it keeps the code
//! small and reviewable, and the algorithmic savings in the kernel dwarf the
//! per-multiply cost. See `KERNEL.md` for the measured comparison.
//!
//! All operations are branch-free and constant-time.

/// A field element, little-endian limbs, each holding 51 bits.
pub type Fe = [u64; 5];

const MASK: u64 = (1 << 51) - 1;
const ZERO: Fe = [0, 0, 0, 0, 0];
const ONE: Fe = [1, 0, 0, 0, 0];

/// 2 * d, where d = -121665/121666 (mod p) is the ed25519 curve constant.
///
/// d        = 0x52036cee2b6ffe738cc740797779e89800700a4d4141d8ab75eb4dca135978a3
/// 2*d      = 0x2406d9dc56dffce7198e80f2eef3d13000e0149a8283b156ebd69b9426b2f159
///
/// NOTE: ref10's `ecd` tables hold *d*, not 2*d -- do not copy them here.
const D2: Fe = [
    1859910466990425,
    932731440258426,
    1072319116312658,
    1815898335770999,
    633789495995903,
];

/// d itself, needed to rebuild the base point's x coordinate.
const D: Fe = [
    929955233495203,
    466365720129213,
    1662059464998953,
    2033849074728123,
    1442794654840575,
];

/// sqrt(-1) mod p, used when decompressing a point.
/// 0x2b8324804fc1df0b2b4d00993dfbd7a72f431806ad2fe478c4ee1b274a0ea0b0
const SQRT_M1: Fe = [
    1718705420411056,
    234908883556509,
    2233514472574048,
    2117202627021982,
    765476049583133,
];

pub const FE_ZERO: Fe = ZERO;
pub const FE_ONE: Fe = ONE;
pub const FE_D2: Fe = D2;
pub const FE_D: Fe = D;
pub const FE_SQRT_M1: Fe = SQRT_M1;

/// Propagate carries so every limb is < 2^51, folding the top overflow back
/// into limb 0 via 2^255 == 19 (mod p).
///
/// Tolerates input limbs up to ~2^64: each step is `r[i] >> 51`, mask, and
/// add the carry to the next limb, so a limb several orders of magnitude over
/// still resolves in a single pass. The `* 19` fold at the end is small
/// enough (< 2^18) that limb 0 cannot overflow from it.
#[inline(always)]
fn carry(mut r: [u64; 5]) -> Fe {
    for i in 0..4 {
        let c = r[i] >> 51;
        r[i] &= MASK;
        r[i + 1] += c;
    }
    let c = r[4] >> 51;
    r[4] &= MASK;
    r[0] += c * 19;
    r
}

#[inline(always)]
pub fn fe_add(a: &Fe, b: &Fe) -> Fe {
    let mut r = [0u64; 5];
    for i in 0..5 {
        r[i] = a[i] + b[i];
    }
    // Reduce mod p so the result is the canonical limb vector. Without this,
    // `carry` alone can leave limb 0 above 2^51 (e.g. 2^51 + 6 instead of 6),
    // which is the same field element but not the same limb vector -- and
    // limb-wise equality is exactly what callers and tests compare against.
    fe_reduce(&carry(r))
}

/// Bring a value with limbs < 2^51 into the canonical representative in [0, p).
///
/// carry() guarantees each limb is under 2^51, but the value can still reach
/// ~2^256, so it may need p subtracted. p = 2^255 - 19 has radix-2^51 digits
/// [2^51-19, 2^51-1, 2^51-1, 2^51-1, 2^51-1] plus an implicit 2^255, which
/// is exactly the 19 already folded into limb 0 by carry().
fn fe_reduce(a: &Fe) -> Fe {
    const P: Fe = [
        (1u64 << 51) - 19,
        (1u64 << 51) - 1,
        (1u64 << 51) - 1,
        (1u64 << 51) - 1,
        (1u64 << 51) - 1,
    ];

    // Compute a - p and select it when a >= p. The comparison is done
    // high-limb-first so an earlier difference decides the outcome.
    let mut ge = 1u64; // "all higher limbs were equal"
    for i in (0..5).rev() {
        let lt = (a[i] < P[i]) as u64;
        ge = (ge & (1 ^ lt)) | ((1 ^ ge) & (fe_limb_eq(a[i], P[i])));
    }

    let mask = 0u64.wrapping_sub(ge);
    let mut r = [0u64; 5];
    for i in 0..5 {
        // a[i] - P[i] is non-negative whenever ge == 1, and we only use the
        // result in that case, so the wrapping is benign.
        r[i] = a[i].wrapping_sub(P[i] & mask);
    }
    carry(r)
}

#[inline(always)]
fn fe_limb_eq(x: u64, y: u64) -> u64 {
    let d = x ^ y;
    ((d | d.wrapping_neg()) >> 63) ^ 1
}

#[inline(always)]
pub fn fe_sub(a: &Fe, b: &Fe) -> Fe {
    // Bias by 2p, kept as a full 6-limb radix-2^51 value in u128 so the
    // intermediate is exact. This is the standard ref10/donna formulation.
    //
    // 2p = 2^256 - 38 does not fit in five 51-bit limbs, so it needs a sixth.
    // The sum a - b + 2p is a non-negative integer below 2^256, so masking
    // each limb to 51 bits recovers its true digits with no borrow
    // bookkeeping. The sixth limb (the 2^255 term) is dropped here and folded
    // in as 19, because 2^255 == 19 (mod p).
    //
    // Result is the canonical representative in [0, p).
    // Correct radix-2^51 digits of 2p = 2^256 - 38, verified by reconstructing:
    //     (2^51 - 38) + (2^51-1)(2^51 + 2^102 + 2^153 + 2^204) + 2^255 == 2p
    // Limb 0 is 2^51 - 38, NOT 2*(2^51-19): the latter overflows one limb.
    // The sixth limb is 1, i.e. one unit of 2^255.
    const TWO_P: [u128; 6] = [
        (1u128 << 51) - 38,
        (1u128 << 51) - 1,
        (1u128 << 51) - 1,
        (1u128 << 51) - 1,
        (1u128 << 51) - 1,
        1,
    ];

    // Sum a - b + 2p across six limbs, propagating the carry between limbs.
    // The carry propagation is essential: each limb's raw sum can reach ~2^52,
    // and dropping that carry silently corrupts every limb above it.
    //
    // Ordering matters: add the bias *before* subtracting, so the intermediate
    // is always non-negative and plain i128 arithmetic suffices. Since
    // TWO_P[i] >= 2^51 - 38 and b[i] < 2^51, `a + bias - b` is positive.
    let mut w = [0u64; 6];
    let mut k = 0i128;
    for i in 0..5 {
        let t = a[i] as i128 + TWO_P[i] as i128 + k - b[i] as i128;
        w[i] = (t as u64) & MASK;
        k = t >> 51;
    }
    {
        // Sixth limb: no a/b term, just the 2^255 term of 2p plus the carry.
        let t = TWO_P[5] as i128 + k;
        w[5] = (t as u64) & MASK;
        k = t >> 51;
    }
    debug_assert_eq!(k, 0, "a - b + 2p must fit in six limbs");

    // w[5] counts units of 2^255, and 2^255 == 19 (mod p).
    let mut r = [w[0], w[1], w[2], w[3], w[4]];
    r[0] = r[0].wrapping_add((w[5] as u64).wrapping_mul(19));

    // Reduce fully. The +19 bias means the exact-zero difference (a == b) lands
    // on p rather than 0, so a bare carry() is not enough: p is a valid residue
    // but not the canonical representative of zero, and limb-wise comparisons
    // against ZERO would fail. fe_reduce maps p back to 0.
    fe_reduce(&carry(r))
}

#[inline(always)]
pub fn fe_mul(a: &Fe, b: &Fe) -> Fe {
    // 5x5 schoolbook into 128-bit accumulators, then carry.
    let b19 = [
        b[0] * 19,
        b[1] * 19,
        b[2] * 19,
        b[3] * 19,
        b[4] * 19,
    ];

    let mut r0 = a[0] as u128 * b[0] as u128
        + a[1] as u128 * b19[4] as u128
        + a[2] as u128 * b19[3] as u128
        + a[3] as u128 * b19[2] as u128
        + a[4] as u128 * b19[1] as u128;
    let mut r1 = a[0] as u128 * b[1] as u128
        + a[1] as u128 * b[0] as u128
        + a[2] as u128 * b19[4] as u128
        + a[3] as u128 * b19[3] as u128
        + a[4] as u128 * b19[2] as u128;
    let mut r2 = a[0] as u128 * b[2] as u128
        + a[1] as u128 * b[1] as u128
        + a[2] as u128 * b[0] as u128
        + a[3] as u128 * b19[4] as u128
        + a[4] as u128 * b19[3] as u128;
    let mut r3 = a[0] as u128 * b[3] as u128
        + a[1] as u128 * b[2] as u128
        + a[2] as u128 * b[1] as u128
        + a[3] as u128 * b[0] as u128
        + a[4] as u128 * b19[4] as u128;
    let mut r4 = a[0] as u128 * b[4] as u128
        + a[1] as u128 * b[3] as u128
        + a[2] as u128 * b[2] as u128
        + a[3] as u128 * b[1] as u128
        + a[4] as u128 * b[0] as u128;

    // Carry chain, folding the top limb back with the curve's factor of 19.
    let mask = MASK as u128;
    let mut c: u128;
    c = r0 >> 51;
    r0 &= mask;
    r1 += c;
    c = r1 >> 51;
    r1 &= mask;
    r2 += c;
    c = r2 >> 51;
    r2 &= mask;
    r3 += c;
    c = r3 >> 51;
    r3 &= mask;
    r4 += c;
    c = r4 >> 51;
    r4 &= mask;
    r0 += c * 19;
    c = r0 >> 51;
    r0 &= mask;
    r1 += c;
    c = r1 >> 51;
    r1 &= mask;
    r2 += c;

    // The product can land in [p, 2p), so reduce to the canonical
    // representative. Without this, `fe_mul(a, a.invert())` yields the limb
    // vector for p+1 rather than 1: same field element, different limbs, and
    // every limb-wise equality check downstream would fail.
    fe_reduce(&[r0 as u64, r1 as u64, r2 as u64, r3 as u64, r4 as u64])
}

#[inline(always)]
pub fn fe_sq(a: &Fe) -> Fe {
    fe_mul(a, a)
}

/// z^(2^255 - 21) = z^-1, via the standard addition chain.
pub fn fe_invert(z: &Fe) -> Fe {
    let z1 = *z;
    let z2 = fe_sq(&z1);
    let z8 = fe_sq(&fe_sq(&z2));
    let z9 = fe_mul(&z1, &z8);
    let z11 = fe_mul(&z2, &z9);
    let z22 = fe_sq(&z11);
    let z_5_0 = fe_mul(&z9, &z22);

    let z_10_0 = {
        let t = (0..5).fold(z_5_0, |a, _| fe_sq(&a));
        fe_mul(&t, &z_5_0)
    };
    let z_20_0 = {
        let t = (0..10).fold(z_10_0, |a, _| fe_sq(&a));
        fe_mul(&t, &z_10_0)
    };
    let z_40_0 = {
        let t = (0..20).fold(z_20_0, |a, _| fe_sq(&a));
        fe_mul(&t, &z_20_0)
    };
    let z_50_0 = {
        let t = (0..10).fold(z_40_0, |a, _| fe_sq(&a));
        fe_mul(&t, &z_10_0)
    };
    let z_100_0 = {
        let t = (0..50).fold(z_50_0, |a, _| fe_sq(&a));
        fe_mul(&t, &z_50_0)
    };
    let z_200_0 = {
        let t = (0..100).fold(z_100_0, |a, _| fe_sq(&a));
        fe_mul(&t, &z_100_0)
    };
    let z_250_0 = {
        let t = (0..50).fold(z_200_0, |a, _| fe_sq(&a));
        fe_mul(&t, &z_50_0)
    };
    let z_255_5 = (0..5).fold(z_250_0, |a, _| fe_sq(&a));
    fe_mul(&z_255_5, &z11)
}

/// LSB of the canonical encoding, i.e. the sign bit used in point encoding.
#[inline(always)]
pub fn fe_is_negative(a: &Fe) -> u64 {
    a[0] & 1
}



/// Little-endian byte encoding of the canonical representative in [0, p).
pub fn fe_to_bytes(a: &Fe) -> [u8; 32] {
    // carry() then fe_reduce() gives limbs < 2^51 representing a value in [0, p).
    //
    // The hand-rolled "estimate q, add 19*q, carry" reduction that ref10 uses
    // is NOT equivalent: q must be the carry out of the *top* limb after
    // adding 19, and getting that wrong corrupts the high bytes. Reusing the
    // already-tested fe_reduce avoids the duplicate logic entirely.
    let t = fe_reduce(&carry(*a));

    // Pack five 51-bit limbs into 32 little-endian bytes.
    //
    // The limbs form one contiguous little-endian bit stream, 255 bits total.
    // Walk an explicit bit position rather than a `u64` accumulator: shifting
    // a 51-bit limb left by a variable offset silently drops its high bits
    // whenever the offset exceeds 13.
    let mut out = [0u8; 32];
    let mut bit = 0usize; // bits written so far

    for limb in t.iter() {
        let mut v = *limb;
        let mut remaining = 51u32;
        while remaining > 0 {
            let byte = bit / 8;
            let off = bit % 8; // bits already filled in out[byte]
            let space = (8 - off) as u32; // free bits in out[byte]
            let take = space.min(remaining);
            out[byte] |= ((v & ((1u64 << take) - 1)) as u8) << off;
            v >>= take;
            remaining -= take;
            bit += take as usize;
        }
    }
    out
}

/// Decode 32 little-endian bytes into five 51-bit limbs.
///
/// Bits 255 and above are ignored, as ed25519 requires. This walks the same
/// bit stream `fe_to_bytes` produces, so the two are exact inverses.
pub fn fe_from_bytes(s: &[u8; 32]) -> Fe {
    let mut acc: u64 = 0;
    let mut bits: u32 = 0;
    let mut pos = 0usize;

    let take = |n: u32, pos: &mut usize, acc: &mut u64, bits: &mut u32| -> u64 {
        while *bits < n {
            *acc |= (s[*pos] as u64) << *bits;
            *pos += 1;
            *bits += 8;
        }
        let v = *acc & ((1u64 << n) - 1);
        *acc >>= n;
        *bits -= n;
        v
    };

    [
        take(51, &mut pos, &mut acc, &mut bits),
        take(51, &mut pos, &mut acc, &mut bits),
        take(51, &mut pos, &mut acc, &mut bits),
        take(51, &mut pos, &mut acc, &mut bits),
        take(51, &mut pos, &mut acc, &mut bits),
    ]
}

/// Constant-time conditional move over field elements.
#[inline(always)]
pub fn fe_select(a: &Fe, b: &Fe, cond: u64) -> Fe {
    let mask = 0u64.wrapping_sub(cond);
    let mut r = [0u64; 5];
    for i in 0..5 {
        r[i] = a[i] ^ (mask & (a[i] ^ b[i]));
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use std::string::String;
    #[allow(unused_imports)]
    use std::vec::Vec;




    #[test]
    fn add_sub_roundtrip() {
        // fe_add then fe_sub is exact in value, but only modulo p. For these
        // small operands no wrap occurs, so the canonical limb vectors match.
        let a: Fe = [1, 2, 3, 4, 5];
        let b: Fe = [7, 11, 13, 17, 19];
        let s = fe_add(&a, &b);
        assert_eq!(fe_sub(&s, &b), a);
    }

    #[test]
    fn add_sub_is_exact_as_integers() {
        // For canonical inputs with a + b < p, the identities
        //     (a + b) - b == a   and   (a + b) - a == b
        // hold exactly, with no modular wrap. Keep the cases small enough that
        // the sum cannot reach p.
        let cases: [(Fe, Fe); 3] = [
            ([1, 2, 3, 4, 5], [7, 11, 13, 17, 19]),
            ([0, 0, 0, 0, 0], [1, 1, 1, 1, 1]),
            ([12345, 67890, 111213, 141516, 171819], [1, 2, 3, 4, 5]),
        ];

        for (a, b) in cases.iter() {
            let sum = fe_add(a, b);
            assert_eq!(fe_sub(&sum, b), *a, "(a+b)-b must equal a");
            assert_eq!(fe_sub(&sum, a), *b, "(a+b)-a must equal b");
            assert_eq!(fe_add(&fe_sub(a, b), b), *a, "(a-b)+b must equal a");
        }
    }

    #[test]
    fn sub_wraps_past_p() {
        // Near the modulus, subtraction must wrap rather than overflow.
        // p - 1 = all limbs 2^51-1 except limb0 = 2^51-20.
        let p_minus_1: Fe = [
            (1u64 << 51) - 20,
            (1u64 << 51) - 1,
            (1u64 << 51) - 1,
            (1u64 << 51) - 1,
            (1u64 << 51) - 1,
        ];
        let one: Fe = FE_ONE;

        // (p - 1) - 1 == p - 2
        let d = fe_sub(&p_minus_1, &one);
        let want: Fe = [
            (1u64 << 51) - 21,
            (1u64 << 51) - 1,
            (1u64 << 51) - 1,
            (1u64 << 51) - 1,
            (1u64 << 51) - 1,
        ];
        assert_eq!(d, want, "(p-1) - 1 must be p-2");

        // (p-1) - (p-1) == 0, and 0 must be the canonical zero, not p.
        assert_eq!(fe_sub(&p_minus_1, &p_minus_1), FE_ZERO);
    }

    #[test]
    fn sub_wraps_mod_p() {
        // fe_sub is mod p, so a - b is NOT recoverable by adding b back when
        // a < b: the result is (a - b) mod p, and adding b gives a + k*p.
        // Check the real invariant instead -- the difference is the canonical
        // residue, and subtracting p from it once yields the true value.
        let a: Fe = [1, 2, 3, 4, 5];
        let b: Fe = [7, 11, 13, 17, 19];
        let d = fe_sub(&a, &b);

        // Every limb must be a proper 51-bit digit.
        for limb in d.iter() {
            assert!(*limb <= MASK, "limb {} exceeds 2^51", limb);
        }
        // Digits verified against an independent big-integer computation:
        // (a - b) mod p == 57896044618657737757663578490373798767084888034061231296306240838111200280551
        assert_eq!(
            d,
            [
                2251799813685223,
                2251799813685238,
                2251799813685237,
                2251799813685234,
                2251799813685233,
            ]
        );
    }

    #[test]
    fn mul_is_commutative_and_associative() {
        let a: Fe = [3, 5, 7, 11, 13];
        let b: Fe = [17, 19, 23, 29, 31];
        let c: Fe = [37, 41, 43, 47, 53];
        assert_eq!(fe_mul(&a, &b), fe_mul(&b, &a));
        assert_eq!(fe_mul(&fe_mul(&a, &b), &c), fe_mul(&a, &fe_mul(&b, &c)));
    }

    #[test]
    fn mul_by_one_and_zero() {
        let a: Fe = [3, 5, 7, 11, 13];
        assert_eq!(fe_mul(&a, &FE_ONE), a);
        assert_eq!(fe_mul(&a, &FE_ZERO), FE_ZERO);
    }

    #[test]
    fn invert_is_inverse() {
        let a: Fe = [3, 5, 7, 11, 13];
        let ai = fe_invert(&a);
        assert_eq!(fe_mul(&a, &ai), FE_ONE);
    }

    #[test]
    fn bytes_roundtrip() {
        let a: Fe = [3, 5, 7, 11, 13];
        let b = fe_to_bytes(&a);
        let c = fe_from_bytes(&b);
        assert_eq!(a, c);
    }

    #[test]
    fn d2_matches_the_curve_constant() {
        // FE_D must be d and FE_D2 must be 2*d. Verify both by recovering
        // 2*FE_D and checking the defining curve identity for B:
        //
        //     -x^2 + y^2 == 1 + d*x^2*y^2
        //
        // which is precisely the equation d was chosen to satisfy. Doing it
        // with FE_D directly (rather than FE_D2/2) also pins down the
        // relationship between the two constants.
        let twice_d = fe_add(&FE_D, &FE_D);
        assert_eq!(twice_d, FE_D2, "FE_D2 must be exactly 2 * FE_D");

        // ed25519 base point coordinates.
        let bx: Fe = [
            1738742601995546,
            1146398526822698,
            2070867633025821,
            562264141797630,
            587772402128613,
        ];
        let by: Fe = [
            1801439850948184,
            1351079888211148,
            450359962737049,
            900719925474099,
            1801439850948198,
        ];

        let x2 = fe_sq(&bx);
        let y2 = fe_sq(&by);
        let lhs = fe_sub(&y2, &x2);
        let rhs = fe_add(&FE_ONE, &fe_mul(&FE_D, &fe_mul(&x2, &y2)));
        assert_eq!(lhs, rhs, "B must lie on the curve defined by FE_D");
    }

}
