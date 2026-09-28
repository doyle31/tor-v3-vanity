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

/// Propagate carries so **every** limb is < 2^51, folding the top overflow
/// back into limb 0 via 2^255 == 19 (mod p).
///
/// Tolerates input limbs up to ~2^64: each step is `r[i] >> 51`, mask, and
/// add the carry to the next limb, so a limb several orders of magnitude over
/// still resolves in a single pass.
///
/// The 19-fold lands in limb 0 *after* limb 0 has been masked, so limb 0 can
/// come out of a pass still carrying, and limbs above it can then each pick up
/// a carry of their own. `fe_reduce`'s subtraction needs every limb to be a
/// true radix digit, so the pass repeats until limb 0 settles. Each repeat
/// folds another multiple of p out of the value, so it terminates; in practice
/// the second pass is the last.
///
/// (A single pass leaves limb 0 up to 2^18 over and limb 2 at 2^51, which is
/// how `fe_invert(2)` used to come out as [0x1000, 0x1fff, 0x1fff, 0, 0].)
#[inline(always)]
fn carry(mut r: [u64; 5]) -> Fe {
    loop {
        let mut c;
        c = r[0] >> 51;
        r[0] &= MASK;
        r[1] += c;
        c = r[1] >> 51;
        r[1] &= MASK;
        r[2] += c;
        c = r[2] >> 51;
        r[2] &= MASK;
        r[3] += c;
        c = r[3] >> 51;
        r[3] &= MASK;
        r[4] += c;
        c = r[4] >> 51;
        r[4] &= MASK;
        // 2^255 == 19 (mod p). This is the only limb that can still be
        // carrying on exit, so it is the only thing the loop has to test.
        r[0] += c * 19;
        if r[0] < (1u64 << 51) {
            return r;
        }
    }
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

    // Subtract p as a borrow chain rather than comparing limb-by-limb.
    //
    // The borrow out of the top limb is 1 exactly when a < p, so the mask
    // selects a - p when a >= p and leaves a alone otherwise. Two things make
    // this the only correct shape:
    //
    //  * `a` must be a true digit vector first. It is not one: fe_mul's carry
    //    chain leaves limb 2 at 2^51, and a borrow chain over non-normalised
    //    digits computes the wrong difference (it is how fe_invert(2) came out
    //    as [0x1000, 0x1fff, 0x1fff, 0, 0] instead of 2). carry() guarantees
    //    that now.
    //  * A limb-by-limb `>=` scan cannot tell "already greater" from "still
    //    equal", so it resurrected `a >= p` whenever a lower limb happened to
    //    equal p's after a higher limb had already decided `a < p`. A carried
    //    limb lands on 2^51-1 often enough for that to be reachable, and it
    //    is how fe_invert(2), fe_invert(4), fe_invert(8), fe_invert(16),
    //    fe_invert(32) and fe_invert(38) all came out wrong. See
    //    `invert_of_small_values_is_exact`.
    //
    // The chain is also a little cheaper than the scan it replaces: two
    // overflowing_sub per limb instead of a compare plus an equality test.
    let n = carry(*a);
    let mut diff = [0u64; 5];
    let mut borrow = 0u64;
    for i in 0..5 {
        let (d, b1) = n[i].overflowing_sub(P[i]);
        let (d, b2) = d.overflowing_sub(borrow);
        diff[i] = d;
        borrow = (b1 | b2) as u64;
    }

    // borrow == 0 means n >= p, so keep the difference. n - p < 2^51 by
    // construction (n < 2^255 and p = 2^255 - 19), so no carry is needed after.
    let mask = 0u64.wrapping_sub(1 ^ borrow);
    let mut r = [0u64; 5];
    for i in 0..5 {
        r[i] = n[i] ^ (mask & (n[i] ^ diff[i]));
    }
    r
}

/// Field subtraction, as `(a - b) mod p` in the canonical representative.
///
/// # Why this no longer uses i128
///
/// The previous version added a 2p bias held as six *i128* limbs, which forced
/// a 128-bit add, subtract, shift and mask per limb. sm_86 has no 64-bit
/// integer ALU at all, so a 128-bit add is four SASS instructions and a
/// 128-bit shift is five: that formulation cost 171 SASS, more than 40% of a
/// field multiply, to compute a difference. The i128 was only ever there to
/// carry a possibly-negative digit; u64 carries the same information in the
/// sign bit for free.
///
/// # The two digit chains
///
/// `a - b` in radix 2^51 is a borrow chain. Each digit is
/// `a[i] - b[i] - borrow`, and with `a[i], b[i] < 2^51` that lies in
/// `(-2^51, 2^51)`. Viewed as a two's-complement u64 that range has a simple
/// signature: bit 51 is set exactly when the digit is negative, and masking off
/// the low 51 bits gives the right digit either way. So one shift yields both
/// the digit and the next borrow, in four SASS instructions, and nothing here
/// is wider than 64 bits.
///
/// The chain leaves `(a - b) mod 2^255` in the limbs, and the borrow out says
/// whether a 2^255 was handed back. `2^255 == 19 (mod p)`, so a set borrow owes
/// 19, and a second five-limb chain pays it. That chain can itself go negative,
/// but only when the value was below 19 -- so limbs 1..=4 came out of it as
/// MASK, and one more limb-local `- 19` completes the correction.
#[inline(always)]
pub fn fe_sub(a: &Fe, b: &Fe) -> Fe {
    // Result is the canonical representative in [0, p).
    let mut r = [0u64; 5];
    let mut borrow = 0u64;
    for i in 0..5 {
        let t = a[i].wrapping_sub(b[i]).wrapping_sub(borrow);
        borrow = (t >> 51) & 1;
        r[i] = t & MASK;
    }

    let t0 = r[0].wrapping_sub(borrow * 19);
    let mut under = (t0 >> 51) & 1;
    r[0] = t0 & MASK;
    for i in 1..5 {
        let t = r[i].wrapping_sub(under);
        under = (t >> 51) & 1;
        r[i] = t & MASK;
    }
    // `under == 1` only when the whole value was below 19, so limbs 1..=4 came
    // out of the loop as MASK and limb 0 owes 19 more to reach `r + p`.
    r[0] -= under * 19;

    // The exact-zero difference (a == b) lands on p, which is a valid residue
    // but not the canonical zero, so a bare carry() is not enough: fe_reduce
    // maps p back to 0.
    fe_reduce(&r)
}

/// Normalise the five 128-bit accumulators of a schoolbook product.
///
/// Each accumulator holds a column of five 51x51 products plus the 19-folded
/// wrap-around column, so it is under 2^109: its high half is under 2^45 and
/// the carry out of the 51-bit split is under 2^58. Both fit in u64, which is
/// the whole point. The previous chain re-derived every carry from the 128-bit
/// value, so each step was a 128-bit shift (five SASS) plus a 128-bit add (four),
/// and the final `c * 19` was a 128-bit *multiply*.
///
/// Splitting each accumulator into `lo`/`hi` u64 halves up front makes the
/// propagation entirely 64-bit: a `>> 51` of a value below 2^109 is
/// `(hi << 13) | (lo >> 51)`, and adding a carry into the next column is a
/// 64-bit add with an explicit overflow test. `fe_reduce` carries the rest.
#[inline(always)]
fn carry_product(r0: u128, r1: u128, r2: u128, r3: u128, r4: u128) -> Fe {
    let mut lo = [r0 as u64, r1 as u64, r2 as u64, r3 as u64, r4 as u64];
    let mut hi = [
        (r0 >> 64) as u64,
        (r1 >> 64) as u64,
        (r2 >> 64) as u64,
        (r3 >> 64) as u64,
        (r4 >> 64) as u64,
    ];

    // Low to high. `fold` is the carry out of the limb below, always under
    // 2^58, so it can overflow a 64-bit limb by at most one.
    let mut fold: u64 = 0;
    for i in 0..5 {
        let (s, ov) = lo[i].overflowing_add(fold);
        hi[i] += ov as u64;
        lo[i] = s;
        fold = (hi[i] << 13) | (s >> 51);
        lo[i] &= MASK;
    }

    // Limb 0 absorbed the top carry, scaled by 19 for the curve.
    let mut r = [0u64; 5];
    r[0] = lo[0].wrapping_add(fold * 19);
    r[1] = lo[1];
    r[2] = lo[2];
    r[3] = lo[3];
    r[4] = lo[4];
    fe_reduce(&r)
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

    let r0 = a[0] as u128 * b[0] as u128
        + a[1] as u128 * b19[4] as u128
        + a[2] as u128 * b19[3] as u128
        + a[3] as u128 * b19[2] as u128
        + a[4] as u128 * b19[1] as u128;
    let r1 = a[0] as u128 * b[1] as u128
        + a[1] as u128 * b[0] as u128
        + a[2] as u128 * b19[4] as u128
        + a[3] as u128 * b19[3] as u128
        + a[4] as u128 * b19[2] as u128;
    let r2 = a[0] as u128 * b[2] as u128
        + a[1] as u128 * b[1] as u128
        + a[2] as u128 * b[0] as u128
        + a[3] as u128 * b19[4] as u128
        + a[4] as u128 * b19[3] as u128;
    let r3 = a[0] as u128 * b[3] as u128
        + a[1] as u128 * b[2] as u128
        + a[2] as u128 * b[1] as u128
        + a[3] as u128 * b[0] as u128
        + a[4] as u128 * b19[4] as u128;
    let r4 = a[0] as u128 * b[4] as u128
        + a[1] as u128 * b[3] as u128
        + a[2] as u128 * b[2] as u128
        + a[3] as u128 * b[1] as u128
        + a[4] as u128 * b[0] as u128;

    carry_product(r0, r1, r2, r3, r4)
}

/// Squaring, via the squaring formula.
///
/// # 15 products instead of 25
///
/// With `a = sum a_i R^i` and `R = 2^51`, the off-diagonal products appear
/// twice, so the column sums are
///
/// ```text
/// c0 = a0^2                + 38 (a1 a4 + a2 a3)
/// c1 = 2 a0 a1             + 38 a2 a4 + 19 a3^2
/// c2 = 2 a0 a2 + a1^2      + 38 a3 a4
/// c3 = 2 a0 a3 + 2 a1 a2  + 19 a4^2
/// c4 = 2 a0 a4 + 2 a1 a3  + a2^2
/// ```
///
/// That is 5 squares plus 10 cross products against `fe_mul`'s 25, so ten of
/// the twenty-five 64x64 multiplies disappear, and each column sums three
/// terms instead of five, so the 128-bit accumulate count drops from 20 to 10.
///
/// `fe_sq` used to *be* `fe_mul(a, a)`. LLVM's CSE does recover the duplicate
/// off-diagonal products there, which is why the naive count only fell from 25
/// to 15 rather than staying at 25; what it cannot recover is the shorter
/// accumulation, and that is a third of the win. Measured on sm_86:
/// `fe_sq` goes 352 -> 264 SASS, of which the multiply-pipe part goes
/// 161 -> 124 IMAD.
///
/// The factors of 2 and of 19 are folded into a *multiplicand* rather than
/// applied to the 102-bit result: doubling a u128 costs four SASS, whereas
/// scaling a 51-bit limb by 2 or 19 costs one. `b2[i] = 2 a_i`,
/// `b19[i] = 19 a_i` and `b38[i] = 2 b19[i] = 38 a_i` cover every case with
/// ten multiplies and five shifts.
#[inline(always)]
pub fn fe_sq(a: &Fe) -> Fe {
    let b2 = [a[0] * 2, a[1] * 2, a[2] * 2, a[3] * 2, a[4] * 2];
    let b19 = [a[0] * 19, a[1] * 19, a[2] * 19, a[3] * 19, a[4] * 19];
    // 38 = 2 * 19, so this is a shift of b19, not a second multiply.
    let b38 = [b19[0] * 2, b19[1] * 2, b19[2] * 2, b19[3] * 2, b19[4] * 2];

    let r0 = a[0] as u128 * a[0] as u128
        + a[1] as u128 * b38[4] as u128
        + a[2] as u128 * b38[3] as u128;
    let r1 = a[0] as u128 * b2[1] as u128
        + a[2] as u128 * b38[4] as u128
        + a[3] as u128 * b19[3] as u128;
    let r2 = a[0] as u128 * b2[2] as u128
        + a[1] as u128 * a[1] as u128
        + a[3] as u128 * b38[4] as u128;
    let r3 = a[0] as u128 * b2[3] as u128
        + a[1] as u128 * b2[2] as u128
        + a[4] as u128 * b19[4] as u128;
    let r4 = a[0] as u128 * b2[4] as u128
        + a[1] as u128 * b2[3] as u128
        + a[2] as u128 * a[2] as u128;

    carry_product(r0, r1, r2, r3, r4)
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

/// The low 64 bits of a field element, packed into one `u64`.
///
/// # The input must be canonical
///
/// This reads limbs, not bits of a residue, so `a` has to be the canonical
/// representative in `[0, p)` -- which every caller has for free, because
/// `fe_mul` ends in `fe_reduce` (`mul_output_is_already_canonical` pins that
/// down). A non-canonical vector is a different integer from its canonical
/// form: `[MASK; 5]` is `2^255 - 1` as limbs but the value `18`, and its
/// encoding is `[18, 0, ...]`, not `[0xFF, 0xFF, ...]`.
///
/// # Which 64 bits a prefix test can see
///
/// The compressed public key is the *little-endian* byte encoding of `y`, but a
/// Tor v3 address is `base32(pubkey || checksum || 0x03)` in RFC 4648 order,
/// and RFC 4648 reads a byte string most-significant-bit first. So the
/// address's **first** character is the top 5 bits of `pubkey[0]` -- the top 5
/// bits of the *least significant* byte of `y`. A prefix of `L` characters
/// therefore constrains bits of `y` counted from the **bottom**, and for
/// `L <= 12` (5L <= 60 bits) the constrained bits never pass bit 63. (The sign
/// bit, which is output bit 255, is out of reach of every prefix up to 51
/// characters, so it cannot influence a match.)
///
/// Limb 0 holds bits 0..=50 and limb 1 bits 51..=101, so the whole observable
/// window is two limbs and one OR:
///
/// ```text
///     word = limb0 | (limb1 << 51)
/// ```
///
/// # Why this is not the accumulator `fe_to_bytes` abandoned
///
/// `fe_to_bytes` refuses to pack limbs into a `u64` because a *51-bit limb*
/// shifted left by a *variable* offset (up to 51) overflows `u64` once the
/// offset exceeds 13, silently dropping the limb's high bits. That hazard is
/// specific to the variable offset. Here the shift is the constant 51, so the
/// only bits that can leave the `u64` are bits 101..=127 of the 102-bit limb 1,
/// which do not exist; the truncation keeps exactly bits 51..=63, which is what
/// is wanted. `matches_word_is_equivalent_to_matches` in the crate root checks
/// the result against the byte-wise walk for every prefix length.
#[inline(always)]
pub fn fe_prefix_word(a: &Fe) -> u64 {
    a[0] | (a[1] << 51)
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

    /// `fe_invert` must be exact for *every* input, not just the ones with
    /// comfortable limb values. This is the test that caught the `fe_reduce`
    /// comparison bug: the 265-step addition chain visits intermediate values
    /// whose carried limbs land exactly on 2^51-1, which the old limb-by-limb
    /// `>=` scan mis-handled. Small inputs are the cheapest way in, because
    /// their chains stay in a narrow range for long.
    #[test]
    fn invert_of_small_values_is_exact() {
        for v in 1u64..=64 {
            let a: Fe = [v, 0, 0, 0, 0];
            let ai = fe_invert(&a);
            assert_eq!(fe_mul(&a, &ai), FE_ONE, "{}^-1 * {} must be 1", v, v);
            // And the double inverse, which also pins the limb vector.
            assert_eq!(fe_invert(&ai), a, "{}^-1 must invert back", v);
        }
    }

    /// The same trap for the shapes the kernel actually feeds the inversion:
    /// points with a maximal top limb, and values just below p.
    #[test]
    fn invert_handles_awkward_limbs() {
        let mut rng: u64 = 0x243F_6A88_85A3_08D3;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for i in 0..2000 {
            // Half the cases pin limb 4 to 2^51-1, the value the old
            // comparison keyed on.
            let a: Fe = if i % 2 == 0 {
                [next() & MASK, next() & MASK, next() & MASK, next() & MASK, MASK]
            } else {
                [next() & MASK, next() & MASK, next() & MASK, next() & MASK, next() & MASK]
            };
            let ai = fe_invert(&a);
            assert_eq!(fe_mul(&a, &ai), FE_ONE, "case {}", i);
        }
    }

    /// A spread of operand shapes chosen to break arithmetic that happens to
    /// work on uniform random inputs.
    ///
    /// The important members are the ones with many limbs equal to `MASK`
    /// (2^51 - 1). `fe_reduce`'s "is the value at least p" test is a comparison
    /// against p's digits, and p's top four digits *are* `MASK`, so an operand
    /// full of `MASK` limbs is exactly the input that separates "every top limb
    /// equals MASK" (>= p) from "some top limb equals MASK" (< p). A uniform
    /// random sweep essentially never produces one, which is how a wrong
    /// comparison can sit there passing the RFC 8032 vectors.
    fn adversarial() -> Vec<Fe> {
        let mut v: Vec<Fe> = vec![
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [3, 5, 7, 11, 13],
            [MASK, MASK, MASK, MASK, MASK],
            [MASK, MASK, MASK, MASK, 0],
            [0, MASK, MASK, MASK, MASK],
            [MASK, 0, MASK, 0, MASK],
            [0, 0, 0, 0, 18],
            [18, 0, 0, 0, 0],
            [19, 0, 0, 0, 0],
            [20, 0, 0, 0, 0],
            [MASK - 18, MASK, MASK, MASK, MASK], // p - 1
            [MASK - 19, MASK, MASK, MASK, MASK], // p
            [MASK - 20, MASK, MASK, MASK, MASK], // p + 1
        ];
        let mut s: u64 = 0x243F_6A88_85A3_08D3;
        let mut bit = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s & MASK
        };
        for _ in 0..256 {
            // Two thirds of the limbs pinned to MASK.
            v.push([
                if bit() % 3 == 0 { bit() } else { MASK },
                if bit() % 3 == 0 { bit() } else { MASK },
                if bit() % 3 == 0 { bit() } else { MASK },
                if bit() % 3 == 0 { bit() } else { MASK },
                if bit() % 3 == 0 { bit() } else { MASK },
            ]);
        }
        v
    }

    /// `c * (a + b) == c * a + c * b`, over the adversarial operand shapes.
    ///
    /// This is the identity that pins down `fe_reduce`. Every operation feeds
    /// the next, so a reduction that is wrong on inputs with many maximal limbs
    /// makes the two sides disagree even though each side is internally
    /// consistent -- the failure mode that `fe_sq` vs `fe_mul` and the RFC 8032
    /// vectors both miss, because they run `fe_reduce` on identical values.
    #[test]
    fn mul_distributes_over_add() {
        let cases = adversarial();
        for a in cases.iter() {
            for b in cases.iter() {
                for c in cases.iter() {
                    let lhs = fe_mul(&fe_add(a, b), c);
                    let rhs = fe_add(&fe_mul(a, c), &fe_mul(b, c));
                    assert_eq!(lhs, rhs, "c*(a+b) for {:?} {:?} {:?}", a, b, c);
                }
            }
        }
    }

    /// `fe_sq` is no longer `fe_mul(a, a)`, so pin it against `fe_mul` over both
    /// the fixed edge cases and a pseudo-random sweep. A wrong squaring formula
    /// has a habit of being invisible in the RFC 8032 vectors.
    #[test]
    fn sq_is_mul_of_self() {
        for a in adversarial().iter() {
            assert_eq!(fe_sq(a), fe_mul(a, a), "fe_sq(a) must equal fe_mul(a, a)");
        }
    }

    /// `a - b == a + (0 - b)`, over the same adversarial shapes.
    ///
    /// The `- 19` that the second digit chain owes only fires when the borrow
    /// out of `a - b` is set, i.e. when `a < b`; and it can itself go negative
    /// only when the result is below 19. The adversarial list contains `18`,
    /// `19`, `20` and `MASK` limbs precisely to reach each of those branches.
    #[test]
    fn sub_is_add_of_negation() {
        let cases = adversarial();
        for a in cases.iter() {
            for b in cases.iter() {
                assert_eq!(fe_sub(a, b), fe_add(a, &fe_sub(&FE_ZERO, b)));
                for limb in fe_sub(a, b).iter() {
                    assert!(*limb <= MASK, "fe_sub left limb {} over 2^51", limb);
                }
            }
        }
    }

    #[test]
    fn bytes_roundtrip() {
        let a: Fe = [3, 5, 7, 11, 13];
        let b = fe_to_bytes(&a);
        let c = fe_from_bytes(&b);
        assert_eq!(a, c);
    }

    /// `fe_mul` must return the **canonical** representative in [0, p), not
    /// merely a limb vector congruent to the product.
    ///
    /// `ge_prefix_word` reads the low bits of `fe_mul`'s output directly and
    /// skips the `fe_reduce(&carry())` that `fe_to_bytes` performs, because
    /// `carry_product` already ends in `fe_reduce`. That is only sound while
    /// the multiply keeps reducing, and nothing else in the tree would notice
    /// if it stopped: a product left at, say, `2^255 - 1` still multiplies and
    /// inverts correctly, and `fe_to_bytes` would silently canonicalise it. So
    /// the invariant is pinned here, over the same adversarial operand shapes
    /// the reduction tests use.
    ///
    /// Re-reducing a canonical value is the identity, so `fe_reduce(carry(x)) ==
    /// x` is exactly the claim being made: `x` is already the output of
    /// `fe_reduce`, hence a proper digit vector standing for a value in [0, p).
    #[test]
    fn mul_output_is_already_canonical() {
        let cases = adversarial();
        for a in cases.iter() {
            for b in cases.iter() {
                let p = fe_mul(a, b);
                for (i, limb) in p.iter().enumerate() {
                    assert!(*limb <= MASK, "product limb {} over 2^51: {:?}", i, p);
                }
                assert_eq!(fe_reduce(&carry(p)), p, "product is not canonical");
            }
        }
    }

    /// `fe_prefix_word` must be exactly the low 8 bytes of `fe_to_bytes`, read
    /// little-endian. This is the field-level half of the equivalence the
    /// crate-root test `matches_word_is_equivalent_to_matches` needs; it is
    /// checked here against the serial bit walk rather than against itself.
    ///
    /// The input is canonicalised first, because that is the contract:
    /// `fe_prefix_word` reads limbs, not bits of a residue, and a non-canonical
    /// limb vector (`[MASK; 5]`, say, which is `2^255 - 1`) is a different
    /// integer from its canonical representative `18` -- reading it directly
    /// would give the encoding of the *limbs*, not of the value. Every caller
    /// gets a canonical element for free from `fe_mul`, which is what
    /// `mul_output_is_already_canonical` pins down.
    #[test]
    fn prefix_word_is_the_low_bytes_of_the_encoding() {
        let mut cases = adversarial();
        // Plus a sweep, because the adversarial shapes are limb-extremal and
        // `fe_to_bytes`' walk has data-dependent byte boundaries.
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s & MASK
        };
        for _ in 0..2000 {
            cases.push([next(), next(), next(), next(), next()]);
        }
        for a in cases.iter() {
            let bytes = fe_to_bytes(a);
            let mut want = 0u64;
            for (i, b) in bytes[..8].iter().enumerate() {
                want |= (*b as u64) << (8 * i);
            }
            let canonical = fe_reduce(&carry(*a));
            assert_eq!(
                fe_prefix_word(&canonical),
                want,
                "low word of {:?}",
                bytes
            );
            // And the canonical form is a fixed point, so the two agree on it.
            assert_eq!(fe_reduce(&carry(canonical)), canonical);
        }
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
