//! Edwards curve group operations and the batched base-point walk.
//!
//! The central idea, borrowed from mkp224o's `worker_batch.inc.h`: once you
//! have one scalar `s` and its point `A = [s]B`, the sequence `[s]B`,
//! `[s+8]B`, `[s+16]B`, ... is a contiguous walk up the group, so each
//! additional candidate costs a *single* point addition rather than a fresh
//! scalar multiplication.
//!
//! Tor's `hs_ed25519_secret_key` stores the 64-byte *expanded* key, whose
//! first 32 bytes are the clamped scalar. Those may be advanced directly,
//! since the pubkey is derived from the scalar and the clamping invariants
//! (`sk[0] & 248 == sk[0]`, `(sk[31] & 63) | 64 == sk[31]`) survive a small
//! increment. mkp224o does exactly this at `worker_batch.inc.h:92`.

use super::field::*;
#[cfg(test)]
use crate::sha512;

/// A point in extended twisted Edwards coordinates (X:Y:Z:T), a = -1.
#[derive(Copy, Clone)]
pub struct Ge {
    pub x: Fe,
    pub y: Fe,
    pub z: Fe,
    pub t: Fe,
}

/// The neutral element, and the base point B = (x, 4/5) with Z = T = 1.
pub fn ge_identity() -> Ge {
    Ge {
        x: FE_ZERO,
        y: FE_ONE,
        z: FE_ONE,
        t: FE_ZERO,
    }
}

pub fn ge_base() -> Ge {
    // x = 15112221349535400772501151409588531511454012693041857206046113283949847762202
    //     0x216936d3cd6e53fec0a4e231fdd6dc5c692cc7609525a7b2c9562d608f25d51a
    let bx: Fe = [
        1738742601995546,
        1146398526822698,
        2070867633025821,
        562264141797630,
        587772402128613,
    ];
    // y = 4/5 mod p
    //     0x6666666666666666666666666666666666666666666666666666666666666658
    let by: Fe = [
        1801439850948184,
        1351079888211148,
        450359962737049,
        900719925474099,
        1801439850948198,
    ];
    let t = fe_mul(&bx, &by);
    Ge {
        x: bx,
        y: by,
        z: FE_ONE,
        t,
    }
}

/// 8B, in cached (Niels-like) form, precomputed for the walk.
pub fn ge_eight() -> Ge {
    let b = ge_base();
    let mut p = b;
    for _ in 0..3 {
        p = ge_double(&p);
    }
    p
}

/// Unified addition, valid for all inputs including the identity.
/// 9M + 1D.
#[inline(always)]
pub fn ge_add(p: &Ge, q: &Ge) -> Ge {
    let a = fe_mul(&fe_sub(&p.y, &p.x), &fe_sub(&q.y, &q.x));
    let b = fe_mul(&fe_add(&p.y, &p.x), &fe_add(&q.y, &q.x));
    let c = fe_mul(&fe_mul(&p.t, &q.t), &FE_D2);
    let d = fe_mul(&fe_add(&p.z, &p.z), &q.z);
    let e = fe_sub(&b, &a);
    let f = fe_sub(&d, &c);
    let g = fe_add(&d, &c);
    let h = fe_add(&b, &a);
    Ge {
        x: fe_mul(&e, &f),
        y: fe_mul(&g, &h),
        z: fe_mul(&f, &g),
        t: fe_mul(&e, &h),
    }
}

/// Point doubling for a = -1, 4S + 4M.
///
/// ref10's `ge_p3_dbl` structure:
///     A = X^2,  B = Y^2,  C = 2*Z^2,  D = -A
///     E = (X+Y)^2 - A - B  =  2*XY
///     G = B + D,  F = G - C,  H = D - B
#[inline(always)]
pub fn ge_double(p: &Ge) -> Ge {
    let a = fe_sq(&p.x);
    let b = fe_sq(&p.y);
    // C = 2*Z^2, i.e. Z^2 doubled. Writing this as Z^2 + Z is a plausible-looking
    // slip that is correct only when Z == 1, so it survives the first doubling
    // (B has Z = 1) and then silently corrupts every later entry of the ladder.
    let c = fe_add(&fe_sq(&p.z), &fe_sq(&p.z));
    let d = fe_neg(&a);
    // 2*XY = (X+Y)^2 - X^2 - Y^2. Omitting the "- Y^2" term has the same
    // shape of failure.
    let e = fe_sub(&fe_sub(&fe_sq(&fe_add(&p.x, &p.y)), &a), &b);
    let g = fe_add(&b, &d);
    let f = fe_sub(&g, &c);
    let h = fe_sub(&d, &b);
    Ge {
        x: fe_mul(&e, &f),
        y: fe_mul(&g, &h),
        z: fe_mul(&f, &g),
        t: fe_mul(&e, &h),
    }
}

/// Field negation.
#[inline(always)]
fn fe_neg(a: &Fe) -> Fe {
    fe_sub(&FE_ZERO, a)
}

/// Compress to the 32-byte wire encoding.
///
/// `zi` is the inverse of Z, supplied by the caller so a whole batch of
/// points can share one inversion (see `batch_invert`).
pub fn ge_to_bytes_with_zi(p: &Ge, zi: &Fe) -> [u8; 32] {
    let x = fe_mul(&p.x, zi);
    let y = fe_mul(&p.y, zi);
    let mut out = fe_to_bytes(&y);
    out[31] ^= (fe_is_negative(&x) as u8) << 7;
    out
}

/// Montgomery's trick: invert `n` field elements using 3(n-1) multiplications
/// and a *single* inversion, instead of `n` inversions.
///
/// This is why batching matters so much: a field inversion is ~254 squarings
/// and ~11 multiplications, whereas a point addition is ~9 multiplications.
/// Inverting every point individually would cost ~30x more than the
/// arithmetic it is compressing.
///
/// Allocation-free: the prefix products are folded into `scratch`, which the
/// caller supplies.
pub fn batch_invert_into(els: &mut [Fe], scratch: &mut [Fe]) {
    let n = els.len();
    if n == 0 {
        return;
    }

    // Running product: scratch[i] = z[0] * ... * z[i].
    let mut acc = FE_ONE;
    for i in 0..n {
        scratch[i] = acc;
        acc = fe_mul(&acc, &els[i]);
    }

    // One inversion for the whole batch.
    let mut acc_inv = fe_invert(&acc);

    // Walk backwards, peeling off one inversion per element.
    for i in (0..n).rev() {
        let tmp = fe_mul(&acc_inv, &scratch[i]);
        acc_inv = fe_mul(&acc_inv, &els[i]);
        els[i] = tmp;
    }
}

/// Allocate a scratch buffer for `batch_invert_into`. Provided by the caller
/// because the device crate is `no_std` and has no allocator.
pub fn batch_invert(els: &mut [Fe], scratch: &mut [Fe]) {
    batch_invert_into(els, scratch)
}

/// Number of precomputed base-point multiples.
///
/// A clamped ed25519 scalar has bit 254 set and bit 255 clear, so bit indices
/// 0..=254 can be set. 255 entries covers it with one slot to spare.
pub const TABLE_LEN: usize = 255;

/// `[scalar]B` as a plain sum of precomputed powers of two.
///
/// `table[i]` is `[2^i]B`, so the result is the sum of `[2^i]B` over the set
/// bits `i` of the scalar. That needs one point addition per set bit and **no
/// doublings at all**.
///
/// This is the whole point of hoisting the ladder out of the per-key work: a
/// plain double-and-add recomputes `[2^i]B` for every key, and that ladder is
/// identical for all of them.
///
/// A random 253-bit scalar has ~126 set bits, so this costs ~126 point
/// additions versus the 512 of the naive `ed25519-compact` loop. The table is
/// 255 * 4 * 5 * 8 = 40800 bytes, which fits in shared memory.
pub fn scalarmult_base_table(scalar: &[u8; 32], table: &[Ge]) -> Ge {
    let mut r = ge_identity();
    let mut first = true;
    // Must cover all 255 bits below p, not just 0..253: clamping always sets
    // bit 254, and a clamped scalar is 253 bits *wide* but the set bit sits at
    // index 254, not 252.
    for i in 0..255usize {
        // Bit i of the little-endian scalar.
        let bit = (scalar[i >> 3] >> (i & 7)) & 1;
        if bit == 1 {
            let addend = table[i];
            if first {
                // Assigning avoids one addition against the identity.
                r = addend;
                first = false;
            } else {
                r = ge_add(&r, &addend);
            }
        }
    }
    r
}

/// Fill `table[i] = [2^i]B` for i in 0..TABLE_LEN by repeated doubling.
///
/// Takes a fixed-size slice rather than returning a `Vec`: the device crate is
/// `no_std` and has no allocator, so the host owns the buffer.
pub fn build_power_table(table: &mut [Ge; TABLE_LEN]) {
    let mut p = ge_base();
    for i in 0..TABLE_LEN {
        table[i] = p;
        p = ge_double(&p);
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use std::string::String;
    #[allow(unused_imports)]
    use std::vec::Vec;

    /// table[i] = [2^i]B
    fn power_table() -> [Ge; TABLE_LEN] {
        let zero = Ge {
            x: FE_ZERO,
            y: FE_ZERO,
            z: FE_ZERO,
            t: FE_ZERO,
        };
        let mut t = [zero; TABLE_LEN];
        build_power_table(&mut t);
        t
    }

    fn compress(p: &Ge) -> [u8; 32] {
        let zi = fe_invert(&p.z);
        ge_to_bytes_with_zi(p, &zi)
    }

    #[test]
    fn base_point_compresses_to_known_value() {
        let b = ge_base();
        assert_eq!(
            hex(&compress(&b)),
            "5866666666666666666666666666666666666666666666666666666666666666"
        );
    }

    #[test]
    fn rfc8032_vectors() {
        // Test vectors from RFC 8032 section 7.1.
        let cases: [(&str, &str); 3] = [
            (
                "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
                "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
            ),
            (
                "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
                "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
            ),
            (
                "c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
                "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
            ),
        ];

        let table = power_table();
        for (seed_hex, want) in cases {
            let seed: Vec<u8> = unhex(seed_hex);
            let expanded = sha512::sha512(&seed);
            let mut scalar = [0u8; 32];
            scalar.copy_from_slice(&expanded[..32]);
            scalar[0] &= 248;
            scalar[31] &= 63;
            scalar[31] |= 64;

            let p = scalarmult_base_table(&scalar, &table);
            assert_eq!(hex(&compress(&p)), want, "seed {}", seed_hex);
        }
    }

    #[test]
    fn table_matches_double_and_add() {
        let table = power_table();
        for seed in 0..8u8 {
            // Build scalars that survive clamping, so each test case is
            // distinct. Clamping clears the low 3 bits of byte 0, so vary
            // higher bytes instead.
            let mut scalar = [0u8; 32];
            scalar[3] = seed.wrapping_mul(37).wrapping_add(11);
            scalar[0] = 0x40 | (seed << 5);
            scalar[31] |= 64;
            scalar[0] &= 248;

            let got = scalarmult_base_table(&scalar, &table);

            // Reference: the textbook double-and-add ladder over all 255 bits.
            let mut acc = ge_identity();
            let mut base = ge_base();
            for bit in 0..255 {
                if (scalar[bit >> 3] >> (bit & 7)) & 1 == 1 {
                    acc = ge_add(&acc, &base);
                }
                base = ge_double(&base);
            }
            assert_eq!(
                hex(&compress(&got)),
                hex(&compress(&acc)),
                "seed {}",
                seed
            );
        }
    }

    #[test]
    fn eight_is_three_doublings() {
        let b = ge_base();
        let mut acc = b;
        for _ in 0..3 {
            acc = ge_double(&acc);
        }
        assert_eq!(hex(&compress(&acc)), hex(&compress(&ge_eight())));
    }

    #[test]
    fn walk_increments_the_scalar() {
        // [s]B + 8B == [(s+8)]B, which is the property the batch relies on.
        let table = power_table();
        let mut scalar = [0u8; 32];
        scalar[4] = 0x37; // a byte clamping leaves alone
        scalar[0] = 0x40;
        scalar[31] |= 64;

        let start = scalarmult_base_table(&scalar, &table);
        let walked = ge_add(&start, &ge_eight());

        let mut next = scalar;
        add_clamped_scalar(&mut next, 8);
        let expected = scalarmult_base_table(&next, &table);

        assert_eq!(hex(&compress(&walked)), hex(&compress(&expected)));
    }

    /// Add a small delta to a clamped scalar, preserving the clamping
    /// invariants that make it a valid ed25519 secret scalar.
    pub fn add_clamped_scalar(scalar: &mut [u8; 32], delta: u64) {
        let mut carry = delta;
        for i in 0..32 {
            if carry == 0 {
                break;
            }
            let sum = scalar[i] as u64 + (carry & 0xFF);
            scalar[i] = sum as u8;
            carry = (carry >> 8) + (sum >> 8);
        }
        // Restore clamping. A small increment never flips these in practice,
        // but assert rather than assume.
        scalar[0] &= 248;
        scalar[31] &= 63;
        scalar[31] |= 64;
    }

    #[test]
    fn batch_invert_matches_individual() {
        let inputs: Vec<Fe> = (1..12u64).map(|i| [i * 7, i * 3, i * 11, i * 5, i + 1]).collect();
        let mut want: Vec<Fe> = inputs.clone();
        for e in want.iter_mut() {
            *e = fe_invert(e);
        }
        let mut got = inputs.clone();
        let mut scratch = [FE_ZERO; 16];
        batch_invert_into(&mut got, &mut scratch);
        assert_eq!(got, want);
    }

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{:02x}", x)).collect()
    }

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
            .collect()
    }
}
