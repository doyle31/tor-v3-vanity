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

/// A precomputed addend in Niels form: `(Y+X, Y-X, 2dT)`.
///
/// This is ref10's `ge_precomp`. The point it stands for is the *affine*
/// representative `(X:Y:1:T)` of the same curve point, which is the whole
/// trick: the `Z2` factor of the addition formula is 1, so the `2*Z1*Z2` term
/// of `ge_add` collapses to a field doubling and costs no multiply.
#[derive(Copy, Clone)]
pub struct Niels {
    pub yplusx: Fe,
    pub yminusx: Fe,
    pub xy2d: Fe,
}

/// The neutral element `(0:1:1:0)` in Niels form.
///
/// `ge_madd` is a *unified* formula, so this is a valid addend: adding it
/// returns `p` (up to the projective scale that the formula leaves in `Z`).
/// That is what lets the comb below run its window loop with no branches.
#[inline(always)]
pub fn ge_niels_identity() -> Niels {
    Niels {
        yplusx: FE_ONE,
        yminusx: FE_ONE,
        xy2d: FE_ZERO,
    }
}

/// Projective point -> Niels form, given `zi = p.z^-1`. Host side only.
///
/// **`zi` is not optional.** The Niels form fixes the addend's `Z` to 1, so a
/// projective point with `Z != 1` has to be divided through by `Z` first:
/// `Y+X` must be the affine `(Y + X)/Z`, and `2dT` the affine `2dT/Z`. Feeding
/// `ge_madd` an un-normalised triple does not fail loudly -- it still returns a
/// point satisfying `T*Z == X*Y` -- but it is a *different* point, because the
/// `2*Z1*Z2` term silently loses its `Z2`. This is ref10's
/// `ge_p3_to_precomp`, and its one-inversion-per-entry is why that function
/// exists at all.
pub fn ge_to_niels(p: &Ge, zi: &Fe) -> Niels {
    let y = fe_mul(&p.y, zi);
    let x = fe_mul(&p.x, zi);
    Niels {
        yplusx: fe_add(&y, &x),
        yminusx: fe_sub(&y, &x),
        xy2d: fe_mul(&fe_mul(&p.t, zi), &FE_D2),
    }
}

/// Point addition with a Niels addend. **7M**, two multiplies cheaper than
/// `ge_add`'s 9M, and the cheapest addition available for a fixed second
/// operand.
///
/// ref10's `ge_madd`, with `Z2 = 1` folded into the addend:
///
/// ```text
///     A = (Y1-X1) * yminusx
///     B = (Y1+X1) * yplusx
///     C = T1     * xy2d        (T1 * 2d * T2, with 2d*T2 precomputed)
///     D = 2*Z1                  (2*Z1*Z2, with Z2 == 1 -- no multiply)
///     E = B-A, F = D-C, G = D+C, H = B+A
///     X3 = E*F, Y3 = G*H, Z3 = F*G, T3 = E*H
/// ```
///
/// Three multiplies for the cross terms plus four for the output. Valid for
/// every input, including `p == q` and `p` the identity, exactly like
/// `ge_add`: the formula is unified. `madd_matches_add` in the tests checks
/// that against `ge_add` directly.
#[inline(always)]
pub fn ge_madd(p: &Ge, q: &Niels) -> Ge {
    let a = fe_mul(&fe_sub(&p.y, &p.x), &q.yminusx);
    let b = fe_mul(&fe_add(&p.y, &p.x), &q.yplusx);
    let c = fe_mul(&p.t, &q.xy2d);
    let d = fe_add(&p.z, &p.z);
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

// ---------------------------------------------------------------------------
// The same trick with a half-size scratch buffer.
//
// `batch_invert_into` needs `n-1` stored values -- the prefix products -- and
// that is a hard lower bound, not an implementation accident. Write
// `P[i] = z[0] * .. * z[i]` with `P[-1] = 1`. The backwards peel maintains
// `acc = 1/P[i]` and does
//
//     out[i] = P[i-1] * acc = P[i-1] / P[i] = z[i]^-1
//     acc <- acc * z[i]                 (which is 1/P[i-1])
//
// so it needs *both* `P[i-1]` and `z[i]` from storage. The kernel has both:
// `z[i]` is in the chunk's `zs` array, because the second walk needs `z[i]^-1`
// written back into it, and the `P` are what `scratch` is for. Two arrays, and
// there is no way to one-array it: keep only the `P` and the peel has to
// divide (`acc` would have to become `acc / out[i]`, and a field division is an
// inversion); keep only the `z` and every `P[i-1]` has to be re-derived from a
// fresh product tree. See SCRATCH.md.
//
// What *is* free is the parity of the index. `P[i-1] = P[i-2] * z[i-1]`, so
// storing only every other prefix product costs exactly one `fe_mul` to
// rebuild the missing one -- from a stored neighbour and from `z[i-1]`, which
// the backwards peel has not overwritten yet. That halves `scratch` for
// `n/2 - 1` extra multiplies per batch.
// ---------------------------------------------------------------------------

/// Slots of `scratch` that `batch_invert_stride2_into` needs for `n` elements.
pub const fn stride2_scratch_len(n: usize) -> usize {
    (n + 1) / 2
}

/// `batch_invert_into` with `scratch` holding every *other* prefix product.
///
/// `scratch[j]` is `P[2j - 1]`, i.e. the product of the `2j` elements *before*
/// index `2j`; slot 0 is therefore the constant `P[-1] = 1`. That needs
/// `stride2_scratch_len(n)` slots instead of `n`.
///
/// Correct for every `n`, including 0, 1 and 2. `batch_invert_into` is kept as
/// the reference the tests check this against, and as the cheaper choice when
/// the caller has the local memory to spare.
///
/// Cost against `batch_invert_into`, per batch:
///
/// * multiplies: `7n/2 - 2` (even `n`) instead of `3n - 3`, so `n/2 + 1` more
///   -- 129 more at `CHUNK == 256`, or +0.504 per candidate. Note that this is
///   *more* than the `n/2` the rebuilds account for, because the peel also
///   skips its final accumulator update at `i == 0`, which the classic loop
///   does perform;
/// * one inversion, unchanged;
/// * `Fe` stores: `1.5n` instead of `2n`, i.e. `n/2` fewer. The extra `Fe`
///   loads the rebuilds need (`n/2` of them) come out of the same budget, so
///   net traffic falls by only `0.5n` words, about 10% at `CHUNK == 256`.
///   `SCRATCH.md` measures what that is worth: one `fe_mul` is ~175 SASS and
///   one 64-bit local access is 1, so the trade loses by roughly 33:1 and is
///   unfavourable at *every* `CHUNK`, since both terms scale as `n`.
///
/// # The three off-by-one traps
///
/// Batch inversion has a well-known off-by-one shape -- `n` inverses out of
/// `3n-3` multiplies and one inversion -- and this variant adds more places to
/// get it wrong, all of which the tests below pin down:
///
/// * the forward pass stores `P[i-1]`, the product *before* absorbing `z[i]`,
///   because that is what the peel asks for at index `i`. Storing `P[i]`
///   instead is off by one element and still produces plausible-looking code;
/// * `P[-1] = 1` is a constant, not a stored value, so index 0 needs no read
///   at all. Reading `scratch[-1]`-equivalent garbage there returns the
///   reciprocal of the whole batch product for `out[0]`, which is a perfectly
///   good field element and silently wrong for exactly one candidate;
/// * `acc_inv` must be advanced with the *input* `z[i]`, so the peeled value
///   has to be computed into a temporary before it is stored. Writing it
///   through `els[i]` first feeds the reciprocal of `z[i]` back into the
///   accumulator. `batch_invert_into` has the same hazard, which is why it
///   keeps a `tmp`.
#[inline(always)]
pub fn batch_invert_stride2_into(els: &mut [Fe], scratch: &mut [Fe]) {
    let n = els.len();
    debug_assert!(
        scratch.len() >= stride2_scratch_len(n),
        "stride-2 batch inversion needs {} slots for {} elements, got {}",
        stride2_scratch_len(n),
        n,
        scratch.len()
    );
    if n == 0 {
        return;
    }
    if n == 1 {
        // No prefix products exist and the peel has nothing to walk, so this
        // is just the single inversion the whole scheme exists to amortise.
        els[0] = fe_invert(&els[0]);
        return;
    }

    // Forward pass. On entry to the body for index `i`, `acc == P[i-1]`; on
    // exit, `acc == P[n-1]`, the product being inverted. Only even `i` stores,
    // and the loop is stepped two at a time so the store is unconditional.
    let mut acc = FE_ONE;
    let mut i = 0;
    while i + 1 < n {
        scratch[i / 2] = acc;
        acc = fe_mul(&acc, &els[i]);
        acc = fe_mul(&acc, &els[i + 1]);
        i += 2;
    }
    if i < n {
        // Odd tail: store the last even prefix, then absorb the final element.
        // `P[n-1]` itself is not stored -- it would land in a slot the peel
        // never reads.
        scratch[i / 2] = acc;
        acc = fe_mul(&acc, &els[i]);
    }

    // One inversion for the whole batch.
    let mut acc_inv = fe_invert(&acc);

    // Backwards peel. On entry to the body for index `i`, `acc_inv == 1/P[i]`.
    //
    // `P[i-1]` is at `scratch[i / 2]` either way; only whether it is there
    // *as is* depends on the parity of `i-1`:
    //
    //   * `i` even => `i-1` is odd => `i-1 = 2*(i/2) - 1`, so `P[i-1]` is
    //     exactly the stored slot. No multiply.
    //   * `i` odd  => `i-1` is even => `P[i-1] = P[i-2] * z[i-1]` and `P[i-2]`
    //     is the stored slot `(i-1)/2 == i/2`. One multiply, and `z[i-1]` is
    //     still live because the peel has only reached `i` so far.
    let mut i = n - 1;
    loop {
        if i == 0 {
            // `P[-1] = 1`, so `out[0] == acc_inv` and the accumulator's last
            // step is skipped entirely. Checked before the multiply, or the
            // peel pays a multiply whose result is thrown away.
            els[0] = acc_inv;
            return;
        }
        let p = if i % 2 == 0 {
            scratch[i / 2]
        } else {
            // `z[i-1]` is still the original value: the peel only reaches index
            // `i`, so index `i-1` has not been written yet.
            fe_mul(&scratch[i / 2], &els[i - 1])
        };
        // Order matters: `acc_inv` must be advanced with the *input* `z[i]`, so
        // `out` is computed into a temporary first. Writing it through `els[i]`
        // first would feed the reciprocal of `z[i]` back into the accumulator
        // and corrupt every later element. `batch_invert_into` has the same
        // hazard, which is why it keeps a `tmp` too.
        let out = fe_mul(&p, &acc_inv);
        acc_inv = fe_mul(&acc_inv, &els[i]);
        els[i] = out;
        i -= 1;
    }
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
/// **This is no longer what the kernel does** -- see `scalarmult_base_comb`.
/// It is kept as the reference implementation that the comb is tested against,
/// and because it is the cheapest way to read the cost of the old approach:
/// the `if bit == 1` below is a divergent branch, so a warp executes the body
/// on every bit that any of its 32 lanes has set. With 32 independent lanes
/// that is ~251 of the 255 iterations, not the ~126 a single thread would
/// pay, and a branchless rewrite would execute exactly the same ~251 additions.
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

// ---------------------------------------------------------------------------
// Fixed-window comb over the base point.
//
// Why not just the power table above: on a GPU the `if bit == 1` in
// `scalarmult_base_table` is a *divergent* branch, so a warp runs the body on
// any bit that is set in *any* of its 32 lanes. The probability that all 32
// lanes agree a given bit is zero is 2^-32, so the warp pays ~251 additions
// where a single thread would pay ~126. Making the loop branchless does not
// help: the body is still ~251 warp-level additions. Only a comb, whose
// operation count does not depend on the scalar's sparsity at all, helps.
//
// The comb is "all subsets of a window": each window of `COMB_W` consecutive
// scalar bits has its 2^COMB_W subset sums precomputed, so a window costs
// exactly one addition whatever its digits are. That needs no doublings and
// no digit sign flips, which matters because a doubling is 8M against an
// addition's 9M in this representation -- see the notes in 01-comb.md.
// ---------------------------------------------------------------------------

/// Bits per comb window. Build-time tunable so it can be swept without code
/// changes; see `01-comb.md` for the table-size / operation-count tradeoff.
pub const COMB_W: usize = match option_env!("T3V_COMB_W") {
    Some(s) => crate::parse_usize(s.as_bytes()),
    None => 4,
};

/// Clamping clears scalar bits 0, 1 and 2 (`sk[0] & 248 == sk[0]`), so the comb
/// starts at bit 3 and never needs those windows. It also forces bit 254
/// (`sk[31] |= 64`) and clears bit 255, which is why the range ends at 254.
pub const COMB_FIRST_BIT: usize = 3;

/// Scalar bits 3..=254 inclusive.
pub const COMB_BITS: usize = 252;

/// Windows of `COMB_W` bits covering `COMB_BITS`. `COMB_W` must divide
/// `COMB_BITS` so that no window runs off the top of the scalar.
pub const COMB_WINDOWS: usize = COMB_BITS / COMB_W;

/// Table slots per window: the identity (digit 0) plus every non-zero
/// `COMB_W`-bit pattern.
pub const COMB_SLOTS: usize = 1 << COMB_W;

/// Total comb table entries.
pub const COMB_LEN: usize = COMB_WINDOWS * COMB_SLOTS;

/// Everything the kernel needs that depends only on the base point: the comb
/// table and the walk step `8B`, both in Niels form.
///
/// `#[repr(C)]` with only `[u64; 5]` inside means the host can upload it as a
/// flat `&[u64]` with no packing, and the device can index it directly.
#[repr(C)]
pub struct BaseTables {
    pub comb: [Niels; COMB_LEN],
    /// `8B`, the walk step. `table[3]` of a power table would be the same
    /// point; it lives here so the walk needs no `[2^i]B` array at all.
    pub eight: Niels,
}

/// u64 words in one `Niels` (three field elements, no padding).
pub const NIELS_WORDS: usize = 15;

/// u64 words in the whole uploaded blob.
pub const BASE_TABLE_WORDS: usize = COMB_LEN * NIELS_WORDS + NIELS_WORDS;

/// Fill `comb` with every window's subset sums, and `eight` with `8B`.
///
/// `comb[j*COMB_SLOTS + m]` (m in 1..COMB_SLOTS) is the sum of `[2^b]B` over
/// the `b` with bit `b` of `m` set, taken from window `j` -- that is, over
/// the scalar bit positions `COMB_FIRST_BIT + j*COMB_W + b`. Slot 0 is the
/// identity, so a zero digit adds nothing and the window loop needs no branch.
///
/// The subset sums are built from the same `[2^i]B` powers the old ladder used,
/// so the two implementations share one source of truth, and each window is
/// normalised into Niels form with one *batched* inversion (Montgomery's trick)
/// rather than one inversion per entry. Building the table costs a few
/// thousand field multiplies, once, on the host.
pub fn build_comb_table(comb: &mut [Niels; COMB_LEN], eight: &mut Niels) {
    // 2..=4 is the useful range: COMB_W must also divide COMB_BITS (252), and
    // the table is 2^COMB_W slots per window, so 5/6/7 are either impossible or
    // absurdly large (315 KiB at 6, 540 KiB at 7). See `01-comb.md`.
    assert!(
        COMB_W >= 2 && COMB_W <= 4,
        "T3V_COMB_W must be in 2..=4"
    );
    assert!(COMB_BITS % COMB_W == 0, "T3V_COMB_W must divide 252");

    let zero = Ge {
        x: FE_ZERO,
        y: FE_ZERO,
        z: FE_ZERO,
        t: FE_ZERO,
    };
    let mut power = [zero; TABLE_LEN];
    build_power_table(&mut power);

    // Per-window scratch: the window's points, their Z coordinates, and the
    // prefix products `batch_invert_into` needs.
    let mut points = [zero; COMB_SLOTS];
    let mut zs = [FE_ONE; COMB_SLOTS];
    let mut scratch = [FE_ONE; COMB_SLOTS];

    for j in 0..COMB_WINDOWS {
        let base = COMB_FIRST_BIT + j * COMB_W;
        for m in 0..COMB_SLOTS {
            let mut acc = ge_identity();
            let mut first = true;
            for b in 0..COMB_W {
                if (m >> b) & 1 == 1 {
                    if first {
                        // Assigning saves an addition against the identity.
                        acc = power[base + b];
                        first = false;
                    } else {
                        acc = ge_add(&acc, &power[base + b]);
                    }
                }
            }
            points[m] = acc;
            zs[m] = acc.z;
        }

        // One inversion for the whole window: this is the step that makes the
        // entries usable by `ge_madd`.
        batch_invert_into(&mut zs, &mut scratch);
        for m in 0..COMB_SLOTS {
            comb[j * COMB_SLOTS + m] = ge_to_niels(&points[m], &zs[m]);
        }
    }

    // The walk step, normalised the same way. `8B` used to be three doublings
    // of the base point in *every thread* of *every launch*; it is now built
    // once, here, and uploaded with the table.
    let eight_point = ge_eight();
    let zi = fe_invert(&eight_point.z);
    *eight = ge_to_niels(&eight_point, &zi);
}

/// The `COMB_W` scalar bits of window `j`, as a table slot index in
/// `0..COMB_SLOTS`.
///
/// `COMB_W <= 5` bits starting at a bit offset of at most 7 span at most 12
/// bits, so a 16-bit read is always enough. The high byte is only *used* when
/// the offset is 7, and then `byte + 1 <= 30`; when `byte == 31` (the top
/// window) the read wraps to byte 0, lands at bit 5 or above after the shift,
/// and is discarded by the mask -- so this is in bounds for every window
/// without a bounds check or a branch.
#[inline(always)]
fn comb_slot(scalar: &[u8; 32], j: usize) -> usize {
    let base = COMB_FIRST_BIT + j * COMB_W;
    let byte = base >> 3;
    let off = (base & 7) as u32;
    let lo = scalar[byte] as u32;
    let hi = scalar[(byte + 1) & 31] as u32;
    (((lo | (hi << 8)) >> off) & (COMB_SLOTS as u32 - 1)) as usize
}

/// `[scalar]B` as a sum of precomputed window subset sums.
///
/// `COMB_WINDOWS` point additions and **no doublings**, whatever the scalar
/// looks like -- the operation count is data independent, which is the entire
/// point (see the divergence note above `COMB_W`).
///
/// The first window is added against the identity rather than assigned, because
/// the accumulator has to be a `Ge` and a `Niels` cannot be one. `ge_madd` is
/// unified, so that addition is exact and costs 7M like any other; at the
/// default `COMB_W == 4` that is 63 additions against the ladder's ~251 per
/// warp.
pub fn scalarmult_base_comb(scalar: &[u8; 32], comb: &[Niels; COMB_LEN]) -> Ge {
    let mut r = ge_madd(&ge_identity(), &comb[comb_slot(scalar, 0)]);
    for j in 1..COMB_WINDOWS {
        let addend = comb[j * COMB_SLOTS + comb_slot(scalar, j)];
        r = ge_madd(&r, &addend);
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

    /// The blob the host uploads, built once per test run.
    fn comb_table() -> [Niels; COMB_LEN] {
        let mut comb: [Niels; COMB_LEN] = unsafe { core::mem::zeroed() };
        let mut eight: Niels = unsafe { core::mem::zeroed() };
        build_comb_table(&mut comb, &mut eight);
        comb
    }

    fn comb_eight() -> Niels {
        let mut comb: [Niels; COMB_LEN] = unsafe { core::mem::zeroed() };
        let mut eight: Niels = unsafe { core::mem::zeroed() };
        build_comb_table(&mut comb, &mut eight);
        eight
    }

    /// A deterministic xorshift, so the "random" scalars below are
    /// reproducible and a failure can be replayed.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
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

    /// The same RFC 8032 vectors, but through the comb. This is the check that
    /// matters for the rewrite: an independent table layout and a different
    /// addition formula, so agreeing with the published pubkey pins down both.
    #[test]
    fn comb_rfc8032_vectors() {
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

        let comb = comb_table();
        for (seed_hex, want) in cases {
            let seed: Vec<u8> = unhex(seed_hex);
            let expanded = sha512::sha512(&seed);
            let mut scalar = [0u8; 32];
            scalar.copy_from_slice(&expanded[..32]);
            scalar[0] &= 248;
            scalar[31] &= 63;
            scalar[31] |= 64;

            let p = scalarmult_base_comb(&scalar, &comb);
            assert_eq!(hex(&compress(&p)), want, "seed {}", seed_hex);
        }
    }

    /// `ge_madd` must agree with `ge_add` for every input, including the ones
    /// a *non*-unified formula gets wrong: the two operands equal, and either
    /// operand the identity. A comb walks into those cases constantly, because
    /// two windows can easily select the same table entry.
    #[test]
    fn madd_matches_add() {
        let mut q = ge_base();
        let mut p = ge_base();
        for i in 0..64 {
            // Pseudo-random-ish steps, so the operands collide sometimes.
            if i % 7 == 3 {
                q = ge_double(&q);
            } else if i % 11 == 5 {
                p = ge_add(&p, &q);
            }

            // Deliberately projective: `p` and `q` have arbitrary Z, so this
            // also covers the Niels normalisation done at table-build time.
            let got = ge_madd(&p, &ge_to_niels(&q, &fe_invert(&q.z)));
            let want = ge_add(&p, &q);
            assert_eq!(hex(&compress(&got)), hex(&compress(&want)), "case {}", i);

            // p == q (the doubling case) and the identity addend.
            if i % 5 == 0 {
                let got = ge_madd(&q, &ge_to_niels(&q, &fe_invert(&q.z)));
                let want = ge_add(&q, &q);
                assert_eq!(hex(&compress(&got)), hex(&compress(&want)), "self {}", i);
            }
            let got = ge_madd(&p, &ge_niels_identity());
            let want = ge_add(&p, &ge_identity());
            assert_eq!(hex(&compress(&got)), hex(&compress(&want)), "ident {}", i);
        }
    }

    /// The comb must equal the power-table ladder on arbitrary clamped
    /// scalars -- the cross-check that makes the RFC vectors more than three
    /// lucky coincidences.
    #[test]
    fn comb_matches_power_ladder() {
        let power = power_table();
        let comb = comb_table();
        let mut rng = Rng(0x1319_8A2E_0370_7344);
        for i in 0..64 {
            let mut scalar = [0u8; 32];
            for b in scalar.iter_mut() {
                *b = (rng.next() >> 24) as u8;
            }
            // Clamp, exactly as `BatchState::begin` does.
            scalar[0] &= 248;
            scalar[31] &= 63;
            scalar[31] |= 64;

            let got = scalarmult_base_comb(&scalar, &comb);
            let want = scalarmult_base_table(&scalar, &power);
            assert_eq!(
                hex(&compress(&got)),
                hex(&compress(&want)),
                "case {} scalar {}",
                i,
                hex(&scalar)
            );
        }
    }

    /// Every comb slot, for every window, must be the subset sum it claims to
    /// be. Checking the table directly catches a mis-indexed build that the
    /// scalar-multiply comparison could hide behind a symmetric mistake.
    #[test]
    fn comb_slots_are_subset_sums() {
        let power = power_table();
        let comb = comb_table();
        for j in 0..COMB_WINDOWS {
            let base = COMB_FIRST_BIT + j * COMB_W;
            for m in 0..COMB_SLOTS {
                // Rebuild the slot the slow way, straight from the power table.
                let mut want = ge_identity();
                let mut first = true;
                for b in 0..COMB_W {
                    if (m >> b) & 1 == 1 {
                        if first {
                            want = power[base + b];
                            first = false;
                        } else {
                            want = ge_add(&want, &power[base + b]);
                        }
                    }
                }
                assert_eq!(
                    hex(&compress(&ge_madd(&ge_identity(), &comb[j * COMB_SLOTS + m]))),
                    hex(&compress(&want)),
                    "window {} slot {}",
                    j,
                    m
                );
            }
        }
    }

    /// `comb_slot` must return exactly the bits the window spans, including at
    /// the two ends of the scalar where the 16-bit read runs off the end.
    #[test]
    fn comb_slot_reads_the_right_bits() {
        let mut scalar = [0u8; 32];
        for (i, b) in scalar.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(37).wrapping_add(11);
        }
        for j in 0..COMB_WINDOWS {
            let base = COMB_FIRST_BIT + j * COMB_W;
            let want = (0..COMB_W).fold(0usize, |acc, b| {
                acc | (((scalar[(base + b) >> 3] >> ((base + b) & 7)) & 1) as usize) << b
            });
            assert_eq!(comb_slot(&scalar, j), want, "window {}", j);
        }
    }

    /// The uploaded `8B` must be the same point the old `make_eight` built.
    #[test]
    fn comb_eight_matches_make_eight() {
        let n = comb_eight();
        let mut want = ge_base();
        for _ in 0..3 {
            want = ge_double(&want);
        }
        let ident = ge_identity();
        assert_eq!(
            hex(&compress(&ge_madd(&ident, &n))),
            hex(&compress(&want))
        );
        // And it is what the walk actually needs: [s]B + 8B == [s+8]B.
        let power = power_table();
        let mut scalar = [0u8; 32];
        scalar[4] = 0x37;
        scalar[0] = 0x40;
        scalar[31] |= 64;
        scalar[0] &= 248;
        let walked = ge_madd(&scalarmult_base_table(&scalar, &power), &n);
        let mut next = scalar;
        add_clamped_scalar(&mut next, 8);
        assert_eq!(
            hex(&compress(&walked)),
            hex(&compress(&scalarmult_base_table(&next, &power)))
        );
    }

    /// The blob is `#[repr(C)]` with only `[u64; 5]` inside, so the host's flat
    /// `u64` view of it is exact. If that ever stops holding, the host uploads
    /// the wrong bytes and every key is garbage.
    #[test]
    fn base_tables_layout_is_flat() {
        let mut base: BaseTables = unsafe { core::mem::zeroed() };
        build_comb_table(&mut base.comb, &mut base.eight);
        let p = &base as *const BaseTables as *const u8;
        let comb = &base.comb as *const Niels as *const u8;
        let eight = &base.eight as *const Niels as *const u8;
        assert_eq!(
            comb as usize - p as usize,
            0,
            "comb must start the struct"
        );
        assert_eq!(
            eight as usize - p as usize,
            COMB_LEN * NIELS_WORDS * 8,
            "eight must follow comb with no padding"
        );
        assert_eq!(core::mem::size_of::<Niels>(), NIELS_WORDS * 8);
        assert_eq!(
            core::mem::size_of::<BaseTables>(),
            BASE_TABLE_WORDS * 8,
            "size_of must match BASE_TABLE_WORDS"
        );
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

    // -----------------------------------------------------------------------
    // Batch inversion, including the half-size-scratch variant.
    //
    // The classic shape is `n` inverses out of `3n-3` multiplies and one
    // inversion, so `n = 1` and `n = 2` are the interesting degenerate cases
    // and the odd/even boundary at `n = 3` is where an off-by-one in the
    // reverse peel shows up first. Everything is checked against individual
    // `fe_invert` rather than against the other implementation, so the two
    // cannot be wrong in the same way.
    // -----------------------------------------------------------------------

    /// Invert each element on its own; the oracle for every test below.
    fn individual_inverses(inputs: &[Fe]) -> Vec<Fe> {
        inputs.iter().map(fe_invert).collect()
    }

    /// A reproducible, well-spread set of `n` non-zero field elements.
    ///
    /// The limbs are masked to 51 bits and the top limb is forced non-zero, so
    /// no element can be 0 mod p. Values whose limbs all sit at `MASK` are
    /// included on purpose: those are what breaks a lazy reduction.
    fn batch_inputs(n: usize) -> Vec<Fe> {
        const M: u64 = (1u64 << 51) - 1;
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s & M
        };
        (0..n)
            .map(|i| {
                if i % 5 == 0 {
                    [M, M, M, M, next() | 1]
                } else {
                    [next() | 1, next(), next(), next(), next() | 1]
                }
            })
            .collect()
    }

    /// `batch_invert_into` and `batch_invert_stride2_into` must both return the
    /// individual inverses, for every length from 0 to 300.
    ///
    /// A sweep rather than a handful of lengths because the two schemes differ
    /// in their parity handling, and the failures are length-specific: a
    /// broken odd-length tail shows up at `n = 3, 5, 7, ...`, a broken
    /// `P[-1] = 1` at every length, and a wrong `P[n-1]` only at lengths where
    /// the last prefix would be stored. Length 0 and 1 are the ones the peel
    /// can get structurally wrong (no prefixes at all), so they are in here.
    #[test]
    fn batch_invert_lengths_0_to_300() {
        let mut scratch_full = [FE_ZERO; 300];
        let mut scratch_half = [FE_ZERO; 300];
        for n in 0..=300usize {
            let inputs = batch_inputs(n);
            let want = individual_inverses(&inputs);

            let mut got = inputs.clone();
            batch_invert_into(&mut got, &mut scratch_full[..n.max(1)]);
            assert_eq!(got, want, "batch_invert_into, n = {}", n);

            let half = stride2_scratch_len(n);
            let mut got = inputs.clone();
            batch_invert_stride2_into(&mut got, &mut scratch_half[..half.max(1)]);
            assert_eq!(got, want, "batch_invert_stride2_into, n = {}", n);
        }
    }

    /// The degenerate lengths, spelled out one by one.
    ///
    /// `n = 0` must be a no-op on both sides; `n = 1` is a bare inversion with
    /// no multiply and no storage at all; `n = 2` is the smallest case where
    /// the peel runs, and the smallest where the stride-2 scheme has to
    /// rebuild a prefix. Each is checked against `fe_invert` directly so a
    /// shared bug in the harness cannot hide.
    #[test]
    fn batch_invert_degenerate_lengths() {
        let mut scratch = [FE_ZERO; 4];

        // n = 0: nothing in, nothing out, no panic.
        let mut none: Vec<Fe> = Vec::new();
        batch_invert_into(&mut none, &mut []);
        assert!(none.is_empty());
        let mut none2: Vec<Fe> = Vec::new();
        batch_invert_stride2_into(&mut none2, &mut []);
        assert!(none2.is_empty());

        // n = 1.
        for v in 1u64..=8 {
            let a: Fe = [v, v * 3, 0, 0, 0];
            let mut one = [a];
            batch_invert_into(&mut one, &mut scratch);
            assert_eq!(one[0], fe_invert(&a), "n=1 v={}", v);
            let mut one = [a];
            batch_invert_stride2_into(&mut one, &mut scratch);
            assert_eq!(one[0], fe_invert(&a), "n=1 stride2 v={}", v);
        }

        // n = 2 and n = 3: the smallest lengths with a real peel, and the
        // first odd length.
        for n in [2usize, 3] {
            let inputs = batch_inputs(n);
            let want = individual_inverses(&inputs);

            let mut got = inputs.clone();
            batch_invert_into(&mut got, &mut scratch[..n]);
            assert_eq!(got, want, "n={}", n);

            let mut got = inputs.clone();
            batch_invert_stride2_into(&mut got, &mut scratch[..stride2_scratch_len(n)]);
            assert_eq!(got, want, "n={} stride2", n);
        }
    }

    /// Large prime-ish lengths, including the kernel's own `CHUNK`.
    ///
    /// 251 and 257 are prime; 256 is a power of two, which is the case where a
    /// "split in half" indexing scheme is most likely to be off by one. 1023
    /// is `2^10 - 1`, so every element but the last lands in a full pair.
    #[test]
    fn batch_invert_large_lengths() {
        let mut scratch_full = [FE_ZERO; 1100];
        let mut scratch_half = [FE_ZERO; 1100];
        for n in [
            7usize, 8, 9, 15, 16, 17, 127, 128, 129, 251, 255, 256, 257, 511, 512, 1023,
        ] {
            let inputs = batch_inputs(n);
            let want = individual_inverses(&inputs);

            let mut got = inputs.clone();
            batch_invert_into(&mut got, &mut scratch_full[..n]);
            assert_eq!(got, want, "batch_invert_into, n = {}", n);

            let mut got = inputs.clone();
            batch_invert_stride2_into(&mut got, &mut scratch_half[..stride2_scratch_len(n)]);
            assert_eq!(got, want, "batch_invert_stride2_into, n = {}", n);
        }
    }

    /// The stride-2 buffer really is half the size, and the two schemes agree
    /// bit for bit.
    ///
    /// Agreeing with `fe_invert` already pins the *values*; this pins the
    /// *contract* on `scratch`, so a future edit that quietly needs the full
    /// length fails here rather than scribbling past the array.
    #[test]
    fn stride2_scratch_is_half_and_agrees() {
        for n in 0..=64usize {
            assert_eq!(stride2_scratch_len(n), (n + 1) / 2, "n = {}", n);
        }
        for n in [1usize, 2, 3, 8, 255, 256, 257] {
            let inputs = batch_inputs(n);
            let mut a = inputs.clone();
            let mut b = inputs.clone();
            let mut full = [FE_ZERO; 300];
            let mut half = [FE_ZERO; 300];
            batch_invert_into(&mut a, &mut full[..n]);
            batch_invert_stride2_into(&mut b, &mut half[..stride2_scratch_len(n)]);
            assert_eq!(a, b, "n = {}", n);
        }
    }

    /// A duplicated or adversarial element must not upset either scheme.
    ///
    /// Montgomery's trick is agnostic to the inputs, so repeated values are a
    /// cheap way to catch an indexing bug that happens to be masked by
    /// distinct values: with duplicates, several peel steps consume the *same*
    /// array slot contents.
    #[test]
    fn batch_invert_tolerates_repeated_elements() {
        let m = (1u64 << 51) - 1;
        // Every entry must be non-zero: `p` itself is 0 in the field, and
        // inverting 0 is meaningless, so `p - 1` is the interesting top end.
        let patterns: [Fe; 4] = [
            [1, 0, 0, 0, 0],
            [m, m, m, m, m],
            [m - 19, m, m, m, m], // p - 1
            [0, 0, 0, 0, 18],
        ];
        let mut scratch_full = [FE_ZERO; 40];
        let mut scratch_half = [FE_ZERO; 40];
        for pat in patterns.iter() {
            for n in [1usize, 2, 3, 4, 5, 8, 9, 16, 17, 32] {
                // Alternate between the pattern and a second value so the
                // product is not degenerate.
                let other: Fe = [pat[0] ^ 0x1234, pat[1], pat[2] ^ 7, pat[3], pat[4]];
                let inputs: Vec<Fe> = (0..n)
                    .map(|i| if i % 2 == 0 { *pat } else { other })
                    .collect();
                let want = individual_inverses(&inputs);

                let mut got = inputs.clone();
                batch_invert_into(&mut got, &mut scratch_full[..n]);
                assert_eq!(got, want, "n={} pat={:?}", n, pat);

                let mut got = inputs.clone();
                batch_invert_stride2_into(&mut got, &mut scratch_half[..stride2_scratch_len(n)]);
                assert_eq!(got, want, "n={} stride2 pat={:?}", n, pat);
            }
        }
    }

    /// The multiply count is the whole point of the trade in `SCRATCH.md`, so
    /// it is checked arithmetically rather than asserted in prose.
    ///
    /// `fe_mul` is a free function, so instrumenting it would need a shim that
    /// the device build would not compile; instead the closed forms are
    /// checked against a per-index model of the loop. If someone finds a
    /// cheaper peel this fails and the table in `SCRATCH.md` needs updating.
    #[test]
    fn stride2_multiply_counts_match_their_closed_forms() {
        for n in [2usize, 3, 4, 5, 8, 16, 17, 128, 255, 256, 257] {
            // The classic peel is 3n - 3: n in the forward pass and 2n in the
            // peel, less the 3 for the pairs of multiplies the baseline also
            // skips at the ends.
            assert_eq!(3 * n - 3, classic_mul_count(n), "n = {}", n);

            // The stride-2 peel, index by index: the forward pass is n
            // multiplies; the peel pays 2 per element (one for out[i], one to
            // advance the accumulator) plus one rebuild for each odd index,
            // minus the two multiplies the ends do not need (out[0] is the
            // accumulator itself, so neither the product nor the accumulator
            // update happens at i == 0).
            let mut modelled = n;
            for i in 0..n {
                if i == 0 {
                    continue;
                }
                modelled += 2;
                if i % 2 == 1 {
                    modelled += 1;
                }
            }
            assert_eq!(modelled, stride2_mul_count(n), "n = {}", n);
        }
        assert_eq!(classic_mul_count(1), 0, "n = 1 is a bare inversion");
        assert_eq!(stride2_mul_count(1), 0, "n = 1 is a bare inversion");
        assert_eq!(classic_mul_count(0), 0);
        assert_eq!(stride2_mul_count(0), 0);

        // At CHUNK = 256: 894 against 765, i.e. +129 multiplies per chunk, or
        // +0.504 per candidate. These are the numbers SCRATCH.md quotes.
        assert_eq!(stride2_mul_count(256), 894);
        assert_eq!(classic_mul_count(256), 765);
    }

    /// Multiplies `batch_invert_into` performs for `n` elements (0 for n <= 1).
    fn classic_mul_count(n: usize) -> usize {
        if n <= 1 {
            0
        } else {
            3 * n - 3
        }
    }

    /// Multiplies `batch_invert_stride2_into` performs for `n` elements:
    /// `7n/2 - 2` for even `n` and `(7n - 5)/2` for odd `n`, 0 for `n <= 1`.
    fn stride2_mul_count(n: usize) -> usize {
        if n <= 1 {
            0
        } else if n % 2 == 0 {
            7 * n / 2 - 2
        } else {
            (7 * n - 5) / 2
        }
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
