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
/// changes; `COMBPARAMS.md` has the table-size / operation-count arithmetic
/// behind the default of 4.
///
/// **Changing this is a one-line change and nothing else**: every window size
/// the comb supports goes through the same `_w` code below, and the host test
/// suite exercises *all* of them (`comb_slot_exhaustive_for_w`,
/// `comb_matches_ladder_for_w`) whatever this is set to. That matters because
/// the tests used to run only the size the crate happened to be built with, so
/// a sweep of `T3V_COMB_W` could only ever be validated one point at a time --
/// and, worse, a *bad* point was only discovered at run time.
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

/// u64 words in one `Niels` (three field elements, no padding).
pub const NIELS_WORDS: usize = 15;

/// Bytes in one `Niels`.
pub const NIELS_BYTES: usize = NIELS_WORDS * 8;

/// Smallest window worth building. `W == 1` is the `[2^i]B` ladder this comb
/// replaced, one table entry per bit and 252 additions, so it is excluded by
/// name rather than by the arithmetic below.
pub const COMB_W_MIN: usize = 2;

/// Largest window the digit extractor can read.
///
/// The extractor takes `W` bits starting at a bit offset of at most 7, out of a
/// two-byte read covering bits 0..=15 of `(lo | hi << 8)`. The last bit it needs
/// is therefore at position `off + W - 1 <= 15`, i.e. `W <= 16 - off`, and the
/// worst case is `off == 7`: **`W <= 9`**. At 10 the extractor would silently
/// drop the top digit bit and hand back a slot in the wrong half of the window.
///
/// That is the largest value the *arithmetic* allows, and it is far above
/// anything usable: `W == 9` is 28 windows of 512 slots, a 1.6 MiB table, so the
/// L1 rule below rejects it long before the extractor's own limit is reached.
/// The bound is stated anyway because it is what makes the extractor's in-bounds
/// argument finite, and because `comb_slot_read_is_safe` checks it per window
/// rather than trusting it.
pub const COMB_W_MAX: usize = 9;

/// L1 data cache per SM, bytes. sm_86 has a 128 KiB unified L1/shared block.
///
/// **This is the binding constraint on the window size, and it is not a soft
/// one.** The table is read by all eight warps of a block, and the block is the
/// only one resident per SM at 255 registers/thread, so the eight warps march
/// through the windows roughly in lockstep: the table is pulled in *once* per SM
/// per launch and the other seven passes hit. That is a 1x cost. A table that
/// does not fit turns each of the eight passes into a fresh miss, so the cost
/// jumps 8x: at `W == 7` that is 4.1 MB of L2 traffic per SM per launch, about
/// 78 GB/s across the device, which does not fit in DRAM alongside everything
/// else the kernel streams. `COMBPARAMS.md` works through the numbers.
pub const COMB_L1_BYTES: usize = 128 * 1024;

/// Table slots per window for a window of `w` bits: the identity (digit 0) plus
/// every non-zero `w`-bit pattern.
pub const fn comb_slots(w: usize) -> usize {
    1 << w
}

/// Windows of `w` bits needed to cover `COMB_BITS`.
///
/// **Integer division, so this is a floor.** A `w` that does not divide
/// `COMB_BITS` leaves the top `COMB_BITS - comb_bits_covered(w)` bits
/// uncovered, and because clamping always sets bit 254 that is not a rounding
/// detail -- it is a wrong answer. `comb_w_ok` rejects such a `w` at compile
/// time, and `w5_would_drop_the_top_bits` demonstrates what it would cost.
pub const fn comb_windows(w: usize) -> usize {
    COMB_BITS / w
}

/// Scalar bits the windows of width `w` actually cover. Equal to `COMB_BITS`
/// exactly when `w` divides it; `comb_windows(w) * w < COMB_BITS` is the
/// dropped-bit count otherwise.
pub const fn comb_bits_covered(w: usize) -> usize {
    comb_windows(w) * w
}

/// Total comb table entries for a window of `w` bits.
pub const fn comb_len(w: usize) -> usize {
    comb_windows(w) * comb_slots(w)
}

/// Bytes of device memory the whole uploaded blob occupies for a window of
/// `w` bits: the comb table plus `8B` (`comb_len(w) + 1` Niels).
pub const fn comb_table_bytes(w: usize) -> usize {
    (comb_len(w) + 1) * NIELS_BYTES
}

/// Is `w` a window size this implementation can actually run?
///
/// The three conditions, in the order they matter:
///
/// 1. `w` divides `COMB_BITS`, or the top windows are silently dropped
///    (`w5_would_drop_the_top_bits` shows the resulting wrong point).
/// 2. `w <= COMB_L1_BYTES` worth of table, or every launch re-streams it eight
///    times over (see `COMB_L1_BYTES`).
/// 3. `COMB_W_MIN <= w <= COMB_W_MAX`, so the digit extractor reads in bounds.
///
/// This is a `const fn` so it can gate a `const _: () = assert!(...)` -- see
/// below -- and it is the single definition the runtime asserts and the tests
/// use, so the build-time rule and the tested rule cannot drift apart.
pub const fn comb_w_ok(w: usize) -> bool {
    w >= COMB_W_MIN
        && w <= COMB_W_MAX
        && COMB_BITS % w == 0
        && comb_table_bytes(w) <= COMB_L1_BYTES
}

// The three `T3V_COMB_W` preconditions, as *compile* errors.
//
// Before this, a bad window size was caught by `assert!`s in two run-time
// places, and the failure modes were both bad. The host assert fired inside the
// per-device search thread, so the binary started, searched nothing, and the
// only clue was a panic message on a detached thread. The in-kernel assert was
// better (LLVM folds a compile-time-false assert, so `render` compiled to four
// instructions that call `panic` and trap -- verified with
// `T3V_COMB_W=5 cargo build --release`, which produced a 73-line PTX and a
// 16-instruction cubin) but it still only surfaced when someone ran the
// binary. A build-time tunable should be validated by the build.
//
// Three asserts rather than one so each failure names the rule it broke.
const _: () = assert!(
    COMB_W >= COMB_W_MIN && COMB_W <= COMB_W_MAX,
    "T3V_COMB_W: the comb window must be in COMB_W_MIN..=COMB_W_MAX (2..=9); \
     see COMBPARAMS.md"
);
const _: () = assert!(
    COMB_BITS % COMB_W == 0,
    "T3V_COMB_W: the comb window must divide COMB_BITS (252), or the top \
     windows are silently dropped -- 252 % 5 == 2 drops bits 253 and 254, and \
     clamping always sets bit 254"
);
const _: () = assert!(
    comb_table_bytes(COMB_W) <= COMB_L1_BYTES,
    "T3V_COMB_W: the comb table must fit in the 128 KiB L1, or all eight warps \
     re-stream it every launch; W=5 already needs 181 KiB"
);

/// Windows of `COMB_W` bits covering `COMB_BITS`. `COMB_W` divides
/// `COMB_BITS` -- enforced above at compile time -- so no window runs off the
/// top of the scalar and `comb_bits_covered(COMB_W) == COMB_BITS`.
pub const COMB_WINDOWS: usize = comb_windows(COMB_W);

/// Table slots per window: the identity (digit 0) plus every non-zero
/// `COMB_W`-bit pattern.
pub const COMB_SLOTS: usize = comb_slots(COMB_W);

/// Total comb table entries.
pub const COMB_LEN: usize = comb_len(COMB_W);

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
///
/// **This is the only production entry point into the builder** and it refuses
/// any window size that `comb_w_ok` rejects, so the host can never upload a
/// table the device was not built to read.
pub fn build_comb_table(comb: &mut [Niels; COMB_LEN], eight: &mut Niels) {
    build_comb_table_w::<COMB_W>(comb, eight)
}

/// [`build_comb_table`] for any window size, checking that the size is one this
/// build can run.
pub fn build_comb_table_w<const W: usize>(comb: &mut [Niels], eight: &mut Niels) {
    assert!(
        comb_w_ok(W),
        "a {W}-bit comb window is not usable: it must be in {COMB_W_MIN}..={COMB_W_MAX}, \
         divide {COMB_BITS}, and fit its table in {COMB_L1_BYTES} bytes of L1 \
         (a {W}-bit window needs {})",
        comb_table_bytes(W)
    );
    build_comb_table_at(W, comb, eight);
}

/// Largest slot count [`build_comb_table_at`] will allocate scratch for, and
/// therefore the largest window that function accepts.
///
/// The scratch is three arrays of `2^w` points on the stack, so this is a
/// stack bound rather than a comb bound: 128 slots is 15 KiB, which is fine for
/// the host and irrelevant on the device, where the builder never runs. It is
/// also the largest width `comb_w_ok` can *accept*, since `W == 8` needs 256
/// slots and stops dividing 252 anyway.
pub const COMB_AT_MAX_SLOTS: usize = 128;

/// The table builder itself, for a window size given at run time.
///
/// **Performs no validation**: it will happily build the wrong table for a
/// `w` that does not tile the scalar, which is exactly what
/// `w5_would_drop_the_top_bits` uses it to demonstrate. Everything else in the
/// crate goes through [`build_comb_table_w`], which gates on `comb_w_ok`.
pub fn build_comb_table_at(w: usize, comb: &mut [Niels], eight: &mut Niels) {
    let slots = comb_slots(w);
    let windows = comb_windows(w);
    assert!(
        slots <= COMB_AT_MAX_SLOTS,
        "a {w}-bit comb window needs {slots} slots of scratch; this builder \
         allocates at most {}",
        COMB_AT_MAX_SLOTS
    );
    assert!(
        comb.len() >= windows * slots,
        "the comb table needs {} entries for a {w}-bit window, got {}",
        windows * slots,
        comb.len()
    );

    let zero = Ge {
        x: FE_ZERO,
        y: FE_ZERO,
        z: FE_ZERO,
        t: FE_ZERO,
    };
    let mut power = [zero; TABLE_LEN];
    build_power_table(&mut power);

    // Per-window scratch: the window's points, their Z coordinates, and the
    // prefix products `batch_invert_into` needs. Sized for the widest window
    // the builder accepts rather than for `w`, because `slots` is a run-time
    // value here and a `[T; slots]` repeat needs a constant.
    let mut points = [zero; COMB_AT_MAX_SLOTS];
    let mut zs = [FE_ONE; COMB_AT_MAX_SLOTS];
    let mut scratch = [FE_ONE; COMB_AT_MAX_SLOTS];

    for j in 0..windows {
        let base = COMB_FIRST_BIT + j * w;
        for m in 0..slots {
            let mut acc = ge_identity();
            let mut first = true;
            for b in 0..w {
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
        for m in 0..slots {
            comb[j * slots + m] = ge_to_niels(&points[m], &zs[m]);
        }
    }

    // The walk step, normalised the same way. `8B` used to be three doublings
    // of the base point in *every thread* of *every launch*; it is now built
    // once, here, and uploaded with the table.
    let eight_point = ge_eight();
    let zi = fe_invert(&eight_point.z);
    *eight = ge_to_niels(&eight_point, &zi);
}

/// The `W` scalar bits of window `j`, as a table slot index in
/// `0..2^W`.
///
/// # Why two bytes are always enough
///
/// A window is `W` bits at a bit offset of at most 7, so it spans at most
/// `W + 7` bits, and `W <= COMB_W_MAX == 9` keeps that inside the 16-bit read.
/// The high byte is real data exactly when the window does not start in the
/// scalar's last byte, i.e. when `byte + 1 <= 31`; in that case `byte + 1` is a
/// genuine index and the read is in bounds.
///
/// The only window that starts in byte 31 is the top one, and there the read
/// wraps to byte 0 -- which is *not* the next window's bits, so it has to be
/// masked away. It always is. The top window ends at bit 254, so
/// `base + W - 1 <= 254`; with `byte == 31` meaning `base >= 248` and
/// `off = base - 248`, that gives
///
/// ```text
///     off + W = (base - 248) + W <= (254 - W + 1 - 248) + W = 7 <= 8
/// ```
///
/// The wrapped byte lands at bit `8 - off` of the shifted value and the mask
/// keeps only bits `0..W`, so with `off + W <= 7` no bit of the wrap survives.
/// (The bound is 7, not 8; `comb_top_window_lands_on_bit_254` asserts the looser
/// `off + W <= 8` so the test does not depend on the tighter derivation.)
/// `comb_slot_read_is_safe` checks the general per-window condition rather than
/// trusting this argument to stay true if `COMB_FIRST_BIT` or `COMB_BITS` ever
/// move.
///
/// The consequence worth stating plainly: the top window is *not* truncated or
/// special-cased anywhere. It is an ordinary `W`-bit window whose last bit is
/// the scalar's last bit, and clamping's forced bit 254 lands in it like any
/// other. At the default `W == 4` that means window 62 covers bits 251..=254,
/// byte 31, offset 3, and its `hi` read wraps to byte 0 and is masked away --
/// which is the only place in the whole extractor where the read is not real
/// data, and the arithmetic above is why it still works.
#[inline(always)]
pub fn comb_slot_w<const W: usize>(scalar: &[u8; 32], j: usize) -> usize {
    let base = COMB_FIRST_BIT + j * W;
    let byte = base >> 3;
    let off = (base & 7) as u32;
    let lo = scalar[byte] as u32;
    let hi = scalar[(byte + 1) & 31] as u32;
    // `slot_mask::<W>()` rather than `(1 << W) - 1` so the mask is a compile
    // -time constant and the shift does not depend on W being in range.
    (((lo | (hi << 8)) >> off) & slot_mask::<W>()) as usize
}

/// The digit mask for a `W`-bit window: `2^W - 1`, as a `u32`.
#[inline(always)]
pub const fn slot_mask<const W: usize>() -> u32 {
    (1u32 << W) - 1
}

/// `[scalar]B` as a sum of precomputed window subset sums, for any window size.
///
/// `comb_windows(W)` point additions and **no doublings**, whatever the scalar
/// looks like -- the operation count is data independent, which is the entire
/// point (see the divergence note above `COMB_W`).
///
/// **Performs no `comb_w_ok` check**: it takes a plain slice so the tests can
/// drive it at a window size this build rejects, which is the only way to show
/// that such a size is wrong rather than merely discouraged. Production code
/// reaches it through [`scalarmult_base_comb`], which is instantiated at
/// `COMB_W`, and `COMB_W` is validated at compile time. What is checked here
/// is the one thing that would be a memory error rather than a wrong answer:
/// the table has to be long enough for the windows to index into.
pub fn scalarmult_base_comb_w<const W: usize>(scalar: &[u8; 32], comb: &[Niels]) -> Ge {
    let slots = comb_slots(W);
    let windows = comb_windows(W);
    assert!(
        comb.len() >= windows * slots,
        "a {W}-bit comb needs a table of {} entries, got {}",
        windows * slots,
        comb.len()
    );

    // The first window is added against the identity rather than assigned,
    // because the accumulator has to be a `Ge` and a `Niels` cannot be one.
    // `ge_madd` is unified, so that addition is exact -- and because
    // `ge_identity` is a compile-time constant, `ptxas` folds it: LLVM peels
    // window 0 out of the loop and, in the peeled copy, the three multiplies
    // against the identity's known `1, 1, 0` collapse. Measured in the
    // generated PTX, the peeled copy has 55 `mul` instructions against the
    // loop body's 376, so window 0 costs about one field multiply instead of
    // seven. At the default `W == 4` that is 62 * 7 + 1 = 435 multiplies per
    // scalar multiplication instead of 441.
    let mut r = ge_madd(&ge_identity(), &comb[comb_slot_w::<W>(scalar, 0)]);
    for j in 1..windows {
        r = ge_madd(&r, &comb[j * slots + comb_slot_w::<W>(scalar, j)]);
    }
    r
}

/// [`scalarmult_base_comb_w`] at the window size this build uses, against the
/// fixed-size table the device is handed.
///
/// `comb_windows(W)` point additions and **no doublings**, whatever the scalar
/// looks like. At the default `COMB_W == 4` that is 63 additions against the
/// ladder's ~251 per warp.
pub fn scalarmult_base_comb(scalar: &[u8; 32], comb: &[Niels; COMB_LEN]) -> Ge {
    scalarmult_base_comb_w::<COMB_W>(scalar, comb)
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
    fn comb_matches_power_ladder_for_w<const W: usize>() {
        let power = power_table();
        let mut comb: Vec<Niels> = vec![unsafe { core::mem::zeroed() }; comb_len(W)];
        let mut eight: Niels = unsafe { core::mem::zeroed() };
        // Deliberately the unvalidated builder: 6 and 7 are mathematically
        // correct but rejected by `comb_w_ok` on table size, and this test
        // exists to show the maths holds there too.
        build_comb_table_at(W, &mut comb, &mut eight);

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

            let got = scalarmult_base_comb_w::<W>(&scalar, &comb);
            let want = scalarmult_base_table(&scalar, &power);
            assert_eq!(
                hex(&compress(&got)),
                hex(&compress(&want)),
                "W={W} case {i} scalar {}",
                hex(&scalar)
            );
        }
    }

    /// The same cross-check at the window size this build was compiled with.
    #[test]
    fn comb_matches_power_ladder() {
        comb_matches_power_ladder_for_w::<COMB_W>();
    }

    /// And at every window size the comb claims to support. Together with
    /// `comb_slot_reads_the_right_bits_for_every_window` this makes the whole
    /// comb path -- extractor, table layout, Niels normalisation, the
    /// identity slot -- W-parameterised in the test suite, not just in the
    /// kernel.
    ///
    /// 6 and 7 are here even though `comb_w_ok` rejects them, because the
    /// *maths* is right at those widths and only the table size is not. That
    /// is worth pinning down: if a future change made a wide window cheaper
    /// (a shared-memory table, say), these tests are already there.
    #[test]
    fn comb_matches_power_ladder_for_every_window() {
        comb_matches_power_ladder_for_w::<2>();
        comb_matches_power_ladder_for_w::<3>();
        comb_matches_power_ladder_for_w::<4>();
        comb_matches_power_ladder_for_w::<6>();
        comb_matches_power_ladder_for_w::<7>();
    }

    /// The RFC vectors through the comb at several window widths. Wider than
    /// the ladder comparison in value: these are *published* keys, so a match
    /// pins the whole path to the specification and not merely to itself.
    #[test]
    fn comb_rfc8032_vectors_for_every_window() {
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
        // Build the three clamped scalars once.
        let mut scalars: Vec<[u8; 32]> = Vec::new();
        for (seed_hex, _) in cases.iter() {
            let seed: Vec<u8> = unhex(seed_hex);
            let expanded = sha512::sha512(&seed);
            let mut scalar = [0u8; 32];
            scalar.copy_from_slice(&expanded[..32]);
            scalar[0] &= 248;
            scalar[31] &= 63;
            scalar[31] |= 64;
            scalars.push(scalar);
        }

        // Const-generic, so the window size has to be a literal rather than a
        // loop variable: the extractor, the table layout and the loop bound all
        // specialise on it, which is exactly why a width is a compile-time
        // constant throughout the kernel.
        macro_rules! at_w {
            ($w:literal) => {{
                let mut comb: Vec<Niels> = vec![unsafe { core::mem::zeroed() }; comb_len($w)];
                let mut eight: Niels = unsafe { core::mem::zeroed() };
                build_comb_table_at($w, &mut comb, &mut eight);
                for (i, scalar) in scalars.iter().enumerate() {
                    let got = scalarmult_base_comb_w::<$w>(scalar, &comb);
                    assert_eq!(hex(&compress(&got)), cases[i].1, "W={} vector {}", $w, i);
                }
            }};
        }
        at_w!(2);
        at_w!(3);
        at_w!(4);
        at_w!(6);
        at_w!(7);
    }

    /// Every comb slot, for every window, must be the subset sum it claims to
    /// be. Checking the table directly catches a mis-indexed build that the
    /// scalar-multiply comparison could hide behind a symmetric mistake.
    ///
    /// This one is quadratic in the table size, so it runs at the widths whose
    /// tables are small enough to be cheap: 1008 slots at W=4, 672 at W=3, 504
    /// at W=2. At W=6 it would be 2688 and at W=7 4608, and `comb_matches_-
    /// power_ladder_for_every_window` already covers those widths end to end.
    #[test]
    fn comb_slots_are_subset_sums() {
        for w in [2usize, 3, 4] {
            let power = power_table();
            let mut comb: Vec<Niels> = vec![unsafe { core::mem::zeroed() }; comb_len(w)];
            let mut eight: Niels = unsafe { core::mem::zeroed() };
            build_comb_table_at(w, &mut comb, &mut eight);
            for j in 0..comb_windows(w) {
                let base = COMB_FIRST_BIT + j * w;
                for m in 0..(1usize << w) {
                    // Rebuild the slot the slow way, straight from the power table.
                    let mut want = ge_identity();
                    let mut first = true;
                    for b in 0..w {
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
                        hex(&compress(&ge_madd(&ge_identity(), &comb[j * (1 << w) + m]))),
                        hex(&compress(&want)),
                        "W={w} window {j} slot {m}"
                    );
                }
            }
        }
    }

    /// The identity slot must be the identity, at every width. A zero digit has
    /// to add nothing at all, which is what lets the window loop run with no
    /// branch -- and a Niels identity is only valid because `ge_madd` is a
    /// unified formula.
    #[test]
    fn comb_slot_zero_is_the_identity_at_every_window() {
        for w in [2usize, 3, 4, 6, 7] {
            let mut comb: Vec<Niels> = vec![unsafe { core::mem::zeroed() }; comb_len(w)];
            let mut eight: Niels = unsafe { core::mem::zeroed() };
            build_comb_table_at(w, &mut comb, &mut eight);
            for j in 0..comb_windows(w) {
                let zero = comb[j * (1 << w)];
                // Adding it to a point must be a no-op.
                let mut p = ge_base();
                for _ in 0..(j % 7) {
                    p = ge_add(&p, &ge_eight());
                }
                let got = ge_madd(&p, &zero);
                assert_eq!(
                    hex(&compress(&got)),
                    hex(&compress(&p)),
                    "W={w} window {j} slot 0 must be the identity"
                );
            }
        }
    }

    /// The digits of window `j` taken straight from the scalar, one bit at a
    /// time. This is the deliberately naive reading that `comb_slot_w` has to
    /// agree with, and it is written in terms of `W` so the same oracle serves
    /// every window size.
    fn comb_slot_reference(scalar: &[u8; 32], w: usize, j: usize) -> usize {
        let base = COMB_FIRST_BIT + j * w;
        (0..w).fold(0usize, |acc, b| {
            acc | (((scalar[(base + b) >> 3] >> ((base + b) & 7)) & 1) as usize) << b
        })
    }

    /// A spread of scalars that between them make the two byte reads interesting:
    /// a clamped one, an unclamped one, one with every bit set (so the high
    /// byte is `0xFF` and any spill into it shows up), and one with only the
    /// forced top bit set.
    fn extractor_scalars() -> Vec<[u8; 32]> {
        let mut clamped = [0u8; 32];
        for (i, b) in clamped.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(37).wrapping_add(11);
        }
        clamped[0] &= 248;
        clamped[31] &= 63;
        clamped[31] |= 64;

        let mut unclamped = [0u8; 32];
        for (i, b) in unclamped.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(101).wrapping_add(7);
        }

        let mut top_only = [0u8; 32];
        top_only[0] = 0x08;
        top_only[31] = 0x40;

        // Bit 255 set as well. Clamping forbids it, but the extractor must not
        // read past bit 254 into it: the reference oracle is the same either
        // way, so this scalar is a direct check that the top window stops at
        // 254.
        let mut over = clamped;
        over[31] |= 0x80;

        vec![clamped, unclamped, [0xFF; 32], top_only, over]
    }

    /// The digit extractor must return exactly the bits the window spans, for
    /// every window, including the two ends of the scalar where the 16-bit read
    /// runs off the end.
    fn comb_slot_reads_the_right_bits_for_w<const W: usize>() {
        for scalar in extractor_scalars().iter() {
            for j in 0..comb_windows(W) {
                assert_eq!(
                    comb_slot_w::<W>(scalar, j),
                    comb_slot_reference(scalar, W, j),
                    "W={W} window {j} scalar {}",
                    hex(scalar)
                );
            }
        }
    }

    /// The same, at the window size this build was compiled with.
    #[test]
    fn comb_slot_reads_the_right_bits() {
        comb_slot_reads_the_right_bits_for_w::<COMB_W>();
    }

    /// Every window size the comb claims to support, on the same extractor
    /// check. This is the test that was missing: the digit extractor, the
    /// window count and the table layout all depend on `COMB_W`, and before
    /// this the suite only ever exercised the one value the crate happened to
    /// be built with, so a sweep could validate a window size only by
    /// rebuilding and re-running.
    #[test]
    fn comb_slot_reads_the_right_bits_for_every_window() {
        comb_slot_reads_the_right_bits_for_w::<2>();
        comb_slot_reads_the_right_bits_for_w::<3>();
        comb_slot_reads_the_right_bits_for_w::<4>();
        comb_slot_reads_the_right_bits_for_w::<6>();
        comb_slot_reads_the_right_bits_for_w::<7>();
    }

    /// The top window is where an off-by-one would hide, because that is the
    /// only window whose two-byte read wraps. Spell out the two properties
    /// separately so a failure says which one broke.
    ///
    /// 1. *Coverage*: the last window ends at the scalar's last usable bit
    ///    (254) -- not one bit short of it, and not past it.
    /// 2. *Wrapping*: if the top window's high byte is a wrap onto byte 0, the
    ///    mask must discard all of it. `off + W <= 8` is the condition; see
    ///    `comb_slot_w` for the derivation.
    #[test]
    fn comb_top_window_lands_on_bit_254() {
        for w in [2usize, 3, 4, 6, 7] {
            assert_eq!(
                comb_bits_covered(w),
                COMB_BITS,
                "W={w} must cover every comb bit",
                w = w
            );

            let last_base = COMB_FIRST_BIT + (comb_windows(w) - 1) * w;
            let last_bit = last_base + w - 1;
            assert_eq!(
                last_bit,
                254,
                "W={w} top window must end at bit 254",
                w = w
            );

            // The wrapped read is masked away.
            assert!(
                last_base >= 248,
                "W={w} top window should start in byte 31, the only byte whose \
                 successor wraps",
                w = w
            );
            let off = last_base & 7;
            assert!(
                off + w <= 8,
                "W={w} top window wraps onto byte 0 and its bits survive the \
                 mask (off={off}, w={w})",
                off = off, w = w
            );

            // And no window may read past bit 254 at all.
            for j in 0..comb_windows(w) {
                let base = COMB_FIRST_BIT + j * w;
                assert!(
                    base + w - 1 <= 254,
                    "W={w} window {j} reads past bit 254",
                    w = w, j = j
                );
                let byte = base >> 3;
                assert!(
                    byte + 1 <= 31 || (base & 7) + w <= 8,
                    "W={w} window {j} reads out of bounds",
                    w = w, j = j
                );
            }
        }
    }

    /// Every window size in the documented range, so the extractor and the
    /// table agree, and the digit extraction is exhaustively correct.
    #[test]
    fn comb_slot_read_is_safe() {
        for w in COMB_W_MIN..=COMB_W_MAX {
            for j in 0..comb_windows(w) {
                let base = COMB_FIRST_BIT + j * w;
                let byte = base >> 3;
                let off = base & 7;
                // Two cases, and only two: either `byte + 1` is a real index,
                // or it wraps and the mask has to save us.
                if byte + 1 <= 31 {
                    assert!(byte + 1 <= 31, "W={} window {}", w, j);
                } else {
                    assert!(
                        off + w <= 8,
                        "W={} window {} wraps and leaks (off={})",
                        w, j, off
                    );
                }
                // The two-byte read covers bits 0..=15; the window needs
                // `off..=off + W - 1`. At W=9 and off=7 that is bit 15, the last
                // bit available -- exactly tight, and the reason
                // `COMB_W_MAX` is 9.
                assert!(
                    off + w <= 16,
                    "W={} window {} reads past bit 15 of its two-byte read",
                    w,
                    j
                );
            }
        }
    }

    /// `comb_w_ok` is the single definition of "usable". Check it against the
    /// two properties it is meant to enforce, and pin down the sizes on both
    /// sides of each boundary so a change to the predicate has to be deliberate.
    #[test]
    fn comb_w_ok_matches_its_own_definition() {
        // Divides 252, and its table fits in L1.
        for w in [2usize, 3, 4] {
            assert!(comb_w_ok(w), "W={} should be usable", w);
        }
        // Divides 252 (36 windows) but the table is 540 KiB, 4.1x L1.
        assert!(!comb_w_ok(7), "W=7 cannot fit its table in L1");
        // Does not divide 252, and the table does not fit either.
        assert!(!comb_w_ok(5), "W=5 neither tiles 252 nor fits L1");
        // Out of range at the bottom.
        assert!(!comb_w_ok(0), "W=0 is meaningless");
        assert!(!comb_w_ok(1), "W=1 is the ladder, not a comb");
        // Out of range at the top, and does not tile 252 either.
        assert!(!comb_w_ok(8), "W=8 does not divide 252");
    }

    /// The point of the whole exercise: a window size that does not divide 252
    /// is not a slightly-wrong comb, it is a *different point*.
    ///
    /// `COMB_BITS / 5 == 50` windows of 5 bits cover 250 of the 252 bits, so
    /// integer division drops bits 253 and 254 -- and clamping always sets bit
    /// 254, so it is dropped on *every single scalar*. This is built with the
    /// unvalidated builder to make the failure concrete rather than
    /// hypothetical: a build-time assert on `comb_w_ok` says "don't do this",
    /// and this says what "this" costs.
    #[test]
    fn w5_would_drop_the_top_bits() {
        assert_eq!(COMB_BITS % 5, 2, "the premise: 5 does not tile 252");
        assert!(!comb_w_ok(5), "and so it is rejected");

        // Build the W=5 table anyway. Two windows short of a full comb.
        let mut comb: Vec<Niels> = vec![unsafe { core::mem::zeroed() }; comb_len(5)];
        let mut eight: Niels = unsafe { core::mem::zeroed() };
        build_comb_table_at(5, &mut comb, &mut eight);

        // A clamped scalar: bits 3..=254 are all legal, bit 254 forced.
        let mut scalar = [0u8; 32];
        for (i, b) in scalar.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(53).wrapping_add(3);
        }
        scalar[0] &= 248;
        scalar[31] &= 63;
        scalar[31] |= 64;
        assert_eq!(scalar[31] & 0x40, 0x40, "bit 254 is set, as clamping demands");

        let power = power_table();
        let got = scalarmult_base_comb_w::<5>(&scalar, &comb);
        let want = scalarmult_base_table(&scalar, &power);
        assert_ne!(
            hex(&compress(&got)),
            hex(&compress(&want)),
            "a 5-bit comb must not agree with the ladder -- that is the point"
        );

        // Pin down *how* it is wrong: clearing bits 253 and 254 from the scalar
        // has to reproduce the W=5 result exactly. That makes it a dropped-bit
        // bug rather than a coincidence.
        let mut truncated = scalar;
        truncated[31] &= 0x3F; // clear bit 254
        truncated[31] &= !0x20u8; // clear bit 253
        assert_eq!(
            hex(&compress(&scalarmult_base_table(&truncated, &power))),
            hex(&compress(&got)),
            "the W=5 comb should equal the ladder over bits 3..=252"
        );
    }

    /// The window count must be exact for every usable `W`, and the top window
    /// must not run off the end of the scalar.
    #[test]
    fn comb_window_count_is_exact() {
        for w in [2usize, 3, 4] {
            assert_eq!(
                comb_windows(w) * w,
                COMB_BITS,
                "W={w} must tile the comb range exactly",
                w = w
            );
            assert_eq!(
                comb_len(w),
                comb_windows(w) * (1 << w),
                "W={w} table length",
                w = w
            );
            assert_eq!(
                comb_len(w) * NIELS_BYTES + NIELS_BYTES,
                comb_table_bytes(w),
                "W={w} uploaded bytes",
                w = w
            );
            assert!(
                comb_table_bytes(w) <= COMB_L1_BYTES,
                "W={w} table must fit L1",
                w = w
            );
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

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{:02x}", x)).collect()
    }

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
            .collect()
    }
}
