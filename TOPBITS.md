# TOPBITS: partial compression, so a prefix test never sees 255 bits

**Branch** `agent/topbits`, base `b0a8695`. Host-side tests:
`cargo +nightly test --manifest-path core/Cargo.toml --release` -> **46 passed**
(36 pre-existing, 10 added). No GPU was touched; every number below is static
analysis of the emitted PTX and of the SASS `ptxas` produces from it.

## 1. What this changes

`BatchState::check` used to compress a candidate all the way to 32 bytes and let
`BytePrefix::matches` pick the first eight out of them:

```rust
let x = fe_mul(&p.x, zi);          // only for the sign bit, at output bit 255
let y = fe_mul(&p.y, zi);
let mut out = fe_to_bytes(&y);     // ~32-step serial bit walk, all 255 bits
out[31] ^= (fe_is_negative(&x) as u8) << 7;
out                                      // then ~30 of 32 bytes are discarded
```

It now computes the 64 bits a prefix can actually observe and stops:

```rust
let word = ge_prefix_word(&self.cur, z_inv);   // y = p.y * zi, then
                                                  // word = limb0 | (limb1 << 51)
... if bp.matches_word(word) { ... }            // 2 masked 64-bit compares
```

Three separate costs disappear, and none of them can change a match decision:

1. **`fe_mul(&p.x, zi)`** -- one full 5x5 schoolbook field multiply. Its only
   output is the parity of `x`, which is written to bit 255 of the key. A Tor v3
   address is `base32(pubkey || checksum || 0x03)`, and RFC 4648 reads the byte
   string most-significant-bit first, so the address's first characters come
   from the *first* bytes of the key. Bit 255 is the 32nd byte, and 12 base32
   characters is 60 bits: nothing a user can ask for reaches it.
2. **The `fe_to_bytes` bit walk** -- five 51-bit limbs walked byte by byte with a
   data-dependent `take = space.min(remaining)` and a running bit counter, ~32
   dependent iterations to produce 32 bytes. Replaced by two limb reads and an
   OR. This is *not* the abandoned accumulator (which overflowed because
   `limb << bits` with a **variable** `bits > 13` drops a limb's high bits): the
   shift here is the constant 51, so the only bits that can leave the `u64` are
   bits 101..127 of the 102-bit limb 1, which do not exist. The truncation keeps
   exactly bits 51..63, which is what is wanted.
3. **The `fe_reduce(&carry())` inside `fe_to_bytes`** -- and this one needs an
   argument rather than an assertion. The value is *not* left unreduced:
   `fe_mul` ends in `carry_product`, which ends in `fe_reduce`, so its output is
   already the canonical representative in `[0, p)`. Re-reducing is a no-op on
   it. Nothing else in the tree would notice if that stopped being true -- a
   product left at `2^255 - 1` still multiplies and inverts correctly, and
   `fe_to_bytes` would quietly canonicalise it -- so the invariant is now pinned
   by `field::tests::mul_output_is_already_canonical`, which asserts
   `fe_reduce(carry(x)) == x` over the same adversarial operand shapes the
   existing reduction tests use.

The masks and the values the compares need are precomputed **on the host**,
once per prefix, by `core::prefix_word_masks` (`src/gpu.rs` calls it from
`BytePrefixOwned::from_str`). The kernel therefore does no prefix-byte loading,
no `memcmp`, no variable shift, and no reduce: per candidate it is one
`fe_mul`, one OR, two ANDs and two compares.

## 2. Measurements

Reproduce with:

```sh
cargo +nightly rustc --manifest-path core/Cargo.toml --target nvptx64-nvidia-cuda \
    --target-dir core/target --crate-type cdylib --release
# -> core/target/nvptx64-nvidia-cuda/release/tor_v3_vanity_core.ptx
/opt/cuda/bin/ptxas -arch=sm_86 -v render.ptx -o render.cubin     # registers, stack
/opt/cuda/bin/nvdisasm -c render.cubin                              # SASS
```

`ptxas` 13.4, `-arch=sm_86` (the PTX itself says `.target sm_70`, which ptxas 13.4
no longer accepts -- CUDA 13 dropped Volta -- so sm_86, the RTX 3050's arch, is
the only target available for static analysis here; and the driver JITs for the
real device anyway). The register and stack figures are stable across `-O2` and
`-O3`.

| | base | this branch | delta |
|---|---:|---:|---:|
| `render`, PTX instructions | 9962 | 9537 | **-425 (-4.3%)** |
| `render`, SASS instructions | 27016 | 26360 | **-656 (-2.4%)** |
| per-candidate loop (pass 2: `check` + `advance`), SASS | 4621 | 3992 | **-629 (-13.6%)** |
| -- of which compress+match, SASS | 1017 | 391 | **-626 (-61.5%)** |
| -- of which ring write + `ge_madd`, SASS | 3603 | 3600 | -3 |
| pass-1 loop (collect Z), SASS | 3485 | 3487 | +2 (untouched) |
| **registers/thread** | **255** | **168** | **-87 (-34%)** |
| **local bytes/thread** (stack frame) | **20896** | **20864** | **-32** |
| spill stores / loads | 0 / 0 | 0 / 0 | - |
| PTX local depot | 20896 | 20864 | -32 |

Per-candidate compression region, opcode by opcode (SASS):

| op | base | new | delta | | op | base | new | delta |
|---|---:|---:|---:|---|---|---:|---:|---:|
| IMAD | 412 | 191 | -221 | | LOP3 | 43 | 16 | -27 |
| IADD3 | 232 | 102 | -130 | | STL (local store) | 38 | 0 | -38 |
| ISETP | 93 | 24 | -69 | | LDL (local load) | 22 | 10 | -12 |
| SHF | 48 | 11 | -37 | | CALL (`memcmp`) | 1 | 0 | -1 |
| SEL | 45 | 9 | -36 | | BRA | 23 | 6 | -17 |
| IMNMX | 10 | 0 | -10 | | ULDC | 7 | 1 | -6 |
| | | | | | **total** | **1017** | **391** | **-626** |

Two things in that table are worth calling out. The `STL`/`LDL` rows: the old
`out[byte] |= ...` in `fe_to_bytes` cannot be registerised (the index is
data-dependent), so the 32-byte key was being **spilled to the local depot and
read back on every single candidate** -- that is the entire 32-byte drop in the
stack frame, and the reason `st.local.b8` falls 164 -> 126 over the whole kernel.
The `CALL` row: `data.starts_with(...)` was a real `memcmp` call.

### Field multiplies removed: exactly 1

Counting `mul` in the compress+match region of the PTX: **104 -> 54**. For
calibration, the pass-1 loop (one `ge_madd`, i.e. 7 `fe_mul`) has 374 `mul`, and
two `fe_mul` that share the `zi` multiplicand cost 101 rather than 106 because
the five `b[i] * 19` prescalings are common subexpressions. So 104 -> 54 is
exactly one field multiply, with the shared-prescaling saving accounted for.
Confirmed independently in SASS: IMAD 412 -> 191, and one `fe_mul` is ~191 IMAD
on sm_86.

### What that implies for keys/s (estimate, not measured)

Per candidate the kernel executes, in SASS instructions:

| | base | new |
|---|---:|---:|
| pass 1 (`ge_madd`, collect Z) | 3485 | 3487 |
| pass 2 `advance` (`ge_madd`) | ~3400 | ~3400 |
| pass 2 `check` | 1017 | 391 |
| amortised batch inversion (3 `fe_mul`/candidate + 1/256 of an `fe_invert`) | ~1300 | ~1300 |
| **total** | **~9200** | **~8580** |

That is **-6.8% instructions per candidate** (`fe_mul` ~350 SASS, `fe_sq`
~250, from the two measured regions). Against the measured 149.4M keys/s with
matching and 380M without (6.69 ns vs 2.63 ns per key, so matching costs
4.06 ns/key), a 6.8% cut in the matching path's work is worth roughly **+7%
keys/s, ~160M**. I did not run this on the GPU, so treat it as arithmetic on the
static counts rather than a result. The direction is the one CONTEXT.md says to
trust: instruction count, not occupancy.

### Occupancy: unchanged, despite the register drop

255 -> 168 registers/thread is a 34% cut, but it does **not** buy warps here.
With `BLOCK = 256`, a block needs `256 * regs` of the SM's 65536 registers: at
255 that is 65280 (exactly one block resident) and at 168 it is 43008 (still
only one, because a second would need 86016). So 8 warps/SM before and after.
`gpu::geometry` picks `threads = min(..., 65536/168 = 390, BLOCK = 256) = 256`
either way.

It does open a door: at 168 registers a **128-thread** block would fit three per
SM (3 * 128 * 168 = 64512), i.e. 12 warps/SM. `T3V_BLOCK=128` is a build-time
knob, so that is a sweep someone can run without touching code. CONTEXT.md's
`fe_sq` experience says instruction count beat occupancy, so I would sweep it
*after* A/B-ing this, not instead of it.

## 3. Why the fast path is equivalent

Write `j = last_byte_idx`, `k = 8j`, `m = last_byte_mask`, and `slice` for the
decoded prefix bytes. The reference accepts a candidate when

```text
data[0 .. j] == slice[0 .. j]      &&      data[j] & m == slice[j]
```

`data` is the little-endian encoding of `y`, so `data[i]` is bits `8i .. 8i+8` of
`y`, and -- this is the whole argument -- for `j <= 7` every byte either side
touches is `data[i] = (word >> 8i) & 0xFF` where
`word = y[0] | (y[1] << 51)` is the low 64 bits of `y`. Substituting:

```text
(1)  (word & (2^k - 1))      == LE_pack(slice[0 .. j])
(2)  ((word >> k) & 0xFF) & m == slice[j]
```

(1) is `(word & full_mask) == full_val` with `full_mask = 2^k - 1` and
`full_val` the little-endian packing of `slice[0 .. j]`. Both are known on the
host.

(2) is where the care is needed, twice over.

* `(word >> k) & 0xFF & m` is `(word >> k) & m` because `m < 256`. Multiplying
  the whole equation by `2^k` gives `(word & (m << k)) == (slice[j] << k)`, with
  no loss: `k <= 56` and `slice[j] < 256`, so the left shift is exact for both
  sides. That is `(word & top_mask) == top_val`, again host-side constants.
* The right-hand side is the **whole** prefix byte, not the masked byte. For a
  base32 prefix `slice[j] & !m == 0` always holds (the bits past the prefix are
  `base32::decode`'s "aa" padding, i.e. zero), so masking the right-hand side
  too would *also* be equivalent on real prefixes -- but it would stop being
  equivalent on any other input, and "equivalent on the inputs that occur" is a
  much weaker claim than the identity above. Shifting `slice[j]` up rather than
  masking it keeps the equivalence unconditional, for arbitrary bytes.

So `matches_word` is not an approximation of `matches`; it is `matches` with the
byte indexing folded into the masks. The masks are pre-shifted so the kernel
does no shifts and no byte loads at all.

Three edge cases that fall out of the identity rather than being special-cased:

* **`k = 0`** (a one-character prefix, where the host's `5*len/8 - 1`
  underflows and wraps back to 0): `full_mask = 0`, `full_val = 0`, so (1) is
  vacuously true and (2) does the whole job. `prefix_word_masks` branches on
  `k == 0` to keep `1u64 << 64` out of the expression.
* **`m == 0`**, which happens when `5L % 8 == 0`, i.e. `L == 8` in the supported
  range. Then (2) is `0 == slice[j]`: the reference accepts only prefixes whose
  last byte is zero. That is a **pre-existing bug in the host's mask arithmetic**
  (see §5) and the fast path reproduces it exactly, because `top_val` keeps the
  out-of-mask bits. The two paths therefore cannot disagree about which keys a
  search finds, which is the property the A/B needs.
* **`j > 7`**: the observable bits would run past bit 63, i.e. a prefix longer
  than 12 base32 characters. There is no correct `u64` for that, so
  `prefix_word_masks` asserts (and the host gets a clear message before
  anything is uploaded). `--bench-prefix` already capped at 12; a search for 13
  characters now fails fast instead of running. See §5.

### The tests, and evidence they bite

| test | what it pins |
|---|---|
| `tests::matches_word_is_equivalent_to_matches` | every `j` in 0..=7 x 11 masks x 3 prefix-byte shapes, at the matching word, at all 64 single-bit flips, and at an unrelated word. Feeds **random** prefix bytes, including ones with bits set outside the mask, which no real prefix has -- an implementation that masked the right-hand side would pass on real prefixes and fail here. |
| `tests::word_masks_cover_exactly_the_compared_bits` | the mask union is exactly the compared set, the two halves never overlap, no stray value bits. |
| `field::tests::prefix_word_is_the_low_bytes_of_the_encoding` | the word is the low 8 bytes of `fe_to_bytes`, over the adversarial limb shapes plus a 2000-value sweep. |
| `field::tests::mul_output_is_already_canonical` | the reduction `ge_prefix_word` relies on and does not repeat. |
| `group::tests::fast_path_agrees_with_the_byte_path` | both paths on the same point with the same `zi`, for **every prefix length 1..=12**, over 66 points (8 real, 56 pseudorandom limbs, the identity, a maximal-limb `y`) x 4 prefix shapes. |
| `group::tests::a_matching_key_matches_on_the_fast_path` | a key built from its own address bits matches under **both** paths. This is the test for the failure mode that looks like a slow GPU: an inverted comparison, a mask on the wrong side, or an off-by-one all leave the two paths agreeing on every *rejection* while rejecting every *match*, so a search returns nothing and still reports a plausible rate. The random-prefix sweep cannot see that. At `L == 8` it keeps looking for a key the degenerate reference can match (1 in 256). |
| `kernel::tests::check_captures_exactly_what_the_byte_path_would` | end to end: the real `check`, in the real chunked walk (one batch inversion per chunk, `check` against each `Z^-1`), against a real `KernelParams`/`BytePrefix` on host memory, comparing the **sets** of captured expanded keys with the byte-wise verdicts on the same candidates. Sets, not counts, so a false positive and a false negative in one launch cannot cancel. 12 lengths x 3 prefix shapes. |
| `tests::byte_prefix_layout_is_flat` | `#[repr(C)]` offsets and `size_of`, since the host uploads this struct by value and the four masks are new. |

Mutation-tested, to show the tests are not vacuous. Each of these was applied,
run, and reverted:

| mutation | tests that failed |
|---|---|
| swap `full_val` and `top_val` in `matches_word` | `matches_word_is_equivalent_to_matches`, `fast_path_agrees_with_the_byte_path`, `a_matching_key_matches_on_the_fast_path`, `check_captures_exactly_what_the_byte_path_would` |
| drop the second compare | `matches_word_is_equivalent_to_matches`, `fast_path_agrees_with_the_byte_path`, `check_captures_exactly_what_the_byte_path_would` |
| mask the right-hand side too (`top_val & top_mask`) | the same three |
| `fe_prefix_word`: shift limb 1 by 50 instead of 51 | `prefix_word_is_the_low_bytes_of_the_encoding` (field **and** group), plus the three equivalence tests |

The derivation of the prefix bytes used by the group and kernel tests was
cross-checked against the real `base32` crate in a scratch program: for every
length 1..=12 and 2000 random prefixes, `base32::decode(s + "aa")` equals "the
key's first `5*len` address bits, everything else cleared", and
`last_byte_idx` / `last_byte_mask` come out as `prefix_shape` computes them.
Worth knowing: `base32` 0.4 **truncates** rather than rounds up, so
`byte_prefix_len` is `(5*(len+2))/8`, which is 1 for a one-character prefix.

## 4. Where an early exit is, and is not

`matches_word` is `(word & full_mask) == full_val && (word & top_mask) == top_val`,
so the second compare is skipped when the first fails -- a candidate is rejected
on the whole-byte half of the prefix without touching the partial byte.

**On the GPU that early exit is worth nothing**, and it is worth being precise
about why. A warp only skips the second compare when *no* lane in it has passed
the first. With one prefix of `n` characters requested, a lane passes with
probability `2^-n`, so all 32 lanes fail with probability `(1 - 2^-n)^32`, i.e.
`2^(-32n)`: for `n = 1` that is 1 in 4 billion. In practice both compares
execute every candidate and LLVM may emit them with or without a branch (it
emitted 6 BRA in the compress region, down from 23). They are kept as two
masked compares because that is the clearest statement of the identity in §3,
not because the branch pays.

**The early exit that does exist is at the prefix level, not the candidate
level.** `check` already returns on the first prefix that matches, so with `N`
prefixes the compare loop costs `N` masked compares against a word computed
once, and a match short-circuits the rest. That is why `ge_prefix_word` is split
from `matches_word` instead of being one function per prefix.

**The early exit that is impossible**: rejecting before the multiply. `y =
p.y * zi` has to be computed before any of its bits are known -- the low 64 bits
of a product depend on every limb of both operands -- and the batch inversion
that produced `zi` is amortised over the whole chunk, so there is nothing to
save by deferring. The one field multiply that remains is irreducible.

**Where the limb structure could still be exploited, and is not**: for
`L <= 10` the entire comparison lives in limb 0 (bits 0..50), and only
`L = 11, 12` touch limb 1. A version that compared limb 0 alone for short
prefixes would save the `limb1 << 51` (one SHF) and nothing else, because the
shift is hoisted out of the prefix loop and shared by every prefix. It would
also need a `j`-dependent branch, which on a GPU is a divergence hazard for a
2-instruction saving. Not worth it; the SHF is 1 of 391.

## 5. Findings for whoever picks this up next

**1. An 8-character prefix is broken in the host's mask arithmetic (pre-existing,
not introduced here, deliberately not fixed).** `last_byte_mask` is
`((1 << n_bits) - 1) << (8 - n_bits)`, which is 0 when `n_bits == 0`, and a zero
mask makes the reference test degenerate to `0 == slice[last_byte_idx]`. For
`L = 8`, `5L = 40` bits is a whole number of bytes, so `last_byte_idx = 4` and
`slice[4]` holds characters 7 and 8. The filter the kernel actually applies is
therefore

```text
key[0..4] == slice[0..4]      &&      slice[4] == 0
```

-- 34 bits checked against the request, and the last 6 bits of it (character 7's
low three bits and all of character 8) required to be **zero**. So an
8-character search only ever matches anything at all when the *requested*
characters 7 and 8 are zero-valued, i.e. `'a'`, or `'a'`/`'b'` for the 7th; for
any other 8-character request it captures nothing, forever, while looking like a
search that is merely unlucky. When the request does pass that test, it
over-captures by 256x (one key in 256 has a zero byte 4) and the host throws the
false ones away. The one-character case has the mirror-image bug:
`5 * s.len() / 8 - 1` underflows for `len == 1` and only works because the host
is built in release. The whole fix is one line in
`src/gpu.rs::BytePrefixOwned::from_str` (the `saturating_sub` for the
one-character case is already in, from this branch):

```rust
// A whole number of bytes means no partial byte, so the last byte is a *full*
// byte to be compared, not an absent one.
let last_byte_mask = if n_bits == 0 {
    0xFF
} else {
    (((1u32 << n_bits) - 1) << (8 - n_bits)) as u8
};
```

I left the semantics alone so the A/B is attributable and so the equivalence
claim stays exact, and the host re-verifies every captured key against the real
address (`main.rs`), so this costs matches, not correctness. **A/B this branch
first, then land the mask fix as its own change** -- it will move "keys
found/s" at length 8 by orders of magnitude and would swamp any perf attribution
if bundled.

**2. Prefixes longer than 12 characters now fail fast** (`prefix_word_masks`
asserts). That is a deliberate trade: 12 characters is 60 bits and the word is
64, and 13 characters would need bit 64, which is past the end of a `u64`.
`--bench-prefix` already refused those. A search for 13+ characters used to
work slowly and now panics with a message; if that ever needs supporting, the
byte-wise `matches` is still there and needs a `u128` word or a second compare.

**3. For `agent/nowalk`**: the overlap here is not accidental, and folding this
in would be strictly better than either change alone. `fe_to_bytes` is called
from exactly two places -- `ge_to_bytes_with_zi` (now test-only) and nothing
else -- so once `nowalk` removes the walk there is no second caller to keep the
two-byte-at-a-time path alive. But the two changes are not independent either:
`nowalk` was asked to "compute only the needed bits", and the answer to *which*
bits is the whole content of §3 -- the answer is not "the top 5*len bits of the
limb vector", it is "the low 64 bits of the canonical y, with the byte mask
folded in". If `nowalk` guessed the high limbs, it will produce a fast path that
disagrees with `matches` for every prefix whose length is not a multiple of 8,
and the equivalence tests in §3 are what will catch it. I would suggest
`nowalk` either adopt `fe_prefix_word` + `matches_word` wholesale (in which case
this branch's `core/src` half is redundant and only the host half matters), or
keep its structure and at minimum run
`group::tests::fast_path_agrees_with_the_byte_path` and
`group::tests::a_matching_key_matches_on_the_fast_path` against its version.
Neither of us should touch `render`'s chunk loop, and neither has.

**4. Not done, deliberately**: the double walk itself (2 point additions per
candidate, ~6735 of the ~8430 SASS per candidate -- by far the biggest remaining
item), the batch inversion's 3 `fe_mul` per candidate (~1050 SASS, versus ~248
for the amortised `fe_invert` itself), and `fe_sq`'s 264 SASS. Those are bigger
than what this change removes and they are the next things to attack.
