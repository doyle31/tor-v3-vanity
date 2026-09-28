# Throughput: the rate is real, the instruction counts are not

## The rate is confirmed

Three measurement paths, which share no code:

| path | result |
|---|---|
| host: launches x 15,728,640 keys/launch | 228M keys/s |
| device: its own atomic match counter, 1-char prefix | 228M keys/s |
| rows written / elapsed, 3-char prefix | **223.4M keys/s** |

The device path: at a 1-character prefix exactly 1 key in 32 matches. An 8-second
run reported 56,784,984 dropped + 292,602 captured = 57,077,586 matches against
1,824,522,240 keys tried, where 1/32 predicts 57,016,320 -- agreement to 0.1%. A
match count cannot be inflated without also inflating keys/launch, since it is
strictly proportional to keys examined.

The row path is the cleanest, because it consults neither `keys_per_launch` nor
the drop counter: 82,602 rows over 12.115s, x 2^15, = 223,408,447 keys/s, with
zero drop warnings.

## The keys are real

82,602 keys from that run, checked independently in Python:

* 82,602 distinct, 0 duplicates
* 0 unclamped scalars
* 0 bad prefixes, 0 bad checksums, 0 bad lengths
* 240/240 verified by a full ed25519 implementation

So this is genuine key generation at ~223M keys/s, not a counter incrementing
faster than the arithmetic can run.

## What the rate implies, and what is therefore in doubt

At 223M keys/s the whole GPU has roughly 610 warp-instructions per candidate at
1.7 GHz, or ~753 at the 2.1 GHz boost (20 SMs x 4 issue slots x clock).

A `ge_madd` is 7 `fe_mul` x 25 limb products, and a 51x51 product on hardware
with only 32x32->64 multiplies is 4 `IMAD.WIDE`, so the multiply floor alone is
700 per candidate. That floor sits inside the boost-clock budget, which makes the
rate *plausible* -- but only just, and it leaves no room for the u128
accumulation, carry chain or conditional subtraction that the code plainly
contains. So the per-candidate cost is somewhere near 750, and nothing in the
surviving documentation says it is.

**The discredited numbers.** `NOWALK.md` is a subagent artefact written against
pre-comb code, and its derived figures contradict each other by more than the
margin above:

| claim in NOWALK.md | implied per candidate |
|---|---|
| pass-1 body 3,487 SASS, 745 `IMAD.WIDE` | 3,487 |
| pass-2 body 4,158 SASS, 853 `IMAD.WIDE` | 4,158 |
| "marginal `IMAD.WIDE` / candidate" 314 | 314 |
| "whole chunk" 8.53M / 256 | 33,320 |

The first two sum to 7,645. The last implies 33,320. The 4.4x gap is charged to
batch inversion, but a batch inversion is 3 `fe_mul` per candidate plus
(254 squarings + 11 multiplies)/256 -- about 1,125, not the 25,678 the gap needs.
And 314 is 5x below the pass bodies it sits beside.

`NOWALK.md` is marked deprecated for this reason. Its +34% A/B for the walk
removal is a direct measurement and is not in question; the per-candidate
instruction figures derived from it are.

## What to do next

Re-derive the per-candidate cost from the current SASS, and establish the trip
count of the walk loops, which nobody has. Per-candidate cost is
`loop body x trip count x (BATCH / CHUNK)`, and the budget it has to fit is
~750 warp-instructions.

```
ptxas -arch=sm_86 -O3 -v core/target/nvptx64-nvidia-cuda/release/*.ptx
nvdisasm -c <cubin>
```

The five large backward-branch loops are 3,487 / 4,158 / 3,478 / 3,449 / 3,412
instructions, carrying 745 / 853 / 745 / 748 / 733 `IMAD.WIDE`. 745 is almost
exactly 7 x 100 + 45, one `ge_madd`'s worth of multiplies plus the `x19` folds,
which is suggestive -- but a suggestive instruction count is not a trip count, and
if the 3,487 body really ran 256 times per chunk the rate would be impossible by
an order of magnitude. That contradiction is unresolved, and it is a question
about the SASS, not about the measurement.

## Consequence for benchmarking

The rate is trustworthy, so A/B comparisons of two builds on the match path are
meaningful. The `--bench` keygen figure (511M) is *not* corroborated by any
independent path the way the match rate now is, and it implies ~266
warp-instructions per candidate, well under the 700 multiply floor. Treat it as
unverified and do not quote it as a speedup.
