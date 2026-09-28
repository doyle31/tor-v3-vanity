# t3v

Generate Tor v3 vanity `.onion` hostnames by brute-forcing ed25519 keys on an
NVIDIA GPU.

An onion address is `base32(pubkey || checksum || version)`, so its first
characters are a function of the public key. A key is derived from a clamped
scalar, and each scalar gives one key, so a prefix of *N* base32 characters is
found by generating keys until one matches -- 2^(5N) candidates in expectation.
The work is one point addition per candidate: the scalars that remain clamped
after clamping are exactly the multiples of 8, so the search walks an arithmetic
progression and every step yields a testable key.

## Install

Requires Rust nightly, the `nvptx64-nvidia-cuda` target, and CUDA.

```sh
rustup toolchain install nightly
rustup component add llvm-bitcode-linker --toolchain nightly
rustup target add nvptx64-nvidia-cuda --toolchain nightly
cargo +nightly install --path .
```

Nightly is not optional. The kernel crate uses `#![feature(abi_ptx,
stdarch_nvptx)]`, and there is no `rust-toolchain` file, so a bare `cargo build`
fails on the device crate.

If CUDA is not at `/usr/local/cuda` or `/opt/cuda`:

```sh
CUDA_PATH=/path/to/cuda cargo +nightly install --path .
```

## Use

```sh
# one key per file, in a directory
t3v --dst mykeys/ myprefix

# several prefixes at once
t3v --dst mykeys/ foo,bar,baz

# every match appended to one file
t3v -o keys.csv foo
t3v -o keys.json --format json foo

# a key Tor will accept as hs_ed25519_secret_key
cat mykeys/foo*.onion > /var/lib/tor/hidden_service/hs_ed25519_secret_key
```

Prefixes are base32 (`a-z`, `2-7`), at most 12 characters. Output is one file
per key by default; `--out` writes every match to a single file, whose encoding
follows `--format` or, failing that, the file extension.

Keys are written as they are found, so a search can be interrupted at any point
and everything already on disk is usable. At 5 characters and longer the tool
`fsync`s each key by default: those searches run for hours, a key arrives every
few seconds, and durability is free. Below 5 characters keys arrive thousands per
second and `fsync` is not. `--fsync` and `--no-fsync` override.

## Benchmarking

```sh
t3v --bench 10              # kernel only, no matching
t3v --bench 10 --match      # kernel plus the matcher
t3v --bench-prefix 5        # end to end: keys *found* per second
```

`--bench-prefix` is the number that predicts how long a search takes. `--bench`
reports keys examined per second, which is a different and much larger number.

On an RTX 3050 6GB laptop (20 SMs, 45 W cap) the match path sustains
**~223M keys/s**, confirmed three independent ways including a row count that
consults neither the launch counter nor the drop counter. See
[THROUGHPUT.md](THROUGHPUT.md) before quoting any rate from this repo.

Expected time to the first key at that rate:

| prefix | candidates | expected time |
|---|---|---|
| 3 | 2^15 | 0.0001 s |
| 4 | 2^20 | 0.005 s |
| 5 | 2^25 | 0.15 s |
| 6 | 2^30 | 4.8 s |
| 7 | 2^35 | 2.6 min |
| 8 | 2^40 | 1.4 h |
| 9 | 2^45 | 1.8 d |
| 10 | 2^50 | 58 d |
| 11 | 2^55 | 5.1 y |
| 12 | 2^60 | 164 y |

A 1- or 2-character search finds keys far faster than the host can derive and
write them, and the per-launch capture ring holds only 16 384 keys per prefix,
so most matches at those lengths are dropped. The progress line says so
explicitly rather than under-reporting.

## Tuning

Compile-time kernel constants live in [kernel.conf](kernel.conf), which carries
the measured sweep behind each value. Any of them can be overridden for a
one-off experiment, which wins over the file and needs no edit:

```sh
T3V_CHUNK=128 cargo +nightly build --release
```

| variable | default | purpose |
|---|---|---|
| `T3V_BLOCK` | 128 | upper bound on threads per block; the host derives the real value from register usage |
| `T3V_CHUNK` | 256 | candidates batch-inverted together; the inversion amortisation factor |
| `T3V_BATCH` | 512 | candidates per thread per launch; amortises SHA-512 and the base-point scalar multiplication |
| `T3V_COMB_W` | 4 | bits per base-point table window; only 2, 3 and 4 are legal |
| `T3V_PTX_PROFILE` | `release` | profile for the kernel crate |
| `T3V_PTX_RUSTFLAGS` | none | `RUSTFLAGS` for the kernel build only |
| `CUDA_PATH` | autodetected | CUDA root, used to find `libcudart` |

Invalid values are rejected on the main thread at startup, naming the variable,
rather than panicking inside a worker thread where the cause is invisible.

## How it works

```
src/         host: CLI, CUDA launch and readback, output sink, benchmarks
core/        device crate, compiled to PTX and embedded by build.rs
  field.rs     5x51 field arithmetic over u64 limbs
  group.rs     Edwards points, the comb table, batch inversion
  kernel.rs    the walk, the capture ring, `render`
  sha512.rs    the one hash a key needs
kernel.conf  compile-time tunables, with their measured sweeps
```

`build.rs` compiles `core/` for `nvptx64-nvidia-cuda` as a `cdylib`, which makes
rustc hand the LLVM bitcode to `llvm-bitcode-linker` and emit PTX, which the
binary embeds and loads at runtime. The tunables are passed to the device crate
through the nested build's environment *and* to the host crate through
`cargo:rustc-env`; setting only one of the two leaves the kernel and the host
that launches it disagreeing, which is the failure mode to watch for.

The kernel walks a chunk once, keeping each candidate's `Y` and `Z`, batch-inverts
the chunk's `Z` coordinates, and compresses from the result. A prefix is decided
by the low bits of `y`, so `X/Z` is never computed and neither is the rest of the
32-byte encoding. See [kernel.conf](kernel.conf) for the tuning constants and
[NOWALK.md](NOWALK.md) for why the walk happens once rather than twice -- note
that file's instruction counts are deprecated even though its measurements are
not.

## Disclaimer

Brand new and not thoroughly vetted. Please report bugs
[here](https://github.com/dr-bonez/tor-v3-vanity/issues).
