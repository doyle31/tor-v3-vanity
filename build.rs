//! Builds the CUDA kernel crate for `nvptx64-nvidia-cuda` and hands the
//! resulting PTX to the host crate via `KERNEL_PTX_PATH`.
//!
//! The device crate is compiled with `--crate-type cdylib`, which makes rustc
//! run `llvm-bitcode-linker` (shipped in the `llvm-bitcode-linker` rustup
//! component) instead of a native linker, and emit `<crate>.ptx`.

use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const TARGET: &str = "nvptx64-nvidia-cuda";
const DEVICE_CRATE: &str = "core";

/// The compile-time kernel tunables, each with the environment variable that
/// can override it.
///
/// These must be compile-time constants: they size local arrays in the kernel
/// and drive loop unrolling, so a run-time value would push those buffers out of
/// registers. Hence a config file rather than a command-line option -- and a
/// file is also where the reasoning for a tuned value belongs.
const TUNABLES: [(&str, &str); 4] = [
    ("block", "T3V_BLOCK"),
    ("chunk", "T3V_CHUNK"),
    ("batch", "T3V_BATCH"),
    ("comb_w", "T3V_COMB_W"),
];

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=T3V_PTX_PROFILE");
    println!("cargo:rerun-if-env-changed=T3V_PTX_RUSTFLAGS");
    // The device crate reads these with `option_env!`, so the nested cargo
    // rebuilds the kernel when they change -- but only if this script re-runs
    // too. Without these lines a sweep silently measures the previous build.
    for (_, var) in TUNABLES {
        println!("cargo:rerun-if-env-changed={var}");
    }

    let tunables = resolve_tunables()?;
    link_cuda();
    let ptx = build_kernel(&tunables)?;

    println!("cargo:rustc-env=KERNEL_PTX_PATH={}", ptx.display());
    Ok(())
}

/// The resolved tunable values, keyed by environment-variable name.
type Tunables = Vec<(&'static str, usize)>;

/// Resolve every tunable, from the environment first and `kernel.conf` second,
/// and hand them to the nested build.
///
/// The values must reach the *device* crate, which build.rs compiles itself via
/// a nested `cargo rustc`. A `cargo:rustc-env` line would not get there: those
/// apply to the host crate's own rustc invocation, not to build.rs's process
/// environment, which is what the nested command inherits. So they are set on
/// the `Command` directly.
fn resolve_tunables() -> Result<Tunables> {
    let path = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?).join("kernel.conf");
    println!("cargo:rerun-if-changed={}", path.display());

    let file_values = match fs::read_to_string(&path) {
        Ok(text) => parse_kernel_conf(&text)?,
        // Not an error: the built-in defaults stand on their own.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(e) => return Err(format!("cannot read {}: {e}", path.display()).into()),
    };

    let mut resolved = Vec::new();
    let mut overridden = Vec::new();
    for (key, var) in TUNABLES {
        // An explicitly set variable wins, so a one-off experiment needs no
        // edit and cannot be silently clobbered by the file.
        let from_env = env::var(var).ok().filter(|v| !v.is_empty());
        let value = match from_env {
            Some(v) => {
                let parsed = v.parse::<usize>().map_err(|_| {
                    format!("{var} is set to {v:?}, which is not a number")
                })?;
                // Say so loudly. The file is now the place a build's settings
                // are recorded, so an environment variable quietly shadowing it
                // means the binary does not match the checked-in config -- and
                // that is exactly how a benchmark ends up measuring something
                // other than what the repository says.
                if let Some((_, from_file)) = file_values.iter().find(|(k, _)| *k == key) {
                    if *from_file != parsed {
                        overridden.push(format!(
                            "{var}={parsed} (env) overrides {key} = {from_file} in kernel.conf"
                        ));
                    }
                } else {
                    overridden.push(format!(
                        "{var}={parsed} (env) is not recorded in kernel.conf"
                    ));
                }
                parsed
            }
            None => match file_values.iter().find(|(k, _)| *k == key) {
                Some((_, v)) => *v,
                // Absent from both: leave it unset so the device crate's own
                // default applies, and there is exactly one place to change it.
                None => continue,
            },
        };
        resolved.push((var, value));
    }

    if !overridden.is_empty() {
        println!("cargo:warning=KERNEL TUNED BY ENVIRONMENT, NOT BY kernel.conf");
        for line in &overridden {
            println!("cargo:warning=  {line}");
        }
        println!(
            "cargo:warning=  this build will not match the checked-in config; \
             unset the variable, or put the value in kernel.conf to make it permanent"
        );
    }
    Ok(resolved)
}

/// Parse `key = value` lines with `#` comments.
///
/// Deliberately not TOML: three integer keys do not justify a build-time
/// dependency, and this format stays readable to anyone who has never built the
/// project. An unknown key is an error rather than a warning, because silently
/// ignoring a misspelled tunable means quietly benchmarking the wrong kernel.
fn parse_kernel_conf(text: &str) -> Result<Vec<(&'static str, usize)>> {
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let (key, value) = line.split_once('=').ok_or_else(|| {
            format!("kernel.conf:{}: expected `key = value`, got {:?}", n + 1, line)
        })?;
        let key = key.trim();
        let known = TUNABLES.iter().any(|(k, _)| *k == key);
        if !known {
            let names: Vec<&str> = TUNABLES.iter().map(|(k, _)| *k).collect();
            return Err(format!(
                "kernel.conf:{}: unknown key `{}`; known keys are {}",
                n + 1,
                key,
                names.join(", ")
            )
            .into());
        }
        let value: usize = value.trim().parse().map_err(|_| {
            format!("kernel.conf:{}: `{}` is not a number", n + 1, value.trim())
        })?;
        // The key is one of ours, so this cannot dangle: the `&str` borrow is
        // from TUNABLES, not from `text`.
        let key: &'static str = TUNABLES.iter().find(|(k, _)| *k == key).unwrap().0;
        out.push((key, value));
    }
    Ok(out)
}

/// Point the linker at the CUDA runtime/driver libraries.
///
/// `rustacuda` links `cudart`, which is not in a default linker search path,
/// so the CUDA install has to be added explicitly. `CUDA_PATH`/`CUDA_HOME`
/// win when set, otherwise the usual install locations are probed.
fn link_cuda() {
    let mut candidates: Vec<PathBuf> = ["CUDA_PATH", "CUDA_HOME"]
        .iter()
        .filter_map(|var| env::var_os(var))
        .map(PathBuf::from)
        .collect();
    candidates.push("/usr/local/cuda".into());
    candidates.push("/opt/cuda".into());

    let mut found = false;
    for root in &candidates {
        let libdir = root.join("lib64");
        if libdir.is_dir() {
            println!("cargo:rustc-link-search=native={}", libdir.display());
            found = true;
        }
    }

    if !found {
        println!(
            "cargo:warning=CUDA install not found in {}; set CUDA_PATH if the CUDA runtime library cannot be found",
            candidates
                .iter()
                .map(|c| c.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
}

/// Compile the device crate to PTX and return the path to the generated file.
fn build_kernel(tunables: &Tunables) -> Result<PathBuf> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let crate_dir = manifest_dir.join(DEVICE_CRATE);
    let crate_name = device_crate_name(&crate_dir.join("Cargo.toml"))?;

    // Track device crate inputs so cargo re-runs this script on changes.
    watch_dir(&crate_dir.join("src"))?;
    for name in &["Cargo.toml", "Cargo.lock"] {
        let path = crate_dir.join(name);
        if path.exists() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }

    require_bitcode_linker()?;

    // The kernel is compute code: always optimize unless asked otherwise.
    let profile = env::var("T3V_PTX_PROFILE").unwrap_or_else(|_| "release".into());
    let profile_dir = match profile.as_str() {
        "release" => "release",
        "debug" => "debug",
        other => return Err(format!("unsupported T3V_PTX_PROFILE `{other}`").into()),
    };

    let cargo = env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let mut cmd = Command::new(cargo);
    cmd.arg("rustc")
        .arg("--manifest-path")
        .arg(crate_dir.join("Cargo.toml"))
        .arg("--target")
        .arg(TARGET)
        .arg("--target-dir")
        .arg(crate_dir.join("target"))
        .arg("--crate-type")
        .arg("cdylib");
    if profile_dir == "release" {
        cmd.arg("--release");
    }

    // The parent build's rustflags describe the *host* target and are
    // meaningless (sometimes fatal) here, so drop them unless the caller
    // supplied kernel-specific flags.
    cmd.env_remove("RUSTFLAGS");
    cmd.env_remove("CARGO_ENCODED_RUSTFLAGS");
    if let Ok(flags) = env::var("T3V_PTX_RUSTFLAGS") {
        cmd.env("RUSTFLAGS", flags);
    }

    // The tunables have to reach *both* compilations, from the same source.
    //
    // The device crate is compiled by the nested command below, and reads them
    // with `option_env!`; that needs them in the nested process's environment.
    // The host crate is a separate compilation that also reads the same
    // constants -- `core::BLOCK` picks the launch geometry -- and gets them
    // from `cargo:rustc-env`. Setting only one of the two leaves the two
    // compilations disagreeing, which is exactly how a value in kernel.conf
    // ends up in the kernel but not in the host that launches it.
    for (var, value) in tunables {
        cmd.env(var, value.to_string());
        println!("cargo:rustc-env={var}={value}");
    }

    let output = cmd.output()?;
    if !output.status.success() {
        return Err(format!(
            "failed to build `{DEVICE_CRATE}` for {TARGET}:\n{}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    let ptx = crate_dir
        .join("target")
        .join(TARGET)
        .join(profile_dir)
        .join(format!("{crate_name}.ptx"));
    if !ptx.is_file() {
        return Err(format!("expected PTX output at {}", ptx.display()).into());
    }
    Ok(ptx)
}

/// Fail early with actionable instructions if the linker is missing.
///
/// Without it rustc cannot link bitcode for this target, and the resulting
/// error is much less obvious than this one.
fn require_bitcode_linker() -> Result<()> {
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let output = Command::new(rustc).arg("--print").arg("sysroot").output()?;
    if !output.status.success() {
        return Ok(());
    }

    let sysroot = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim().to_owned());
    let host = env::var("HOST").unwrap_or_default();
    let linker = sysroot
        .join("lib/rustlib")
        .join(&host)
        .join("bin/self-contained/llvm-bitcode-linker");
    if linker.is_file() {
        return Ok(());
    }

    Err(format!(
        "`llvm-bitcode-linker` is missing from the active toolchain.
The `{TARGET}` target requires a nightly toolchain with the kernel build
components installed:

    rustup toolchain install nightly
    rustup component add llvm-bitcode-linker --toolchain nightly
    rustup target add {TARGET} --toolchain nightly
"
    )
    .into())
}

/// Read the package name from a manifest so the PTX filename can be derived.
fn device_crate_name(manifest: &Path) -> Result<String> {
    let manifest = fs::read_to_string(manifest)?;
    let name = manifest
        .lines()
        .map(str::trim)
        .find_map(|line| line.strip_prefix("name"))
        .and_then(|rest| rest.split_once('='))
        .map(|(_, value)| value.trim().trim_matches('"').to_owned())
        .ok_or("device crate manifest has no `name`")?;
    Ok(name.replace('-', "_"))
}

fn watch_dir(dir: &Path) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            watch_dir(&path)?;
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
    Ok(())
}
