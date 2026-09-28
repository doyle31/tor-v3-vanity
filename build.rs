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

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=T3V_PTX_PROFILE");
    println!("cargo:rerun-if-env-changed=T3V_PTX_RUSTFLAGS");
    // The device crate reads these with `option_env!`, so the nested cargo
    // rebuilds the kernel when they change -- but only if this script re-runs
    // too. Without these lines the sweep silently measures the previous build.
    for tunables in ["T3V_BLOCK", "T3V_CHUNK", "T3V_BATCH", "T3V_COMB_W"] {
        println!("cargo:rerun-if-env-changed={tunables}");
    }

    link_cuda();
    let ptx = build_kernel()?;

    println!("cargo:rustc-env=KERNEL_PTX_PATH={}", ptx.display());
    Ok(())
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
fn build_kernel() -> Result<PathBuf> {
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
