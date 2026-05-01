use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const LLAMA_CPP_REV: &str = "b8941";

fn main() {
    println!("cargo:rerun-if-env-changed=GGML_RS_SOURCE_DIR");
    println!("cargo:rerun-if-env-changed=GGML_RS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=GGML_RS_INCLUDE_DIR");
    println!("cargo:rerun-if-changed=wrapper.h");

    let include_dir = env::var_os("GGML_RS_INCLUDE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| ggml_source_dir().join("include"));

    if let Some(lib_dir) = env::var_os("GGML_RS_LIB_DIR").map(PathBuf::from) {
        link_from_dir(&lib_dir);
    } else {
        let source_dir = ggml_source_dir();
        let dst = build_ggml(&source_dir);
        link_built_libs(&dst);
    }

    generate_bindings(&include_dir);
}

fn ggml_source_dir() -> PathBuf {
    if let Some(path) = env::var_os("GGML_RS_SOURCE_DIR").map(PathBuf::from) {
        return path;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let llama_dir = out_dir.join("llama.cpp");
    if !llama_dir.exists() {
        let status = Command::new("git")
            .args([
                "clone",
                "--depth",
                "1",
                "--branch",
                LLAMA_CPP_REV,
                "https://github.com/ggml-org/llama.cpp.git",
            ])
            .arg(&llama_dir)
            .status()
            .expect("failed to spawn git clone for llama.cpp");
        assert!(
            status.success(),
            "failed to clone llama.cpp {LLAMA_CPP_REV}"
        );
    }
    llama_dir.join("ggml")
}

fn build_ggml(source_dir: &Path) -> PathBuf {
    let wrapper_dir = cmake_wrapper_dir(source_dir);
    let mut config = cmake::Config::new(&wrapper_dir);
    if cfg!(target_os = "windows") {
        config.generator("Ninja");
    }
    config
        .profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("GGML_STANDALONE", "OFF")
        .define("GGML_BUILD_TESTS", "OFF")
        .define("GGML_BUILD_EXAMPLES", "OFF")
        .define("GGML_BACKEND_DL", "OFF")
        .define("GGML_NATIVE", "OFF")
        .define("GGML_CPU", "ON")
        .define("GGML_METAL", metal_enabled())
        .define("GGML_METAL_EMBED_LIBRARY", metal_enabled())
        .define("GGML_VULKAN", vulkan_enabled());
    config.build()
}

fn cmake_wrapper_dir(source_dir: &Path) -> PathBuf {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let wrapper_dir = out_dir.join("ggml-cmake-wrapper");
    std::fs::create_dir_all(&wrapper_dir).expect("failed to create ggml CMake wrapper directory");
    let source_dir = source_dir.display().to_string().replace('\\', "/");
    let cmake = format!(
        "cmake_minimum_required(VERSION 3.20)\n\
         project(ggml_rs_sys LANGUAGES C CXX ASM)\n\
         add_subdirectory({} ${{CMAKE_BINARY_DIR}}/ggml)\n",
        source_dir
    );
    std::fs::write(wrapper_dir.join("CMakeLists.txt"), cmake)
        .expect("failed to write ggml CMake wrapper");
    wrapper_dir
}

fn link_built_libs(dst: &Path) {
    let mut search_dirs = vec![dst.join("lib"), dst.join("lib64")];
    search_dirs.extend(find_dirs_named(dst, "Release"));
    for lib_name in ggml_static_lib_files() {
        search_dirs.extend(find_dirs_containing(dst, lib_name));
    }
    for dir in search_dirs.into_iter().filter(|dir| dir.exists()) {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    link_ggml_libs(dst);
    link_platform_libs();
}

fn link_from_dir(lib_dir: &Path) {
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    for lib_name in ggml_static_lib_files() {
        for dir in find_dirs_containing(lib_dir, lib_name) {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
        if let Some(parent) = lib_dir.parent() {
            for dir in find_dirs_containing(parent, lib_name) {
                println!("cargo:rustc-link-search=native={}", dir.display());
            }
        }
    }
    link_ggml_libs(lib_dir);
    link_platform_libs();
}

fn ggml_static_lib_files() -> &'static [&'static str] {
    &[
        "libggml.a",
        "libggml-base.a",
        "libggml-cpu.a",
        "libggml-blas.a",
        "libggml-metal.a",
        "libggml-vulkan.a",
        "ggml.lib",
        "ggml-base.lib",
        "ggml-cpu.lib",
        "ggml-blas.lib",
        "ggml-metal.lib",
        "ggml-vulkan.lib",
    ]
}

fn link_ggml_libs(root: &Path) {
    for lib in [
        "ggml",
        "ggml-base",
        "ggml-cpu",
        "ggml-blas",
        "ggml-metal",
        "ggml-vulkan",
    ] {
        if has_static_lib(root, lib) {
            println!("cargo:rustc-link-lib=static={lib}");
        }
    }
}

fn has_static_lib(root: &Path, lib: &str) -> bool {
    find_dirs(root, &|dir| {
        dir.join(format!("lib{lib}.a")).exists() || dir.join(format!("{lib}.lib")).exists()
    })
    .into_iter()
    .next()
    .is_some()
}

fn link_platform_libs() {
    if cfg!(target_os = "macos") {
        if metal_enabled() == "ON" {
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
        }
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=c++");
    } else if cfg!(target_os = "linux") {
        if vulkan_enabled() == "ON" {
            println!("cargo:rustc-link-lib=vulkan");
        }
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=gomp");
        println!("cargo:rustc-link-lib=dylib=pthread");
        println!("cargo:rustc-link-lib=dylib=m");
        println!("cargo:rustc-link-lib=dylib=dl");
    } else if cfg!(target_os = "windows") && vulkan_enabled() == "ON" {
        println!("cargo:rustc-link-lib=vulkan-1");
    }
}

fn generate_bindings(include_dir: &Path) {
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", include_dir.display()))
        .clang_args(darwin_sysroot_args())
        .allowlist_function("gguf_.*")
        .allowlist_function("ggml_.*")
        .allowlist_type("gguf_.*")
        .allowlist_type("ggml_.*")
        .allowlist_var("GGML_.*")
        .generate()
        .expect("failed to generate ggml/gguf bindings");

    bindings
        .write_to_file(PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs"))
        .expect("failed to write bindings");
}

fn metal_enabled() -> &'static str {
    if cfg!(feature = "metal") || (cfg!(target_os = "macos") && !cfg!(feature = "vulkan")) {
        "ON"
    } else {
        "OFF"
    }
}

fn vulkan_enabled() -> &'static str {
    if cfg!(feature = "vulkan")
        || ((cfg!(target_os = "linux") || cfg!(target_os = "windows")) && !cfg!(feature = "metal"))
    {
        "ON"
    } else {
        "OFF"
    }
}

fn find_dirs_named(root: &Path, name: &str) -> Vec<PathBuf> {
    find_dirs(root, &|path| {
        path.file_name().is_some_and(|part| part == name)
    })
}

fn find_dirs_containing(root: &Path, file_name: &str) -> Vec<PathBuf> {
    find_dirs(root, &|path| path.join(file_name).exists())
}

fn find_dirs(root: &Path, pred: &dyn Fn(&Path) -> bool) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if pred(&dir) {
            out.push(dir.clone());
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            }
        }
    }
    out
}

fn darwin_sysroot_args() -> Vec<String> {
    if !cfg!(target_os = "macos") {
        return Vec::new();
    }
    if let Ok(sdkroot) = env::var("SDKROOT") {
        return vec!["-isysroot".to_string(), sdkroot];
    }
    let Ok(output) = Command::new("xcrun").args(["--show-sdk-path"]).output() else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let sdkroot = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if sdkroot.is_empty() {
        Vec::new()
    } else {
        vec!["-isysroot".to_string(), sdkroot]
    }
}
