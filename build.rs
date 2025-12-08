fn main() {
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-search=native=/opt/cuda-11.8/lib64");
}
