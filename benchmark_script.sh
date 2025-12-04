julia --startup-file=no --color=yes test/runtests.jl 

MMSB_BENCH_CATEGORIES='allocation,delta,propagation' julia --startup-file=no --color=yes -e '…run_benchmarks…'

MMSB_BENCH_CATEGORIES='replay,persistence,stress'
