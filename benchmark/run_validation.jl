#!/usr/bin/env julia

module MMSBValidationHarness

using BenchmarkTools: median
using Dates
using JSON3
using Printf

include("benchmarks.jl")
using .MMSBBenchmarks

export run_all_validations, ValidationResult

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const TARGET_PATH = joinpath(@__DIR__, "benchmarks_targets.json")
const BENCHMARK_ORDER = [
    "1_deterministic_replay",
    "2_delta_integrity",
    "3_page_graph_consistency",
    "4_algorithm_isolation",
    "5_delta_throughput",
    "6_tick_latency",
    "7_memory_footprint",
    "8_invariant_preservation",
    "9_stability_under_perturbation",
    "10_traceability_observability",
]

struct ValidationResult
    id::String
    description::String
    target_text::String
    passed::Bool
    details::Dict{String,Any}
end

const SUITE_THRESHOLDS = Dict(
    "1_deterministic_replay" => (
        category = "replay",
        name = "100_epochs",
        metric = :median_ns,
        threshold = 50_000_000.0,
        fmt = value -> @sprintf("%.2f ms median", value / 1e6),
    ),
    "2_delta_integrity" => (
        category = "delta",
        name = "cpu_sparse",
        metric = :median_ns,
        threshold = 2_000_000.0,
        fmt = value -> @sprintf("%.2f ms sparse apply", value / 1e6),
    ),
    "3_page_graph_consistency" => (
        category = "graph",
        name = "topological_sort_1024",
        metric = :median_ns,
        threshold = 15_000_000.0,
        fmt = value -> @sprintf("%.2f ms topo-sort", value / 1e6),
    ),
    "4_algorithm_isolation" => (
        category = "semiring",
        name = "boolean_fold_add",
        metric = :median_ns,
        threshold = 5_000_000.0,
        fmt = value -> @sprintf("%.2f ms boolean fold", value / 1e6),
    ),
    "6_tick_latency" => (
        category = "system",
        name = "full_pipeline",
        metric = :median_ns,
        threshold = 16_000_000.0,
        fmt = value -> @sprintf("%.2f ms tick", value / 1e6),
    ),
    "10_traceability_observability" => (
        category = "graph",
        name = "bfs_1024",
        metric = :median_ns,
        threshold = 50_000_000.0,
        fmt = value -> @sprintf("%.2f ms BFS", value / 1e6),
    ),
)

const TARGETS_CACHE = Ref{Union{Nothing,Dict{String,Any}}}(nothing)
const SUITE_CACHE = Ref{Union{Nothing,Any}}(nothing)
const STABILITY_CACHE = Ref{Union{Nothing,Tuple{Bool,Dict{String,Any}}}}(nothing)

function load_targets()
    if TARGETS_CACHE[] === nothing
        TARGETS_CACHE[] = JSON3.read(read(TARGET_PATH, String), Dict{String,Any})
    end
    return TARGETS_CACHE[]::Dict{String,Any}
end

function suite_results()
    if SUITE_CACHE[] === nothing
        SUITE_CACHE[] = MMSBBenchmarks.run_benchmarks(save_results=false)
    end
    return SUITE_CACHE[]
end

function metric_value(trial, metric::Symbol)
    if metric == :median_ns
        return Float64(median(trial).time)
    elseif metric == :memory
        return Float64(trial.memory)
    elseif metric == :allocs
        return Float64(trial.allocs)
    else
        error("unknown metric $metric")
    end
end

function capture_cmd(cmd::Cmd)
    stdout_buf = IOBuffer()
    stderr_buf = IOBuffer()
    pipeline_cmd = pipeline(cmd; stdout=stdout_buf, stderr=stderr_buf)
    ok = true
    try
        run(pipeline_cmd)
    catch err
        if err isa Base.ProcessFailedException
            ok = false
        else
            rethrow(err)
        end
    end
    output = String(take!(stdout_buf))
    output *= String(take!(stderr_buf))
    return ok, output
end

function parse_metric(output::String, pattern::Regex; last_only::Bool=true)
    matches = collect(eachmatch(pattern, output))
    isempty(matches) && return nothing
    value = parse(Float64, last_only ? matches[end].captures[1] : matches[1].captures[1])
    return value
end

function suite_runner(id::String, spec::Dict{String,Any})
    cfg = SUITE_THRESHOLDS[id]
    trials = suite_results()
    trial = trials[cfg.category][cfg.name]
    value = metric_value(trial, cfg.metric)
    passed = value <= cfg.threshold
    details = Dict(
        "metric" => cfg.metric,
        "value" => value,
        "threshold" => cfg.threshold,
        "summary" => cfg.fmt(value),
    )
    return ValidationResult(id, spec["description"], spec["target"], passed, details)
end

function throughput_runner(spec::Dict{String,Any})
    single_cmd = Cmd(
        `cargo test --release --test stress_throughput single_thread_1m_deltas_per_sec -- --nocapture`;
        dir=REPO_ROOT,
    )
    multi_cmd = Cmd(
        `cargo test --release --test stress_throughput multi_thread_10m_deltas_per_sec -- --nocapture`;
        dir=REPO_ROOT,
    )
    single_ok, single_out = capture_cmd(single_cmd)
    multi_ok, multi_out = capture_cmd(multi_cmd)
    single_tp = parse_metric(single_out, r"throughput_per_sec=([0-9\.]+)")
    multi_tp = parse_metric(multi_out, r"throughput_per_sec=([0-9\.]+)")
    details = Dict(
        "single_throughput" => single_tp,
        "multi_throughput" => multi_tp,
        "raw_single" => single_out,
        "raw_multi" => multi_out,
    )
    passed = single_ok && multi_ok &&
        single_tp !== nothing && multi_tp !== nothing &&
        single_tp >= 1_000_000.0 && multi_tp >= 1_500_000.0
    return ValidationResult("5_delta_throughput", spec["description"], spec["target"], passed, details)
end

function memory_runner(spec::Dict{String,Any})
    cmd = Cmd(`cargo test --release --test stress_memory -- --nocapture`; dir=REPO_ROOT)
    ok, output = capture_cmd(cmd)
    snapshot = match(r"METRIC:memory_snapshot avg=([0-9\.]+) total=([0-9]+) projected=([0-9]+)", output)
    gc = match(r"METRIC:memory_gc duration_ms=([0-9\.]+)", output)
    frag = match(r"METRIC:memory_fragmentation total_pages=([0-9]+) estimate=([0-9]+)", output)
    details = Dict{String,Any}()
    if snapshot !== nothing
        details["snapshot"] = (
            avg = parse(Float64, snapshot.captures[1]),
            total = parse(Int, snapshot.captures[2]),
            projected = parse(Int, snapshot.captures[3]),
        )
    else
        details["snapshot"] = nothing
    end
    details["gc_ms"] = gc === nothing ? nothing : parse(Float64, gc.captures[1])
    if frag !== nothing
        details["fragmentation_pages"] = parse(Int, frag.captures[1])
        details["fragmentation_estimate"] = parse(Int, frag.captures[2])
    else
        details["fragmentation_pages"] = nothing
        details["fragmentation_estimate"] = nothing
    end
    details["raw"] = output
    passed = ok &&
        details["snapshot"] !== nothing &&
        details["snapshot"].avg <= 1_024 &&
        details["snapshot"].projected <= 1_073_741_824 &&
        details["gc_ms"] !== nothing &&
        details["gc_ms"] <= 3.0 &&
        details["fragmentation_estimate"] !== nothing &&
        details["fragmentation_estimate"] <= 1_073_741_824
    return ValidationResult("7_memory_footprint", spec["description"], spec["target"], passed, details)
end

function stability_metrics()
    if STABILITY_CACHE[] === nothing
        cmd = Cmd(`cargo test --release --test stress_stability -- --nocapture`; dir=REPO_ROOT)
        ok, output = capture_cmd(cmd)
        metric = match(r"METRIC:stability cycles=([0-9]+) max_divergence=([0-9\.]+) invariant_failures=([0-9]+)", output)
        details = Dict{String,Any}("raw" => output)
        if metric !== nothing
            details["cycles"] = parse(Int, metric.captures[1])
            details["max_divergence"] = parse(Float64, metric.captures[2])
            details["invariant_failures"] = parse(Int, metric.captures[3])
        end
        STABILITY_CACHE[] = (ok, details)
    end
    return STABILITY_CACHE[]::Tuple{Bool,Dict{String,Any}}
end

function stability_runner(id::String, spec::Dict{String,Any}; divergence_limit::Float64=0.5)
    ok, details_cache = stability_metrics()
    details = deepcopy(details_cache)
    passed = ok &&
        get(details, "cycles", 0) == 10_000 &&
        get(details, "invariant_failures", 1) == 0 &&
        get(details, "max_divergence", divergence_limit + 1) <= divergence_limit
    return ValidationResult(id, spec["description"], spec["target"], passed, details)
end

function run_benchmark(id::String, spec::Dict{String,Any})
    if haskey(SUITE_THRESHOLDS, id)
        return suite_runner(id, spec)
    elseif id == "5_delta_throughput"
        return throughput_runner(spec)
    elseif id == "7_memory_footprint"
        return memory_runner(spec)
    elseif id == "8_invariant_preservation"
        return stability_runner(id, spec; divergence_limit=Inf)
    elseif id == "9_stability_under_perturbation"
        return stability_runner(id, spec; divergence_limit=0.5)
    else
        error("No runner for $id")
    end
end

function run_all_validations()
    targets = load_targets()
    specs = targets["benchmarks"]
    results = ValidationResult[]
    for id in BENCHMARK_ORDER
        if !haskey(specs, id)
            continue
        end
        push!(results, run_benchmark(id, specs[id]))
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = run_all_validations()
    println("Timestamp: ", string(now()))
    println("MMSB Validation Harness")
    for result in results
        status = result.passed ? "PASS" : "FAIL"
        println("[$status] $(result.id) — $(result.description)")
        println("    Target: $(result.target_text)")
        println("    Details: $(result.details)")
    end
    exit(all(r -> r.passed, results) ? 0 : 1)
end

end # module
