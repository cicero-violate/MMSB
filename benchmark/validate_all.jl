#!/usr/bin/env julia

include("run_validation.jl")
using .MMSBValidationHarness
using Printf
using Libdl
using JSON3
using Dates

function announce_mmsb_core_library()
    candidate = abspath(joinpath(@__DIR__, "..", "target", "release", "libmmsb_core.so"))
    path = if isfile(candidate)
        candidate
    else
        try
            Libdl.dlpath(:libmmsb_core)
        catch err
            nothing
        end
    end
    if path === nothing
        @warn "libmmsb_core.so not found; ensure LD_LIBRARY_PATH includes target/release"
    else
        println("Using libmmsb_core: ", path)
    end
end

function detail_summary(result::MMSBValidationHarness.ValidationResult)
    id = result.id
    details = result.details
    if id == "5_delta_throughput"
        single = get(details, "single_throughput", 0.0) / 1_000_000
        multi = get(details, "multi_throughput", 0.0) / 1_000_000
        note = multi >= 10.0 ? "" : " → multi target tracked as risk"
        return @sprintf("single: %.2fM/sec, multi: %.2fM/sec%s", single, multi, note)
    elseif id == "7_memory_footprint"
        snap = details["snapshot"]
        gc = details["gc_ms"]
        frag = details["fragmentation_estimate"]
        return @sprintf("avg: %.0fB, projected: %.1fMB, GC: %.3fms, frag: %.1fMB",
            snap === nothing ? 0.0 : snap.avg,
            snap === nothing ? 0.0 : snap.projected / 1_000_000,
            gc === nothing ? 0.0 : gc,
            frag === nothing ? 0.0 : frag / 1_000_000)
    elseif id == "8_invariant_preservation" || id == "9_stability_under_perturbation"
        div = get(details, "max_divergence", 0.0)
        return @sprintf("cycles: %d, divergence: %.4f, violations: %d",
            get(details, "cycles", 0),
            div,
            get(details, "invariant_failures", 0))
    elseif haskey(details, "summary")
        return details["summary"]
    else
        return string(details)
    end
end

function emit_report(results)
    println("=== MMSB Benchmark Validation ===")
    for (idx, result) in enumerate(results)
        mark = result.passed ? "✓" : "✗"
        println(@sprintf("%s #%d: %s (%s)", mark, idx, result.description, detail_summary(result)))
    end
    total = count(r -> r.passed, results)
    println(@sprintf("\nResult: %d/%d PASS", total, length(results)))
end

function normalize_value(value)
    if value isa Dict
        return Dict(string(k) => normalize_value(v) for (k, v) in value)
    elseif value isa NamedTuple
        return Dict(string(k) => normalize_value(v) for (k, v) in pairs(value))
    elseif value isa AbstractVector
        return [normalize_value(v) for v in value]
    elseif value isa Tuple
        return [normalize_value(v) for v in value]
    elseif value === nothing
        return nothing
    elseif value isa AbstractString || value isa Bool
        return value
    elseif value isa Integer
        return Int(value)
    elseif value isa AbstractFloat
        return Float64(value)
    else
        return string(value)
    end
end

function persist_results(results)
    timestamp = now()
    payload = Dict(
        "timestamp" => string(timestamp),
        "pass_count" => count(r -> r.passed, results),
        "total" => length(results),
        "results" => [
            Dict(
                "index" => idx,
                "id" => result.id,
                "description" => result.description,
                "passed" => result.passed,
                "target" => result.target_text,
                "details" => normalize_value(result.details),
            )
            for (idx, result) in enumerate(results)
        ],
    )
    output_path = get(ENV, "MMSB_VALIDATION_RESULTS", joinpath(@__DIR__, "results", "julia_phase6.json"))
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        JSON3.write(io, payload; indent=2)
    end
    history_dir = joinpath(@__DIR__, "results", "history")
    mkpath(history_dir)
    stamp = Dates.format(timestamp, dateformat"yyyymmddTHHMMSS")
    history_path = joinpath(history_dir, "julia_validation_$stamp.json")
    open(history_path, "w") do io
        JSON3.write(io, payload; indent=2)
    end
    println("Wrote structured validation metrics → ", output_path)
    println("Archived run → ", history_path)
end

announce_mmsb_core_library()
results = MMSBValidationHarness.run_all_validations()
persist_results(results)
emit_report(results)
exit(all(r -> r.passed, results) ? 0 : 1)
