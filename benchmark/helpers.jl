module MMSBBenchmarkHelpers

using BenchmarkTools
using Statistics

function _format_time(ns::Float64)
    if ns < 1_000.0
        return "$(round(ns, digits=1)) ns"
    elseif ns < 1_000_000.0
        return "$(round(ns / 1_000.0, digits=1)) μs"
    elseif ns < 1_000_000_000.0
        return "$(round(ns / 1_000_000.0, digits=1)) ms"
    else
        return "$(round(ns / 1_000_000_000.0, digits=2)) s"
    end
end

function _format_bytes(bytes::Int)
    if bytes < 1024
        return "$bytes B"
    elseif bytes < 1024^2
        return "$(round(bytes / 1024, digits=1)) KB"
    elseif bytes < 1024^3
        return "$(round(bytes / 1024^2, digits=1)) MB"
    else
        return "$(round(bytes / 1024^3, digits=2)) GB"
    end
end

function analyze_results(results)
    println("\n" * "=" ^ 80)
    println("MMSB Benchmark Analysis")
    println("=" ^ 80)
    for (category, group) in results
        println("\n📊 $category")
        println("-" ^ 60)
        for (name, trial) in group
            med = median(trial).time
            avg = mean(trial).time
            σ = std(trial.times)
            println("  $name")
            println("    median: $(_format_time(med))")
            println("    mean:   $(_format_time(avg))")
            println("    σ:      $(_format_time(σ))")
            println("    allocs: $(trial.allocs)")
            println("    memory: $(_format_bytes(trial.memory))")
        end
    end
end

function check_performance_targets(results)
    targets = Dict(
        ("allocation", "cpu_1kb") => 1_000.0,
        ("delta", "cpu_sparse") => 50_000.0,
        ("propagation", "single_hop") => 10_000.0,
    )
    println("\n🎯 Performance Target Validation")
    println("-" ^ 60)
    all_pass = true
    for ((category, name), target_ns) in targets
        if haskey(results, category) && haskey(results[category], name)
            actual = median(results[category][name]).time
            ratio = actual / target_ns
            if ratio <= 1.0
                println("  ✅ $category/$name: $(_format_time(actual)) (target $( _format_time(target_ns)))")
            else
                println("  ❌ $category/$name: $(_format_time(actual)) exceeds target by $(round((ratio - 1) * 100, digits=1))%")
                all_pass = false
            end
        else
            println("  ⚠️  Missing benchmark for $category/$name")
            all_pass = false
        end
    end
    return all_pass
end

end # module MMSBBenchmarkHelpers
