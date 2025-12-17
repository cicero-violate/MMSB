#!/usr/bin/env julia
# Hybrid Julia analyzer: emits JSON summary + Graphviz DOT call graphs

const BASE_DIR = @__DIR__

include(joinpath(BASE_DIR, "01_ast_cfg.jl"))
include(joinpath(BASE_DIR, "02_ir_ssa.jl"))
include(joinpath(BASE_DIR, "03_build_model.jl"))

using JSON
using Printf

function write_dot_from_ast(file_path::String, dot_dir::String, title::String)
    println(stderr, "[DEBUG] Generating DOT from AST for $(title)")
    all_code = read_all_code(file_path)
    functions, scg = analyze_code(all_code)
    println(stderr, "[DEBUG]   Functions found: $(length(functions))")
    println(stderr, "[DEBUG]   Call edges found: $(length(scg))")
    if isempty(scg)
        println(stderr, "[DEBUG]   ⚠ No call graph - skipping DOT file")
        return
    end
    mkpath(dot_dir)
    dot_path = joinpath(dot_dir, "call_graph.dot")
    target_funcs = Set(Symbol.(keys(functions)))
    all_nodes = Set{Symbol}()
    foreach(f -> push!(all_nodes, f), target_funcs)
    for (src, dst) in scg
        push!(all_nodes, src)
        push!(all_nodes, dst)
    end
    println(stderr, "[DEBUG]   Total nodes: $(length(all_nodes))")
    println(stderr, "[DEBUG]   Writing to: $dot_path")
    open(dot_path, "w") do io
        title_escaped = replace(title, "\"" => "\\\"")
        println(io, "digraph \"$(title_escaped)\" {")
        println(io, "    rankdir=LR;")
        for node in sort(collect(all_nodes))
            label = replace(string(node), "\"" => "\\\"")
            shape = node in target_funcs ? "box" : "ellipse"
            println(io, "    \"$(label)\" [shape=$(shape)];")
        end
        for (src, dst) in scg
            src_label = replace(string(src), "\"" => "\\\"")
            dst_label = replace(string(dst), "\"" => "\\\"")
            println(io, "    \"$(src_label)\" -> \"$(dst_label)\";")
        end
        println(io, "}")
    end
    println(stderr, "[DEBUG]   ✓ DOT file written successfully")
end

function run_model(file_path::String, dot_dir::String)
    model = nothing
    try
        redirect_stdout(stderr) do
            model = build_model(
                file_path;
                scope = SCOPE_TARGET_ONLY,
                show_external_calls = false,
            )
        end
    catch e
        println(stderr, "[Julia analyzer] Failed to build model for $(basename(file_path)): ", e)
        model = nothing
    end
    
    write_dot_from_ast(file_path, dot_dir, basename(file_path))
    return model
end

function convert_functions_to_json(functions::Dict{Symbol, Expr}, scg::Vector{Tuple{Symbol, Symbol}}, file_path::String)
    elements = []
    
    # Build call map from static call graph
    calls_by_func = Dict{Symbol, Vector{String}}()
    for (caller, callee) in scg
        if !haskey(calls_by_func, caller)
            calls_by_func[caller] = String[]
        end
        push!(calls_by_func[caller], string(callee))
    end
    
    # Convert to unique sorted lists
    for (_, calls) in calls_by_func
        sort!(unique!(calls))
    end
    
    # Create elements for each function
    for (fname, expr) in functions
        calls = get(calls_by_func, fname, String[])
        push!(elements, Dict(
            "element_type" => "function",
            "name" => string(fname),
            "file_path" => file_path,
            "line_number" => 0,
            "signature" => string(fname),
            "calls" => calls
        ))
    end
    
    return elements
end

function main()
    if length(ARGS) < 2
        println(stderr, "Usage: 00_main.jl <file_path> <dot_directory>")
        exit(1)
    end
    file_path = abspath(ARGS[1])
    dot_dir = abspath(ARGS[2])
    
    # Parse with AST analyzer
    all_code = read_all_code(file_path)
    functions, scg = analyze_code(all_code)
    
    # Generate DOT file
    run_model(file_path, dot_dir)
    
    # Convert to JSON for Rust
    elements = convert_functions_to_json(functions, scg, file_path)
    println(JSON.json(elements))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
