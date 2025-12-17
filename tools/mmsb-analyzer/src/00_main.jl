#!/usr/bin/env julia
# CFG-based Julia analyzer: generates proper Control Flow Graph DOT files

const BASE_DIR = @__DIR__

include(joinpath(BASE_DIR, "01_ast_cfg.jl"))
include(joinpath(BASE_DIR, "02_ir_ssa.jl"))
include(joinpath(BASE_DIR, "03_build_model.jl"))

using JSON
using Printf

function write_cfg_dot(fname::Symbol, cfg::ControlFlowGraph, dot_path::String)
    open(dot_path, "w") do io
        println(io, "digraph \"$(fname)\" {")
        println(io, "    rankdir=TB;")
        println(io, "    label=\"$(fname) (CC=$(length(cfg.edges) - length(cfg.nodes) + 2))\";")
        println(io, "    labelloc=t;")
        println(io, "")
        
        # Write nodes
        for node in cfg.nodes
            shape = if node.type == CFG_ENTRY
                "ellipse"
            elseif node.type == CFG_EXIT
                "doubleoctagon"
            elseif node.type == CFG_BRANCH
                "diamond"
            elseif node.type == CFG_LOOP_HEADER
                "box"
            else
                "box"
            end
            
            fillcolor = if node.type == CFG_ENTRY
                "lightgreen"
            elseif node.type == CFG_EXIT
                "lightcoral"
            elseif node.type == CFG_BRANCH
                "yellow"
            elseif node.type == CFG_LOOP_HEADER
                "orange"
            else
                "lightblue"
            end
            
            style = if node.type in [CFG_ENTRY, CFG_EXIT]
                "filled,bold"
            elseif node.type == CFG_LOOP_HEADER
                "filled,rounded"
            else
                "filled"
            end
            
            # Build label with line numbers if available
            label = node.label
            if !isempty(node.instructions)
                lines_str = join(node.instructions, ",")
                label = "$(node.label) L$(lines_str)"
            end
            
            println(io, "    n$(node.id) [label=\"$(label)\", shape=$(shape), fillcolor=$(fillcolor), style=\"$(style)\"];")
        end
        
        println(io, "")
        
        # Write edges
        for edge in cfg.edges
            edge_attrs = ""
            if edge.condition === :true
                edge_attrs = " [label=\"T\", color=\"darkgreen\"]"
            elseif edge.condition === :false
                edge_attrs = " [label=\"F\", color=\"red\"]"
            end
            println(io, "    n$(edge.source) -> n$(edge.target)$(edge_attrs);")
        end
        
        println(io, "}")
    end
end

function write_all_function_cfgs(file_path::String, dot_dir::String, title::String)
    println(stderr, "[DEBUG] Generating CFG DOTs for $(title)")
    all_code = read_all_code(file_path)
    functions, scg = analyze_code(all_code)
    
    println(stderr, "[DEBUG]   Functions found: $(length(functions))")
    
    if isempty(functions)
        println(stderr, "[DEBUG]   ⚠ No functions - skipping CFG generation")
        return
    end
    
    mkpath(dot_dir)
    
    # Generate CFG for each function
    cfg_count = 0
    for (fname, expr) in functions
        # Extract CFG from function body
        body = if expr.head == :function && length(expr.args) >= 2
            expr.args[2]
        elseif expr.head == :(=) && length(expr.args) == 2
            expr.args[2]
        else
            nothing
        end
        
        if body !== nothing
            cfg = extract_cfg_from_ast(fname, body)
            dot_path = joinpath(dot_dir, "$(fname)_cfg.dot")
            write_cfg_dot(fname, cfg, dot_path)
            cfg_count += 1
            println(stderr, "[DEBUG]     ✓ $(fname): $(length(cfg.nodes)) nodes, $(length(cfg.edges)) edges")
        end
    end
    
    println(stderr, "[DEBUG]   Generated $(cfg_count) CFG DOT files in: $(dot_dir)")
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
    
    write_all_function_cfgs(file_path, dot_dir, basename(file_path))
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
    
    # Generate CFG DOT files
    run_model(file_path, dot_dir)
    
    # Convert to JSON for Rust
    elements = convert_functions_to_json(functions, scg, file_path)
    println(JSON.json(elements))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
