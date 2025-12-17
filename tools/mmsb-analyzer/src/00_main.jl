#!/usr/bin/env julia
# CFG-based Julia analyzer: generates proper Control Flow Graph DOT files

const BASE_DIR = @__DIR__

include(joinpath(BASE_DIR, "01_ast_cfg.jl"))
include(joinpath(BASE_DIR, "02_ir_ssa.jl"))
include(joinpath(BASE_DIR, "03_build_model.jl"))

using JSON
using Printf

function write_combined_cfg_dot(file_path::String, dot_path::String, title::String)
    """Generate one DOT file with all functions from a file, showing CFGs and inter-function calls"""
    
    all_code = read_all_code(file_path)
    functions, scg = analyze_code(all_code)
    
    if isempty(functions)
        return 0
    end
    
    open(dot_path, "w") do io
        println(io, "digraph ProgramCFG {")
        println(io, "  rankdir=TB;")
        println(io, "  compound=true;")
        println(io, "  newrank=true;")
        println(io, "")
        println(io, "  // Program metadata")
        println(io, "  label=\"$(title) - $(length(functions)) functions\";")
        println(io, "  labelloc=t;")
        println(io, "  fontsize=16;")
        println(io, "")
        
        func_idx = 1
        func_to_cluster = Dict{Symbol, Int}()
        func_entry_nodes = Dict{Symbol, Int}()
        func_exit_nodes = Dict{Symbol, Int}()
        
        # Generate CFG for each function in its own cluster
        for (fname, expr) in sort(collect(functions), by=x->string(x[1]))
            body = if expr.head == :function && length(expr.args) >= 2
                expr.args[2]
            elseif expr.head == :(=) && length(expr.args) == 2
                expr.args[2]
            else
                nothing
            end
            
            if body === nothing
                continue
            end
            
            cfg = extract_cfg_from_ast(fname, body)
            cc = length(cfg.edges) - length(cfg.nodes) + 2
            func_to_cluster[fname] = func_idx
            
            println(io, "  subgraph cluster_$(func_idx) {")
            println(io, "    label=\"$(fname) (CC=$(cc))\";")
            println(io, "    style=filled;")
            println(io, "    fillcolor=lightgray;")
            println(io, "    color=black;")
            println(io, "")
            
            # Track entry/exit for inter-function calls
            func_entry_nodes[fname] = cfg.entry_id
            func_exit_nodes[fname] = cfg.exit_id
            
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
                
                label = node.label
                if !isempty(node.instructions)
                    lines_str = join(node.instructions, ",")
                    label = "$(node.label) L$(lines_str)"
                end
                
                println(io, "    f$(func_idx)_n$(node.id) [label=\"$(label)\", shape=$(shape), fillcolor=$(fillcolor), style=\"$(style)\"];")
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
                println(io, "    f$(func_idx)_n$(edge.source) -> f$(func_idx)_n$(edge.target)$(edge_attrs);")
            end
            
            println(io, "  }")
            println(io, "")
            func_idx += 1
        end
        
        # Inter-function calls
        println(io, "  // Inter-function calls")
        println(io, "  edge [style=dashed, color=blue, penwidth=2];")
        println(io, "")
        
        for (caller, callee) in scg
            if haskey(func_to_cluster, caller) && haskey(func_to_cluster, callee)
                caller_cluster = func_to_cluster[caller]
                callee_cluster = func_to_cluster[callee]
                caller_exit = func_exit_nodes[caller]
                callee_entry = func_entry_nodes[callee]
                
                println(io, "  f$(caller_cluster)_n$(caller_exit) -> f$(callee_cluster)_n$(callee_entry) [ltail=cluster_$(caller_cluster), lhead=cluster_$(callee_cluster), label=\"call\"];")
            end
        end
        
        println(io, "}")
    end
    
    return length(functions)
end

function write_layer_cfg_dot(layer_name::String, file_paths::Vector{String}, dot_path::String)
    """Generate one DOT file combining all functions from all files in a layer"""
    
    all_functions = Dict{Symbol, Expr}()
    all_scg = Vector{Tuple{Symbol, Symbol}}()
    file_sources = Dict{Symbol, String}()
    
    # Collect all functions from all files in layer
    for file_path in file_paths
        all_code = read_all_code(file_path)
        functions, scg = analyze_code(all_code)
        
        for (fname, expr) in functions
            all_functions[fname] = expr
            file_sources[fname] = basename(file_path)
        end
        
        append!(all_scg, scg)
    end
    
    if isempty(all_functions)
        return 0
    end
    
    # Write combined layer DOT
    open(dot_path, "w") do io
        println(io, "digraph LayerCFG {")
        println(io, "  rankdir=TB;")
        println(io, "  compound=true;")
        println(io, "  newrank=true;")
        println(io, "")
        println(io, "  label=\"$(layer_name) - $(length(all_functions)) functions from $(length(file_paths)) files\";")
        println(io, "  labelloc=t;")
        println(io, "  fontsize=18;")
        println(io, "")
        
        func_idx = 1
        func_to_cluster = Dict{Symbol, Int}()
        func_entry_nodes = Dict{Symbol, Int}()
        func_exit_nodes = Dict{Symbol, Int}()
        
        for (fname, expr) in sort(collect(all_functions), by=x->string(x[1]))
            body = if expr.head == :function && length(expr.args) >= 2
                expr.args[2]
            elseif expr.head == :(=) && length(expr.args) == 2
                expr.args[2]
            else
                nothing
            end
            
            if body === nothing
                continue
            end
            
            cfg = extract_cfg_from_ast(fname, body)
            cc = length(cfg.edges) - length(cfg.nodes) + 2
            func_to_cluster[fname] = func_idx
            source_file = get(file_sources, fname, "unknown")
            
            println(io, "  subgraph cluster_$(func_idx) {")
            println(io, "    label=\"$(fname) ($(source_file), CC=$(cc))\";")
            println(io, "    style=filled;")
            println(io, "    fillcolor=lightgray;")
            println(io, "    color=black;")
            println(io, "")
            
            func_entry_nodes[fname] = cfg.entry_id
            func_exit_nodes[fname] = cfg.exit_id
            
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
                
                label = node.label
                if !isempty(node.instructions)
                    lines_str = join(node.instructions, ",")
                    label = "$(node.label) L$(lines_str)"
                end
                
                println(io, "    f$(func_idx)_n$(node.id) [label=\"$(label)\", shape=$(shape), fillcolor=$(fillcolor), style=\"$(style)\"];")
            end
            
            println(io, "")
            
            for edge in cfg.edges
                edge_attrs = ""
                if edge.condition === :true
                    edge_attrs = " [label=\"T\", color=\"darkgreen\"]"
                elseif edge.condition === :false
                    edge_attrs = " [label=\"F\", color=\"red\"]"
                end
                println(io, "    f$(func_idx)_n$(edge.source) -> f$(func_idx)_n$(edge.target)$(edge_attrs);")
            end
            
            println(io, "  }")
            println(io, "")
            func_idx += 1
        end
        
        println(io, "  // Inter-function calls")
        println(io, "  edge [style=dashed, color=blue, penwidth=2];")
        println(io, "")
        
        for (caller, callee) in unique(all_scg)
            if haskey(func_to_cluster, caller) && haskey(func_to_cluster, callee)
                caller_cluster = func_to_cluster[caller]
                callee_cluster = func_to_cluster[callee]
                caller_exit = func_exit_nodes[caller]
                callee_entry = func_entry_nodes[callee]
                
                println(io, "  f$(caller_cluster)_n$(caller_exit) -> f$(callee_cluster)_n$(callee_entry) [ltail=cluster_$(caller_cluster), lhead=cluster_$(callee_cluster), label=\"call\"];")
            end
        end
        
        println(io, "}")
    end
    
    return length(all_functions)
end

function detect_layer(file_path::String)
    """Extract layer from path like src/00_physical/File.jl -> 00_physical"""
    parts = splitpath(file_path)
    for part in parts
        if occursin(r"^\d+_", part)
            return part
        end
    end
    return "root"
end

function write_all_cfgs(file_path::String, dot_dir::String, title::String)
    println(stderr, "[DEBUG] Generating CFG DOT for $(title)")
    
    # Per-file DOT
    file_dot = joinpath(dot_dir, "file_cfg.dot")
    func_count = write_combined_cfg_dot(file_path, file_dot, title)
    
    if func_count > 0
        println(stderr, "[DEBUG]   ✓ Generated file CFG: $(func_count) functions")
    else
        println(stderr, "[DEBUG]   ⚠ No functions found")
    end
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
    
    write_all_cfgs(file_path, dot_dir, basename(file_path))
    return model
end

function convert_functions_to_json(functions::Dict{Symbol, Expr}, scg::Vector{Tuple{Symbol, Symbol}}, file_path::String)
    elements = []
    
    calls_by_func = Dict{Symbol, Vector{String}}()
    for (caller, callee) in scg
        if !haskey(calls_by_func, caller)
            calls_by_func[caller] = String[]
        end
        push!(calls_by_func[caller], string(callee))
    end
    
    for (_, calls) in calls_by_func
        sort!(unique!(calls))
    end
    
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
    
    all_code = read_all_code(file_path)
    functions, scg = analyze_code(all_code)
    
    run_model(file_path, dot_dir)
    
    elements = convert_functions_to_json(functions, scg, file_path)
    println(JSON.json(elements))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
