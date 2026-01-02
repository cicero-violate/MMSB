#!/usr/bin/env julia
# CFG-based Julia analyzer: generates proper Control Flow Graph DOT files

const BASE_DIR = @__DIR__

using JSON
using Printf
using Base: Set

const PROJECT_LOADED = Ref(false)
const PROJECT_ROOT = Ref{String}("")
const ROOT_MODULE_NAME = Ref{Symbol}(Symbol(""))
const ROOT_MODULE_REF = Ref{Union{Module,Nothing}}(nothing)
const SKIP_DIRECTORIES = Set(["test", "tests", "examples", "tools", ".julia"])

function detect_entry_file(root_path::String)
    candidates = [
        joinpath(root_path, "src", "MMSB.jl"),
        joinpath(root_path, "MMSB.jl"),
        joinpath(root_path, "src", "API.jl"),
        joinpath(root_path, "API.jl"),
        joinpath(root_path, "src", "010_MMSBAnalyzerJulia.jl"),
        joinpath(root_path, "010_MMSBAnalyzerJulia.jl"),
        joinpath(root_path, "src", "MMSBAnalyzerJulia.jl"),
        joinpath(root_path, "MMSBAnalyzerJulia.jl"),
    ]

    for candidate in candidates
        if isfile(candidate)
            return candidate
        end
    end

    error("Unable to locate Julia entry file under $root_path")
end

function detect_root_module_name(entry_path::String)
    for line in eachline(entry_path)
        m = match(r"^\s*module\s+([A-Za-z0-9_\.]+)", line)
        if m !== nothing
            return Symbol(m.captures[1])
        end
    end

    error("Entry file $entry_path does not declare a module")
end

function alias_top_level_modules!(root_mod::Module)
    for name in names(root_mod; all=true, imported=false)
        if isdefined(root_mod, name)
            obj = getfield(root_mod, name)
            if obj isa Module && !isdefined(Main, name)
                @eval Main const $(Symbol(name)) = $obj
            end
        end
    end
end

function ensure_project_loaded(root_path::String)
    if PROJECT_LOADED[]
        return
    end

    entry = detect_entry_file(root_path)
    root_module = detect_root_module_name(entry)
    println(stderr, "[DEBUG] Loading root module $(root_module) from $(entry)")
    Base.include(Main, entry)

    if !isdefined(Main, root_module)
        error("Root module $(root_module) not defined after including $(entry)")
    end

    root_mod = getfield(Main, root_module)
    alias_top_level_modules!(root_mod)
    PROJECT_LOADED[] = true
    PROJECT_ROOT[] = root_path
    ROOT_MODULE_NAME[] = root_module
    ROOT_MODULE_REF[] = root_mod
end

function include_target_file(main_path::String)
    if PROJECT_LOADED[] && ROOT_MODULE_REF[] !== nothing
        root = PROJECT_ROOT[]
        if !isempty(root)
            norm_root = normpath(root)
            norm_path = normpath(main_path)
            if startswith(norm_path, norm_root)
                Base.include(ROOT_MODULE_REF[]::Module, main_path)
                return
            end
        end
    end

    Main.include(main_path)
end

include(joinpath(BASE_DIR, "020_ast_cfg.jl"))
include(joinpath(BASE_DIR, "030_ir_ssa.jl"))
include(joinpath(BASE_DIR, "040_build_model.jl"))

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

function collect_julia_files(root_path::String)
    files = String[]
    src_dir = joinpath(root_path, "src")

    if !isdir(src_dir)
        return files
    end

    for (dirpath, _, file_names) in walkdir(src_dir)
        parts = splitpath(dirpath)
        if any(part -> part in SKIP_DIRECTORIES, parts)
            continue
        end

        for fname in file_names
            endswith(fname, ".jl") || continue
            push!(files, joinpath(dirpath, fname))
        end
    end

    sort!(files)
    return files
end

function generate_global_cfgs(root_path::String, dot_root::String)
    ensure_project_loaded(root_path)
    files = collect_julia_files(root_path)

    if isempty(files)
        println(stderr, "[DEBUG] No Julia files discovered under $(root_path)")
        return
    end

    layers = Dict{String, Vector{String}}()

    for file in files
        layer = detect_layer(file)
        push!(get!(layers, layer, String[]), file)
    end

    layers_dir = joinpath(dot_root, "layers")
    project_dir = joinpath(dot_root, "project")
    mkpath(layers_dir)
    mkpath(project_dir)

    println(stderr, "[DEBUG] Generating per-layer CFGs in $(layers_dir)")
    for (layer, paths) in sort(collect(layers); by=x->x[1])
        isempty(paths) && continue
        dot_path = joinpath(layers_dir, "$(layer).dot")
        write_layer_cfg_dot(layer, paths, dot_path)
    end

    println(stderr, "[DEBUG] Generating project-wide CFG in $(project_dir)")
    write_layer_cfg_dot("project", files, joinpath(project_dir, "project_cfg.dot"))
end

function write_all_cfgs(file_path::String, dot_dir::String, title::String)
    println(stderr, "[DEBUG] Generating CFG DOT for $(title)")
    
    mkpath(dot_dir)
    base = replace(basename(file_path), r"\.jl$" => "")
    file_dot = joinpath(dot_dir, "$(base).dot")
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
    if isempty(ARGS)
        println(stderr, "Usage: 00_main.jl <file_path> <dot_directory> <project_root>")
        println(stderr, "   or: 00_main.jl --global-cfgs <project_root> <dot_root>")
        exit(1)
    end

    if ARGS[1] == "--global-cfgs"
        if length(ARGS) < 3
            println(stderr, "Usage: 00_main.jl --global-cfgs <project_root> <dot_root>")
            exit(1)
        end

        root_path = abspath(ARGS[2])
        dot_root = abspath(ARGS[3])
        generate_global_cfgs(root_path, dot_root)
        return
    end

    if length(ARGS) < 3
        println(stderr, "Usage: 00_main.jl <file_path> <dot_directory> <project_root>")
        exit(1)
    end

    file_path = abspath(ARGS[1])
    dot_dir = abspath(ARGS[2])
    root_path = abspath(ARGS[3])

    ensure_project_loaded(root_path)

    all_code = read_all_code(file_path)
    functions, scg = analyze_code(all_code)
    
    run_model(file_path, dot_dir)
    
    elements = convert_functions_to_json(functions, scg, file_path)
    println(JSON.json(elements))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
