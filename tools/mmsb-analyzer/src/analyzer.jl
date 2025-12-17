#!/usr/bin/env julia
# Hybrid Julia analyzer: emits JSON summary + Graphviz DOT call graphs

const BASE_DIR = @__DIR__

include(joinpath(BASE_DIR, "01_ast_cfg.jl"))
include(joinpath(BASE_DIR, "02_ir_ssa.jl"))
include(joinpath(BASE_DIR, "03_build_model.jl"))

using JSON
using Printf

mutable struct Element
    element_type::String
    name::String
    file_path::String
    line_number::Int
    signature::String
    calls::Vector{String}
end

const CALL_RE = r"([A-Za-z_][A-Za-z0-9_\.!]*)\s*\("
const KEYWORDS = Set([
    "if", "for", "while", "begin", "let", "struct", "mutable", "quote", "macro",
    "module", "baremodule", "end", "where"
])

function normalize_calls(body::AbstractString)
    calls = String[]
    for m in eachmatch(CALL_RE, body)
        ident = m.captures[1]
        (ident in KEYWORDS) && continue
        push!(calls, ident)
    end
    sort!(unique!(calls))
    return calls
end

function parse_inline_function(line::String, file_path::String, line_number::Int)
    m = match(r"^\s*([A-Za-z_][A-Za-z0-9_\.!]*)\s*(\([^=]*\))\s*=\s*(.+)$", line)
    m === nothing && return nothing
    fname = m.captures[1]
    signature = string(fname, m.captures[2])
    body = m.captures[3]
    calls = normalize_calls(body)
    return Element("function", fname, file_path, line_number, strip(signature), calls)
end

function parse_block_function(lines::Vector{SubString{String}}, idx::Int, file_path::String)
    line = String(lines[idx])
    fname_match = match(r"^\s*function\s+([A-Za-z_][A-Za-z0-9_\.!]*)", line)
    fname_match === nothing && return idx, nothing
    fname = fname_match.captures[1]
    start_idx = idx
    depth = 1
    body_lines = String[]
    idx += 1
    while idx <= length(lines)
        current = String(lines[idx])
        stripped = strip(current)
        if startswith(stripped, "function ")
            depth += 1
        elseif stripped == "end" || startswith(stripped, "end ")
            depth -= 1
            if depth == 0
                idx += 1
                break
            end
        end
        push!(body_lines, current)
        idx += 1
    end
    signature = strip(line)
    calls = normalize_calls(join(body_lines, "\n"))
    elem = Element("function", fname, file_path, start_idx, signature, calls)
    return idx, elem
end

function parse_julia_file(file_path::String)
    content = read(file_path, String)
    lines = split(content, '\n')
    elements = Element[]
    idx = 1
    while idx <= length(lines)
        raw = String(lines[idx])
        stripped = strip(raw)
        if isempty(stripped) || startswith(stripped, "#")
            idx += 1
            continue
        end

        if (m = match(r"^\s*(module|baremodule)\s+(\w+)", raw)) !== nothing
            push!(elements, Element("module", m.captures[2], file_path, idx, m.captures[1], String[]))
            idx += 1
            continue
        end

        if (m = match(r"^\s*(mutable\s+)?struct\s+(\w+)", raw)) !== nothing
            sig = if something(m.captures[1], "") == ""
                "struct"
            else
                "mutable struct"
            end
            push!(elements, Element("struct", m.captures[2], file_path, idx, sig, String[]))
            idx += 1
            continue
        end

        if startswith(stripped, "function ")
            new_idx, elem = parse_block_function(lines, idx, file_path)
            idx = new_idx
            elem !== nothing && push!(elements, elem)
            continue
        end

        if startswith(stripped, "macro ")
            idx += 1
            continue
        end

        if startswith(stripped, "baremodule ")
            idx += 1
            continue
        end

        if startswith(stripped, "mutable struct ") || startswith(stripped, "struct ")
            idx += 1
            continue
        end

        if startswith(stripped, "const ")
            idx += 1
            continue
        end

        if startswith(stripped, "type ") || startswith(stripped, "abstract type ")
            idx += 1
            continue
        end

        if (elem = parse_inline_function(raw, file_path, idx)) !== nothing
            push!(elements, elem)
        end
        idx += 1
    end
    return elements
end

function augment_calls_with_model!(elements::Vector{Element}, model)
    model === nothing && return
    call_graph = get(model, "call_graph", Tuple{Symbol,Symbol}[])
    calls_by_func = Dict{String, Vector{String}}()
    for (caller, callee) in call_graph
        cname = string(caller)
        calls = get!(calls_by_func, cname, String[])
        push!(calls, string(callee))
    end
    for (_, calls) in calls_by_func
        sort!(unique!(calls))
    end
    for elem in elements
        elem.element_type == "function" || continue
        graph_calls = get(calls_by_func, elem.name, nothing)
        graph_calls === nothing && continue
        elem.calls = graph_calls
    end
end

function write_call_graph_dot(model, dot_dir::String, title::String)
    println(stderr, "[DEBUG] write_call_graph_dot called")
    println(stderr, "[DEBUG]   dot_dir: $dot_dir")
    println(stderr, "[DEBUG]   title: $title")
    
    mkpath(dot_dir)
    println(stderr, "[DEBUG]   ✓ Directory created/verified: $dot_dir")
    
    dot_path = joinpath(dot_dir, "call_graph.dot")
    println(stderr, "[DEBUG]   Full dot path: $dot_path")
    
    target_funcs = Set(Symbol.(get(model, "target_functions", Symbol[])))
    println(stderr, "[DEBUG]   Target functions count: $(length(target_funcs))")
    
    edges = get(model, "call_graph", Tuple{Symbol,Symbol}[])
    println(stderr, "[DEBUG]   Edges count: $(length(edges))")
    
    all_nodes = Set{Symbol}()
    foreach(f -> push!(all_nodes, f), target_funcs)
    for (src, dst) in edges
        (src in target_funcs) || continue
        push!(all_nodes, dst)
    end
    println(stderr, "[DEBUG]   Total nodes: $(length(all_nodes))")
    
    if isempty(all_nodes)
        println(stderr, "[DEBUG]   ⚠ No nodes to write - skipping DOT file")
        return
    end
    
    println(stderr, "[DEBUG]   Opening file for writing...")
    open(dot_path, "w") do io
        title_escaped = replace(title, "\"" => "\\\"")
        println(io, "digraph \"$(title_escaped)\" {")
        println(io, "    rankdir=LR;")
        for node in sort(collect(all_nodes))
            label = replace(string(node), "\"" => "\\\"")
            shape = node in target_funcs ? "box" : "ellipse"
            println(io, "    \"$(label)\" [shape=$(shape)];")
        end
        for (src, dst) in edges
            (src in target_funcs) || continue
            println(io, @sprintf("    \"%s\" -> \"%s\";", replace(string(src), "\"" => "\\\""), replace(string(dst), "\"" => "\\\"")))
        end
        println(io, "}")
    end
    println(stderr, "[DEBUG]   ✓ File written: $dot_path")
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
        println(stderr, "  Exception type: ", typeof(e))
        println(stderr, "  Backtrace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stderr, exc, bt)
            println(stderr)
        end
        model = nothing
    end
    
    # Always try to generate DOT from AST, regardless of model success
    # The AST analysis already extracted the call graph successfully
    write_dot_from_ast(file_path, dot_dir, basename(file_path))
    
    return model
end

function emit_json(elements::Vector{Element})
    dicts = [
        Dict(
            "element_type" => elem.element_type,
            "name" => elem.name,
            "file_path" => elem.file_path,
            "line_number" => elem.line_number,
            "signature" => elem.signature,
            "calls" => elem.calls,
        ) for elem in elements
    ]
    println(JSON.json(dicts))
end

function main()
    if length(ARGS) < 2
        println(stderr, "Usage: analyzer.jl <file_path> <dot_directory>")
        exit(1)
    end
    file_path = abspath(ARGS[1])
    dot_dir = abspath(ARGS[2])
    elements = parse_julia_file(file_path)
    model = run_model(file_path, dot_dir)
    augment_calls_with_model!(elements, model)
    emit_json(elements)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
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
