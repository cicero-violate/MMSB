# ============================================================
# 01_ast_cfg.jl - AST-based Control Flow Graph Extraction
# ============================================================

using MacroTools

# ============================================================
# CFG DATA STRUCTURES
# ============================================================

@enum CFGNodeType begin
    CFG_ENTRY
    CFG_EXIT
    CFG_BASIC_BLOCK
    CFG_BRANCH
    CFG_LOOP_HEADER
end

struct CFGNode
    id::Int
    type::CFGNodeType
    instructions::Vector{Int}  # Line numbers or instruction indices
    label::String
end

struct CFGEdge
    source::Int
    target::Int
    condition::Union{Symbol,Bool,Nothing}  # Accept Symbol, Bool, or nothing
end

struct ControlFlowGraph
    fname::Symbol
    nodes::Vector{CFGNode}
    edges::Vector{CFGEdge}
    entry_id::Int
    exit_id::Int
end

# Global storage
const FUNCTION_CFGS = Dict{Symbol,ControlFlowGraph}()
const AST_NODE_COUNTER = Ref(0)

# ============================================================
# CODE READING & PARSING
# ============================================================

function read_all_code(main_path::String, visited=Set{String}())
    base_dir = dirname(main_path)
    main_code = read(main_path, String)
    all_code = ""
    
    expr = Meta.parseall(main_code)
    MacroTools.prewalk(expr) do ex
        if @capture(ex, include(call_))
            if @capture(call, joinpath(base_, file_))
                inc_path = ""
                if base == :BASE_DIR || base == :__DIR__
                    inc_path = joinpath(base_dir, file)
                elseif isa(base, String)
                    inc_path = joinpath(base_dir, base, file)
                end
                if isfile(inc_path) && !(inc_path in visited)
                    push!(visited, inc_path)
                    inc_code = read_all_code(inc_path, visited)
                    all_code *= "\n# Included from $inc_path\n" * inc_code
                end
            end
        end
        ex
    end
    
    all_code *= "\n# File $main_path\n" * main_code
    return all_code
end

# ============================================================
# AST ANALYSIS - FUNCTIONS & CALL GRAPH
# ============================================================

function analyze_code(code_str::String)
    expr = Meta.parseall(code_str)
    functions = Dict{Symbol, Expr}()
    scg = Vector{Tuple{Symbol, Symbol}}()

    function walk(ex, current_func::Symbol=Symbol(""))
        if isa(ex, Expr)
            # Function definition
            if ex.head == :function
                if length(ex.args) >= 2
                    sig = ex.args[1]
                    body = ex.args[2]
                    
                    # Extract function name
                    fname = if sig isa Symbol
                        sig
                    elseif sig isa Expr && sig.head == :call && length(sig.args) >= 1
                        sig.args[1]
                    else
                        nothing
                    end
                    
                    if fname isa Symbol
                        functions[fname] = ex
                        # Walk body with function name as context
                        walk(body, fname)
                    end
                end
                
            # Short function definition
            elseif ex.head == :(=) && length(ex.args) == 2
                lhs = ex.args[1]
                rhs = ex.args[2]
                
                if lhs isa Expr && lhs.head == :call && length(lhs.args) >= 1
                    fname = lhs.args[1]
                    if fname isa Symbol
                        functions[fname] = ex
                        # Walk RHS with function name as context
                        walk(rhs, fname)
                    end
                end
                
            # Function call
            elseif ex.head == :call && length(ex.args) >= 1
                callee = ex.args[1]
                if callee isa Symbol && current_func != Symbol("")
                    push!(scg, (current_func, callee))
                end
                
                # Walk arguments
                for arg in ex.args[2:end]
                    walk(arg, current_func)
                end
                
            else
                # Recurse into subexpressions
                for arg in ex.args
                    walk(arg, current_func)
                end
            end
        end
    end

    walk(expr)
    
    println("\n=== AST Analysis Results ===")
    println("Functions found: $(length(functions))")
    println("Call edges found: $(length(scg))")
    println("Valid edges (non-empty caller): $(count(e -> e[1] != Symbol(""), scg))")
    println("============================\n")
    
    return functions, scg
end

# ============================================================
# CFG EXTRACTION FROM AST
# ============================================================

function extract_cfg_from_ast(fname::Symbol, body)
    blocks = Vector{Vector{Tuple{Int,Any}}}()
    current_block = Tuple{Int,Any}[]
    
    function walk_statements(ex, line_num::Int=0)
        if ex isa Expr
            if ex.head == :block
                for arg in ex.args
                    if arg isa LineNumberNode
                        line_num = arg.line
                    elseif arg !== nothing
                        walk_statements(arg, line_num)
                    end
                end
                
            elseif ex.head == :if
                !isempty(current_block) && push!(blocks, current_block)
                cond_block = [(line_num, ex)]
                push!(blocks, cond_block)
                current_block = Tuple{Int,Any}[]
                
                if length(ex.args) >= 2
                    walk_statements(ex.args[2], line_num)
                end
                !isempty(current_block) && push!(blocks, current_block)
                
                current_block = Tuple{Int,Any}[]
                if length(ex.args) >= 3
                    walk_statements(ex.args[3], line_num)
                end
                !isempty(current_block) && push!(blocks, current_block)
                current_block = Tuple{Int,Any}[]
                
            elseif ex.head ∈ (:while, :for)
                !isempty(current_block) && push!(blocks, current_block)
                loop_header = [(line_num, ex)]
                push!(blocks, loop_header)
                
                current_block = Tuple{Int,Any}[]
                if length(ex.args) >= 2
                    walk_statements(ex.args[2], line_num)
                end
                !isempty(current_block) && push!(blocks, current_block)
                current_block = Tuple{Int,Any}[]
                
            elseif ex.head == :return
                push!(current_block, (line_num, ex))
                !isempty(current_block) && push!(blocks, current_block)
                current_block = Tuple{Int,Any}[]
                
            else
                push!(current_block, (line_num, ex))
            end
        else
            push!(current_block, (line_num, ex))
        end
    end
    
    walk_statements(body)
    !isempty(current_block) && push!(blocks, current_block)
    
    # Build CFG nodes
    cfg_nodes = CFGNode[]
    cfg_edges = CFGEdge[]
    
    # Entry node
    entry_id = (AST_NODE_COUNTER[] += 1)
    push!(cfg_nodes, CFGNode(entry_id, CFG_ENTRY, Int[], "ENTRY"))
    
    # Exit node  
    exit_id = (AST_NODE_COUNTER[] += 1)
    push!(cfg_nodes, CFGNode(exit_id, CFG_EXIT, Int[], "EXIT"))
    
    # Process blocks
    block_ids = Int[]
    for (i, block) in enumerate(blocks)
        isempty(block) && continue
        
        node_id = (AST_NODE_COUNTER[] += 1)
        push!(block_ids, node_id)
        
        last_stmt = block[end][2]
        node_type = CFG_BASIC_BLOCK
        label = "BB$i"
        
        if last_stmt isa Expr
            if last_stmt.head == :if
                node_type = CFG_BRANCH
                label = "BRANCH$i"
            elseif last_stmt.head ∈ (:while, :for)
                node_type = CFG_LOOP_HEADER
                label = "LOOP$i"
            elseif last_stmt.head == :return
                label = "RETURN$i"
            end
        end
        
        lines = [stmt[1] for stmt in block]
        push!(cfg_nodes, CFGNode(node_id, node_type, lines, label))
    end
    
    # Build edges
    if !isempty(block_ids)
        push!(cfg_edges, CFGEdge(entry_id, block_ids[1], nothing))
        
        for i in 1:(length(block_ids)-1)
            curr_block = blocks[i]
            last_stmt = curr_block[end][2]
            
            if last_stmt isa Expr && last_stmt.head == :if
                if i + 1 <= length(block_ids)
                    push!(cfg_edges, CFGEdge(block_ids[i], block_ids[i+1], :true))  # ← Symbol
                end
                if i + 2 <= length(block_ids)
                    push!(cfg_edges, CFGEdge(block_ids[i], block_ids[i+2], :false))  # ← Symbol
                end
            elseif last_stmt isa Expr && last_stmt.head ∈ (:while, :for)
                if i + 1 <= length(block_ids)
                    push!(cfg_edges, CFGEdge(block_ids[i], block_ids[i+1], :true))   # ← Symbol
                    push!(cfg_edges, CFGEdge(block_ids[i+1], block_ids[i], nothing))
                end
                if i + 2 <= length(block_ids)
                    push!(cfg_edges, CFGEdge(block_ids[i], block_ids[i+2], :false))  # ← Symbol
                end
            elseif !(last_stmt isa Expr && last_stmt.head == :return)
                push!(cfg_edges, CFGEdge(block_ids[i], block_ids[i+1], nothing))
            end
        end
        
        last_block = blocks[end]
        last_stmt = last_block[end][2]
        push!(cfg_edges, CFGEdge(block_ids[end], exit_id, nothing))
    else
        push!(cfg_edges, CFGEdge(entry_id, exit_id, nothing))
    end
    
    return ControlFlowGraph(fname, cfg_nodes, cfg_edges, entry_id, exit_id)
end

# ============================================================
# CFG QUERY FUNCTIONS
# ============================================================

function get_cfg(fname::Symbol)
    return get(FUNCTION_CFGS, fname, nothing)
end

function count_branches(fname::Symbol)
    cfg = get_cfg(fname)
    cfg === nothing && return 0
    return count(n -> n.type == CFG_BRANCH, cfg.nodes)
end

function count_loops(fname::Symbol)
    cfg = get_cfg(fname)
    cfg === nothing && return 0
    return count(n -> n.type == CFG_LOOP_HEADER, cfg.nodes)
end

struct CFGMetrics
    num_nodes::Int
    num_edges::Int
    num_branches::Int
    num_loops::Int
    cyclomatic_complexity::Int
end

function compute_cfg_metrics(fname::Symbol)
    cfg = get_cfg(fname)
    cfg === nothing && return nothing
    
    num_nodes = length(cfg.nodes)
    num_edges = length(cfg.edges)
    num_branches = count_branches(fname)
    num_loops = count_loops(fname)
    
    cyclomatic = num_edges - num_nodes + 2
    
    return CFGMetrics(num_nodes, num_edges, num_branches, num_loops, cyclomatic)
end

println("✓ 01_ast_cfg.jl loaded")
