# ============================================================
# 02_ir_ssa.jl - IR/SSA/PHI Analysis with Global Node IDs
# ============================================================

using Core.Compiler

# ============================================================
# GLOBAL NODE COUNTER FOR IR-BASED NODES
# ============================================================

const IR_NODE_COUNTER = Ref(0)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

function compute_cc(edges, nodes)
    return max(1, length(edges) - length(nodes) + 2)
end

# ============================================================
# IR CALL EXTRACTION
# ============================================================

function extract_ir_calls(ir, fname)
    call_edges = Tuple{Symbol,Symbol}[]

    for i in 1:length(ir.stmts)
        stmt = ir.stmts[i][:stmt]
        info = ir.stmts[i][:info]

        if !(info isa Core.Compiler.CallInfo)
            continue
        end

        mi = try
            Core.Compiler.extract_methodinstance(info)
        catch
            nothing
        end

        mi isa Core.MethodInstance || continue
        meth = mi.def
        meth isa Method || continue

        callee = meth.name
        push!(call_edges, (fname, callee))
    end

    return call_edges
end

# ============================================================
# COLLECT IR DATA WITH GLOBAL NODE IDs
# ============================================================

function collect_ir_data(interp, f, fname)
    mts = methods(f)
    isempty(mts) && return nothing
    
    mt = mts.ms[1]
    sig = mt.sig
    arg_types = [Base.unwrap_unionall(p) for p in sig.parameters[2:end]]
    arg_types_any = convert(Vector{Any}, arg_types)
    atype = Tuple{typeof(f), arg_types...}
    
    # ============================================================
    # GET METHOD MATCHES
    # ============================================================
    matches = try
        Core.Compiler.find_method_matches(interp, arg_types_any, atype)
    catch e
        println("    ⚠️  $fname: find_method_matches failed")
        return nothing
    end
    
    if isempty(matches.info.results.matches)
        println("    ⚠️  $fname: No method matches")
        return nothing
    end
    
    # ============================================================
    # GET IR
    # ============================================================
    mm = matches.info.results.matches[1]
    
    ir, ret = try
        Core.Compiler.typeinf_ircode(interp, mm, nothing)
    catch e
        println("    ⚠️  $fname: typeinf_ircode failed: $e")
        return nothing
    end
    
    if ir === nothing
        println("    ⚠️  $fname: IR is nothing (inference failed)")
        return nothing
    end
    
    # ============================================================
    # EXTRACT IR COMPONENTS
    # ============================================================
    stmts = ir.stmts
    blocks = ir.cfg.blocks
    
    if isempty(blocks)
        println("    ⚠️  $fname: No basic blocks in IR")
        return nothing
    end
    
    # ============================================================
    # ASSIGN GLOBAL NODE IDs FOR IR
    # ============================================================
    entry_id = (IR_NODE_COUNTER[] += 1)
    exit_id = (IR_NODE_COUNTER[] += 1)
    
    # Assign global IDs to each basic block
    block_global_ids = Int[]
    for bb in blocks
        bb_id = (IR_NODE_COUNTER[] += 1)
        push!(block_global_ids, bb_id)
    end
    
    # ============================================================
    # BUILD EDGES
    # ============================================================
    
    # CFG edges: block-to-block control flow (using GLOBAL IDs)
    CFG_edges = Tuple{Int, Int}[]
    for (bb_idx, block) in enumerate(blocks)
        isempty(block.stmts) && continue
        src_global = block_global_ids[bb_idx]
        
        for succ_idx in block.succs
            if succ_idx <= length(blocks) && !isempty(blocks[succ_idx].stmts)
                dst_global = block_global_ids[succ_idx]
                push!(CFG_edges, (src_global, dst_global))
            end
        end
    end
    
    # Intra-block edges: sequential statement flow (SSA indices)
    intra_CFG_edges = Tuple{Int, Int}[]
    for block in blocks
        for j in 1:length(block.stmts)-1
            src = block.stmts[j]
            dst = block.stmts[j+1]
            push!(intra_CFG_edges, (src, dst))
        end
    end
    
    # SSA edges: data dependencies (SSA indices)
    SSA_edges = Tuple{Int, Int}[]
    for i in 1:length(stmts)
        stmt = stmts[i][:stmt]
        for ur in Core.Compiler.userefs(stmt)
            v = ur[]
            isa(v, Core.SSAValue) && push!(SSA_edges, (v.id, i))
        end
    end
    
    # PHI edges: phi node inputs (SSA indices)
    PHI_edges = Tuple{Int, Int}[]
    for i in 1:length(stmts)
        stmt = stmts[i][:stmt]
        if stmt isa Core.PhiNode
            for v in stmt.values
                isa(v, Core.SSAValue) && push!(PHI_edges, (v.id, i))
            end
        end
    end
    
    # IR call edges
    call_edges = extract_ir_calls(ir, fname)
    
    # Return nodes
    return_stmts = [i for i in 1:length(stmts) if isa(stmts[i][:stmt], Core.ReturnNode)]
    
    # Cyclomatic Complexity
    num_nodes = length(blocks) + 2
    num_edges = length(intra_CFG_edges) + length(CFG_edges) + length(return_stmts) + 1
    cc = compute_cc(num_edges, num_nodes)
    
    return Dict(
        "stmts" => stmts,
        "blocks" => blocks,
        "entry_id" => entry_id,
        "exit_id" => exit_id,
        "block_global_ids" => block_global_ids,
        "intra_CFG_edges" => intra_CFG_edges,
        "CFG_edges" => CFG_edges,
        "SSA_edges" => SSA_edges,
        "PHI_edges" => PHI_edges,
        "return_stmts" => return_stmts,
        "call_edges" => call_edges,
        "cc" => cc
    )
end

# ============================================================
# BATCH IR COLLECTION
# ============================================================

function collect_all_ir_data(function_list::Vector{Symbol})
    interp = Core.Compiler.NativeInterpreter()
    
    # Reset global counter
    IR_NODE_COUNTER[] = 0
    
    ir_data = Dict{String, Any}()
    
    println("\n=== Collecting IR Data ===")
    
    for fname in function_list
        if isdefined(Main, fname)
            f = getglobal(Main, fname)
            fdata = collect_ir_data(interp, f, fname)
            if fdata !== nothing
                ir_data[string(fname)] = fdata
                println("  ✓ $fname: $(length(fdata["blocks"])) blocks, $(length(fdata["SSA_edges"])) SSA edges, $(length(fdata["PHI_edges"])) PHI edges")
            else
                println("  ✗ $fname: No IR data (not compiled or no methods)")
            end
        else
            println("  ✗ $fname: Not defined in Main")
        end
    end
    
    println("Total functions with IR: $(length(ir_data))")
    println("==========================\n")
    
    return ir_data
end

# ============================================================
# IR QUERY FUNCTIONS
# ============================================================

function get_ir_data(fname::Symbol, ir_data::Dict)
    return get(ir_data, string(fname), nothing)
end

function has_ssa_edges(fname::Symbol, ir_data::Dict)
    fdata = get_ir_data(fname, ir_data)
    return fdata !== nothing && !isempty(fdata["SSA_edges"])
end

function has_phi_nodes(fname::Symbol, ir_data::Dict)
    fdata = get_ir_data(fname, ir_data)
    return fdata !== nothing && !isempty(fdata["PHI_edges"])
end

function count_ssa_edges(fname::Symbol, ir_data::Dict)
    fdata = get_ir_data(fname, ir_data)
    return fdata === nothing ? 0 : length(fdata["SSA_edges"])
end

function count_phi_edges(fname::Symbol, ir_data::Dict)
    fdata = get_ir_data(fname, ir_data)
    return fdata === nothing ? 0 : length(fdata["PHI_edges"])
end

function get_basic_blocks_ir(fname::Symbol, ir_data::Dict)
    fdata = get_ir_data(fname, ir_data)
    return fdata === nothing ? [] : fdata["blocks"]
end

# ============================================================
# IR STATISTICS
# ============================================================

struct IRStats
    total_functions::Int
    functions_with_ssa::Int
    functions_with_phi::Int
    total_ssa_edges::Int
    total_phi_edges::Int
    total_basic_blocks::Int
    avg_cc::Float64
end

function compute_ir_stats(ir_data::Dict)
    total_functions = length(ir_data)
    functions_with_ssa = 0
    functions_with_phi = 0
    total_ssa_edges = 0
    total_phi_edges = 0
    total_basic_blocks = 0
    total_cc = 0
    
    for (fname, fdata) in ir_data
        !isempty(fdata["SSA_edges"]) && (functions_with_ssa += 1)
        !isempty(fdata["PHI_edges"]) && (functions_with_phi += 1)
        total_ssa_edges += length(fdata["SSA_edges"])
        total_phi_edges += length(fdata["PHI_edges"])
        total_basic_blocks += length(fdata["blocks"])
        total_cc += fdata["cc"]
    end
    
    avg_cc = total_functions > 0 ? total_cc / total_functions : 0.0
    
    return IRStats(
        total_functions,
        functions_with_ssa,
        functions_with_phi,
        total_ssa_edges,
        total_phi_edges,
        total_basic_blocks,
        avg_cc
    )
end

function print_ir_stats(ir_data::Dict)
    stats = compute_ir_stats(ir_data)
    
    println("\n" * "="^60)
    println("IR ANALYSIS STATISTICS")
    println("="^60)
    println("Total functions analyzed:     $(stats.total_functions)")
    println("Functions with SSA edges:     $(stats.functions_with_ssa)")
    println("Functions with PHI nodes:     $(stats.functions_with_phi)")
    println("Total SSA edges:              $(stats.total_ssa_edges)")
    println("Total PHI edges:              $(stats.total_phi_edges)")
    println("Total basic blocks:           $(stats.total_basic_blocks)")
    println("Average cyclomatic complexity: $(round(stats.avg_cc, digits=2))")
    println("="^60 * "\n")
end

println("✓ 02_ir_ssa.jl loaded")
