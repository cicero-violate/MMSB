# ============================================================
# SCOPE CONFIGURATION
# ============================================================
@enum AnalysisScope begin
    SCOPE_TARGET_ONLY # Only functions in target file
    SCOPE_WITH_DEPS # Include called Base/Compiler functions
    SCOPE_FULL_CLOSURE # Transitive closure of all calls
end
# ============================================================
# HELPER FUNCTIONS
# ============================================================
function sanitize_fname(s)
    replace(string(s), r"[^a-zA-Z0-9_]" => "_")
end
function dot_escape(s)
    s = replace(s, "\\" => "\\\\")
    s = replace(s, "\"" => "\\\"")
    s = replace(s, "\n" => "\\n")
    s = replace(s, "\r" => "\\r")
    s = replace(s, "{" => "\\{")
    s = replace(s, "}" => "\\}")
    s = replace(s, "<" => "\\<")
    s = replace(s, ">" => "\\>")
    s = replace(s, "|" => "\\|")
    return s
end
"""
Extract all function calls from IR including Base/Core calls
"""
function extract_all_calls_from_ir(ir, fname::Symbol)
    calls = Tuple{Symbol, Union{Module,Nothing}}[]
   
    for i in 1:length(ir.stmts)
        stmt = ir.stmts[i][:stmt]
       
        # Direct function calls
        if stmt isa Expr && stmt.head == :call
            if length(stmt.args) >= 1
                callee = stmt.args[1]
               
                # Handle GlobalRef (Base.abs, Core.println, etc.)
                if callee isa GlobalRef
                    push!(calls, (callee.name, callee.mod))
                elseif callee isa Symbol
                    push!(calls, (callee, nothing))
                end
            end
        end
       
        # Method info from CallInfo
        info = ir.stmts[i][:info]
       
        if info isa Core.Compiler.CallInfo
            mi = try
                Core.Compiler.extract_methodinstance(info)
            catch
                nothing
            end
           
            if mi isa Core.MethodInstance
                meth = mi.def
                if meth isa Method
                    mod = meth.module
                    push!(calls, (meth.name, mod))
                end
            end
        end
    end
   
    return unique(calls)
end
"""
Discover functions from module with optional depth
"""
function discover_functions_in_module(
    mod::Module;
    include_private::Bool = false,
    max_depth::Int = 1000,
    visited::Set{Module} = Set{Module}()
)
    push!(visited, mod)
    registry = Dict{Symbol, Tuple{Module, Any}}()
   
    all_names = if include_private
        names(mod, all=true, imported=false)
    else
        names(mod, all=false, imported=false)
    end
   
    for name in all_names
        name_str = string(name)
       
        # Skip compiler-generated
        startswith(name_str, "#") && continue
        startswith(name_str, "var\"#") && continue
        name in (:eval, :include) && continue
       
        if !isdefined(mod, name)
            continue
        end
       
        obj = try
            getfield(mod, name)
        catch
            continue
        end
       
        # Found function
        if obj isa Function
            registry[name] = (mod, obj)
        end
       
        # Recurse into submodules
        if max_depth > 0 && obj isa Module && obj ∉ visited && obj ∉ (Core, Base)
            submap = discover_functions_in_module(
                obj,
                include_private=include_private,
                max_depth=max_depth-1,
                visited=visited
            )
            merge!(registry, submap)
        end
    end
   
    return registry
end

"""
Build registry with configurable scope
"""
function build_registry_with_scope(
    ast_functions::Dict{Symbol, Expr},
    scope::AnalysisScope = SCOPE_TARGET_ONLY
)
    registry = Dict{Symbol, Tuple{Module, Any}}()
   
    println("\n=== Building Registry (Scope: $scope) ===")
   
    # Phase 1: Target functions from AST
    for fname in keys(ast_functions)
        if isdefined(Main, fname)
            obj = try
                getfield(Main, fname)
            catch e
                nothing
            end
           
            if obj isa Function
                registry[fname] = (Main, obj)
                println(" ✓ $fname (Main)")
            end
        end
    end
   
    target_count = length(registry)
    println("Target functions: $target_count")
   
    # Phase 2: Add dependencies based on scope
    if scope == SCOPE_WITH_DEPS || scope == SCOPE_FULL_CLOSURE
        println("\nDiscovering dependencies...")
       
        # Determine which functions to analyze
        functions_to_analyze = if scope == SCOPE_WITH_DEPS
            # Only analyze target functions once
            Set(keys(registry))
        else
            # For FULL_CLOSURE, we'll expand iteratively
            Set(keys(registry))
        end
        
        # Track functions we've already analyzed
        analyzed = Set{Symbol}()
        
        # Iterative expansion for SCOPE_FULL_CLOSURE
        iteration = 1
        max_iterations = 100  # Safety limit
        
        while !isempty(functions_to_analyze) && iteration <= max_iterations
            if scope == SCOPE_FULL_CLOSURE
                println("\n  Iteration $iteration: Analyzing $(length(functions_to_analyze)) functions")
            end
            
            # Collect all called functions from IR
            interp = Core.Compiler.NativeInterpreter()
            all_calls = Set{Tuple{Symbol, Union{Module,Nothing}}}()
            
            for fname in functions_to_analyze
                # Skip if already analyzed
                fname ∈ analyzed && continue
                
                if !haskey(registry, fname)
                    continue
                end
                
                mod, f = registry[fname]
                
                # Get IR for this function
                try
                    mts = methods(f)
                    if !isempty(mts)
                        m = mts.ms[1]
                        sig = m.sig
                        argtypes = [Base.unwrap_unionall(p) for p in sig.parameters[2:end]]
                       
                        # Type inference
                        atype = Tuple{typeof(f), argtypes...}
                        matches = Core.Compiler.find_method_matches(interp, argtypes, atype)
                       
                        if !isempty(matches.info.results.matches)
                            mm = matches.info.results.matches[1]
                            ir, _ = Core.Compiler.typeinf_ircode(interp, mm, nothing)
                           
                            if ir !== nothing
                                calls = extract_all_calls_from_ir(ir, fname)
                                union!(all_calls, calls)
                            end
                        end
                    end
                catch
                    continue
                finally
                    push!(analyzed, fname)
                end
            end
            
            # Add called functions to registry
            new_functions = Set{Symbol}()
            
            for (callee_name, callee_mod) in all_calls
                # Skip if already in registry
                haskey(registry, callee_name) && continue
               
                # Determine module to search
                search_mod = if callee_mod !== nothing
                    callee_mod
                elseif isdefined(Main, callee_name)
                    Main
                elseif isdefined(Base, callee_name)
                    Base
                elseif isdefined(Core, callee_name)
                    Core
                else
                    nothing
                end
               
                if search_mod !== nothing && isdefined(search_mod, callee_name)
                    obj = try
                        getfield(search_mod, callee_name)
                    catch
                        nothing
                    end
                   
                    if obj isa Function
                        registry[callee_name] = (search_mod, obj)
                        push!(new_functions, callee_name)
                        
                        if scope == SCOPE_FULL_CLOSURE
                            println("    + $callee_name ($(nameof(search_mod)))")
                        else
                            println(" + $callee_name ($(nameof(search_mod)))")
                        end
                    end
                end
            end
            
            # For SCOPE_WITH_DEPS, stop after first iteration
            if scope == SCOPE_WITH_DEPS
                break
            end
            
            # For SCOPE_FULL_CLOSURE, prepare next iteration
            if scope == SCOPE_FULL_CLOSURE
                if isempty(new_functions)
                    println("\n  Convergence reached: No new functions discovered")
                    break
                end
                
                # Analyze newly discovered functions in next iteration
                functions_to_analyze = new_functions
                iteration += 1
            end
        end
        
        if iteration > max_iterations
            println("\n  ⚠️ Reached maximum iteration limit ($max_iterations)")
        end
       
        deps_count = length(registry) - target_count
        println("\nDependencies added: $deps_count")
        
        if scope == SCOPE_FULL_CLOSURE
            println("Total iterations: $iteration")
            println("Functions analyzed: $(length(analyzed))")
        end
    end
   
    println("\nFinal registry: $(length(registry)) functions")
    println("======================================\n")
   
    return registry
end

# ============================================================
# ENHANCED EXPORT WITH SCOPE CONTROL
# ============================================================
function build_model(
    main_path::String;
    scope::AnalysisScope = SCOPE_TARGET_ONLY,
    show_external_calls::Bool = true
)
    println("\n" * "="^60)
    println("BUILDING PROGRAM MODEL")
    println("="^60)
    println("Input: $main_path")
    println("Scope: $scope")
    println("Show external calls: $show_external_calls")
    println("")
   
    # Step 1: AST analysis
    all_code = read_all_code(main_path)
    functions, scg = analyze_code(all_code)
    valid_scg = filter(e -> e[1] != Symbol(""), scg)
    sorted_funcs = sort(collect(keys(functions)))
   
    println("=== AST Discovery ===")
    println("Functions: $(length(sorted_funcs))")
    println("=====================\n")
   
    # Step 2: Load into runtime
    try
        include_target_file(main_path)
        println("✓ Loaded: $main_path\n")
    catch e
        println("✗ Load failed:")
        showerror(stdout, e)
        return nothing
    end
   
    yield()
   
    # Step 3: Build registry with scope
    function_registry = build_registry_with_scope(functions, scope)
   
    if isempty(function_registry)
        println("⚠️ No functions in registry!")
        return nothing
    end
   
    # Step 4: Compilation
    println("=== Compilation ===")
    compiled = Symbol[]
   
    for fname in keys(function_registry)
        mod, f = function_registry[fname]
       
        try
            mts = methods(f)
            if !isempty(mts)
                for m in mts
                    try
                        sig = m.sig
                        if sig isa DataType && length(sig.parameters) > 1
                            argtypes = Tuple{sig.parameters[2:end]...}
                            code_typed(f, argtypes)
                            push!(compiled, fname)
                            break
                        end
                    catch
                        continue
                    end
                end
            end
        catch
            continue
        end
    end
   
    println("Compiled: $(length(compiled)) / $(length(function_registry))")
    println("===================\n")
   
    # Step 5: IR collection
    interp = Core.Compiler.NativeInterpreter()
    IR_NODE_COUNTER[] = 0
    ir_data = Dict{String, Any}()
   
    println("=== IR Collection ===")
   
    for fname in keys(function_registry)
        mod, f = function_registry[fname]
        fdata = collect_ir_data(interp, f, fname)
       
        if fdata !== nothing
            ir_data[string(fname)] = fdata
        end
    end
   
    println("IR collected: $(length(ir_data)) functions")
    println("=====================\n")
   
    # Build model data structure (suitable for Graphviz DOT generation or Makie visualization)
    target_funcs = sort(collect(keys(functions)))
    dep_funcs = sort([f for f in keys(function_registry) if !(f in keys(functions))])
    all_funcs = vcat(target_funcs, dep_funcs)
   
    model = Dict(
        "all_functions" => all_funcs,
        "target_functions" => target_funcs,
        "dependency_functions" => dep_funcs,
        "ir_data" => ir_data,
        "call_graph" => valid_scg,
        "function_registry" => function_registry,
        "static_call_graph_raw" => scg
    )
   
    println("✓ Model built with $(length(all_funcs)) functions")
    println("="^60 * "\n")
   
    return model
end
