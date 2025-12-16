#!/usr/bin/env julia
# Suppress startup messages
ENV["JULIA_STARTUP_FILE"] = "no"

"""
Julia Structure Analyzer
Called by Rust analyzer to extract Julia AST information
"""

using JSON

struct JuliaElement
    element_type::String
    name::String
    file_path::String
    line_number::Int
    signature::String
    calls::Vector{String}
end

function extract_calls(expr)
    """Extract function calls from expression"""
    calls = String[]
    
    function walk(e)
        if isa(e, Expr)
            if e.head == :call && length(e.args) > 0
                # First arg is the function name
                fname = e.args[1]
                if isa(fname, Symbol)
                    push!(calls, String(fname))
                elseif isa(fname, Expr) && fname.head == :.
                    # Module.function call
                    push!(calls, string(fname))
                end
            end
            # Recurse into sub-expressions
            for arg in e.args
                walk(arg)
            end
        end
    end
    
    walk(expr)
    return unique(calls)
end

function analyze_julia_file(filepath::String)
    """Analyze a single Julia file and extract structures"""
    elements = JuliaElement[]
    
    try
        content = read(filepath, String)
        lines = split(content, '\n')
        
        # Parse the entire file
        try
            parsed = Meta.parse("begin\n" * content * "\nend")
            
            # Walk through top-level expressions
            if isa(parsed, Expr) && parsed.head == :block
                line_num = 1
                
                for expr in parsed.args
                    if isa(expr, LineNumberNode)
                        line_num = expr.line
                        continue
                    end
                    
                    if !isa(expr, Expr)
                        continue
                    end
                    
                    # Struct definitions
                    if expr.head == :struct
                        mutable = length(expr.args) > 0 && expr.args[1] == true
                        struct_name = if length(expr.args) >= 2
                            name_expr = expr.args[2]
                            if isa(name_expr, Symbol)
                                String(name_expr)
                            elseif isa(name_expr, Expr) && name_expr.head == :<:
                                String(name_expr.args[1])
                            else
                                "Unknown"
                            end
                        else
                            "Unknown"
                        end
                        
                        sig = mutable ? "mutable struct" : "struct"
                        push!(elements, JuliaElement(
                            "struct",
                            struct_name,
                            filepath,
                            line_num,
                            sig,
                            String[]
                        ))
                    end
                    
                    # Function definitions
                    if expr.head == :function || (expr.head == :(=) && isa(expr.args[1], Expr) && expr.args[1].head == :call)
                        func_expr = expr.head == :function ? expr.args[1] : expr.args[1]
                        
                        if isa(func_expr, Expr) && func_expr.head == :call
                            func_name = if isa(func_expr.args[1], Symbol)
                                String(func_expr.args[1])
                            else
                                "Anonymous"
                            end
                            
                            # Extract signature
                            sig_parts = String[]
                            for arg in func_expr.args[2:end]
                                if isa(arg, Symbol)
                                    push!(sig_parts, String(arg))
                                elseif isa(arg, Expr)
                                    push!(sig_parts, string(arg))
                                end
                            end
                            signature = func_name * "(" * join(sig_parts, ", ") * ")"
                            
                            # Extract function calls in body
                            body = expr.head == :function ? expr.args[2] : expr.args[2]
                            calls = extract_calls(body)
                            
                            push!(elements, JuliaElement(
                                "function",
                                func_name,
                                filepath,
                                line_num,
                                signature,
                                calls
                            ))
                        end
                    end
                    
                    # Module definitions
                    if expr.head == :module
                        module_name = String(expr.args[2])
                        push!(elements, JuliaElement(
                            "module",
                            module_name,
                            filepath,
                            line_num,
                            "module",
                            String[]
                        ))
                    end
                end
            end
        catch parse_error
            # If parsing fails, fall back to regex-based extraction
            for (i, line) in enumerate(lines)
                # Struct definitions
                m = match(r"^\s*(?:mutable\s+)?struct\s+(\w+)", line)
                if m !== nothing
                    is_mutable = occursin("mutable", line)
                    push!(elements, JuliaElement(
                        "struct",
                        m.captures[1],
                        filepath,
                        i,
                        is_mutable ? "mutable struct" : "struct",
                        String[]
                    ))
                end
                
                # Function definitions
                m = match(r"^\s*function\s+(\w+)", line)
                if m !== nothing
                    push!(elements, JuliaElement(
                        "function",
                        m.captures[1],
                        filepath,
                        i,
                        line,
                        String[]
                    ))
                end
            end
        end
        
    catch e
        println(stderr, "Error analyzing $filepath: $e")
    end
    
    return elements
end

function main()
    if length(ARGS) < 1
        println(stderr, "Usage: julia_analyzer.jl <file.jl>")
        exit(1)
    end
    
    filepath = ARGS[1]
    println(stderr, "[Julia analyzer] Processing $filepath")
    elements = analyze_julia_file(filepath)
    
    # Convert to JSON for Rust consumption
    json_output = JSON.json(map(e -> Dict(
        "element_type" => e.element_type,
        "name" => e.name,
        "file_path" => e.file_path,
        "line_number" => e.line_number,
        "signature" => e.signature,
        "calls" => e.calls
    ), elements))
    
    println(json_output)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
