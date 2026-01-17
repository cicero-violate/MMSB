module GraphDSL

export graph, node, edge

macro graph(expr)
    return expr
end

node(id) = id
edge(parent, child, weight=:default) = (parent, child, weight)

end # module
