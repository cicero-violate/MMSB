module PropagationScheduler

"""
    schedule!(engine, commands)

Simple placeholder scheduler that enqueues commands onto the propagation queue.
"""
function schedule!(engine, commands)
    for cmd in commands
        engine.enqueue(cmd)
    end
    engine.drain()
end

end # module
