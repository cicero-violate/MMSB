module DeviceFallback
using CUDA
using ..DeviceSync: GPUCommandBuffer
export has_gpu_support, fallback_to_cpu, CPUPropagationQueue
has_gpu_support()=CUDA.functional()
struct CPUPropagationQueue
commands::Vector{Any}
end
CPUPropagationQueue()=CPUPropagationQueue(Any[])
fallback_to_cpu(f)=(try;f();catch e;@warn "GPU failed, using CPU" error=e;true;end)
end
