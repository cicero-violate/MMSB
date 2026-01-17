// GPU Persistent Kernel for Propagation
// Implements command buffer consumption without per-delta launches

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// Forward declare NCCL types
typedef void* ncclComm_t;

extern "C" {
    int ncclAllReduce(const void*, void*, size_t, int, int, ncclComm_t, void*);
    int ncclAllGather(const void*, void*, size_t, int, ncclComm_t, void*);
}

// Command buffer structures
struct PropagationCommand {
    uint64_t page_id;
    void* page_data;
    void* page_mask;
    uint32_t page_size;
    uint32_t dep_count;
    void** dep_data;
    void** dep_masks;
};

struct CommandBuffer {
    PropagationCommand* commands;
    volatile uint32_t* write_idx;
    volatile uint32_t* read_idx;
    uint32_t capacity;
    volatile uint32_t* shutdown_flag;
    ncclComm_t nccl_comm;
    int device_rank;
};

// Persistent kernel that processes command buffer
__global__ void persistent_propagation_kernel(CommandBuffer* cmd_buf) {
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t num_threads = blockDim.x * gridDim.x;
    
    while (!(*cmd_buf->shutdown_flag)) {
        // Wait for work
        while (*cmd_buf->read_idx == *cmd_buf->write_idx && 
               !(*cmd_buf->shutdown_flag)) {
            __threadfence_system();
        }
        
        if (*cmd_buf->shutdown_flag) break;
        
        // Consume command
        uint32_t local_read = atomicAdd((uint32_t*)cmd_buf->read_idx, 1);
        uint32_t cmd_idx = local_read % cmd_buf->capacity;
        
        if (local_read >= *cmd_buf->write_idx) continue;
        
        __threadfence_system();
        PropagationCommand cmd = cmd_buf->commands[cmd_idx];
        
        // Process command - apply deltas from dependencies
        for (uint32_t dep = 0; dep < cmd.dep_count; dep++) {
            uint8_t* page_data = (uint8_t*)cmd.page_data;
            bool* page_mask = (bool*)cmd.page_mask;
            uint8_t* dep_data = (uint8_t*)cmd.dep_data[dep];
            bool* dep_mask = (bool*)cmd.dep_masks[dep];
            
            // Parallel merge
            for (uint32_t i = tid; i < cmd.page_size; i += num_threads) {
                if (dep_mask[i]) {
                    page_data[i] = dep_data[i];
                    page_mask[i] = true;
                }
            }
        }
        
        __syncthreads();
    }
}

// Host API
extern "C" {

CommandBuffer* create_command_buffer(uint32_t capacity) {
    CommandBuffer* buf;
    cudaMallocManaged(&buf, sizeof(CommandBuffer));
    cudaMallocManaged(&buf->commands, capacity * sizeof(PropagationCommand));
    cudaMallocManaged(&buf->write_idx, sizeof(uint32_t));
    cudaMallocManaged(&buf->read_idx, sizeof(uint32_t));
    cudaMallocManaged(&buf->shutdown_flag, sizeof(uint32_t));
    
    *buf->write_idx = 0;
    *buf->read_idx = 0;
    *buf->shutdown_flag = 0;
    buf->capacity = capacity;
    buf->nccl_comm = nullptr;
    buf->device_rank = 0;
    
    return buf;
}

void set_nccl_communicator(CommandBuffer* buf, ncclComm_t comm, int rank) {
    buf->nccl_comm = comm;
    buf->device_rank = rank;
}

int collective_all_reduce(CommandBuffer* buf, void* sendbuf, void* recvbuf, 
                          size_t count, void* stream) {
    if (!buf->nccl_comm) return -1;
    return ncclAllReduce(sendbuf, recvbuf, count, 
                         1, // ncclUint8
                         0, // ncclSum
                         buf->nccl_comm, stream);
}

int collective_all_gather(CommandBuffer* buf, void* sendbuf, void* recvbuf,
                          size_t sendcount, void* stream) {
    if (!buf->nccl_comm) return -1;
    return ncclAllGather(sendbuf, recvbuf, sendcount,
                         1, // ncclUint8
                         buf->nccl_comm, stream);
}

void destroy_command_buffer(CommandBuffer* buf) {
    *buf->shutdown_flag = 1;
    cudaDeviceSynchronize();
    
    cudaFree(buf->shutdown_flag);
    cudaFree(buf->read_idx);
    cudaFree(buf->write_idx);
    cudaFree(buf->commands);
    cudaFree(buf);
}

void launch_persistent_kernel(CommandBuffer* buf, int num_blocks, int threads_per_block) {
    persistent_propagation_kernel<<<num_blocks, threads_per_block>>>(buf);
}

void enqueue_command(CommandBuffer* buf, PropagationCommand* cmd) {
    uint32_t idx = atomicAdd((uint32_t*)buf->write_idx, 1);
    uint32_t slot = idx % buf->capacity;
    buf->commands[slot] = *cmd;
    __threadfence_system();
}

void wait_queue_empty(CommandBuffer* buf) {
    while (*buf->read_idx < *buf->write_idx) {
        usleep(100);
    }
}

} // extern "C"
