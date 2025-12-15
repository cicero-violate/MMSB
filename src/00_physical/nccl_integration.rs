use parking_lot::Mutex;
use std::collections::HashMap;
use std::ffi::c_void;

// NCCL opaque types
type ncclComm_t = *mut c_void;
type ncclUniqueId = [u8; 128];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum ncclRedOp_t {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum ncclDataType_t {
    Int8 = 0,
    Uint8 = 1,
    Int32 = 2,
    Uint32 = 3,
    Int64 = 4,
    Uint64 = 5,
    Float32 = 6,
    Float64 = 7,
}

extern "C" {
    fn ncclGetUniqueId(id: *mut ncclUniqueId) -> i32;
    fn ncclCommInitRank(comm: *mut ncclComm_t, ndev: i32, id: ncclUniqueId, rank: i32) -> i32;
    fn ncclCommDestroy(comm: ncclComm_t) -> i32;
    fn ncclAllReduce(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        count: usize,
        datatype: ncclDataType_t,
        op: ncclRedOp_t,
        comm: ncclComm_t,
        stream: *mut c_void,
    ) -> i32;
    fn ncclAllGather(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        sendcount: usize,
        datatype: ncclDataType_t,
        comm: ncclComm_t,
        stream: *mut c_void,
    ) -> i32;
}

#[derive(Debug)]
pub struct NCCLCommunicator {
    comm: ncclComm_t,
    rank: i32,
    world_size: i32,
}

#[derive(Debug)]
pub struct NCCLContext {
    communicators: Mutex<HashMap<i32, NCCLCommunicator>>,
    unique_id: ncclUniqueId,
}

impl NCCLContext {
    pub fn new(num_gpus: i32) -> Result<Self, i32> {
        let mut unique_id = [0u8; 128];
        let result = unsafe { ncclGetUniqueId(&mut unique_id) };
        
        if result != 0 {
            return Err(result);
        }
        
        Ok(Self {
            communicators: Mutex::new(HashMap::new()),
            unique_id,
        })
    }
    
    pub fn init_communicator(&self, rank: i32, world_size: i32) -> Result<(), i32> {
        let mut comm: ncclComm_t = std::ptr::null_mut();
        let result = unsafe {
            ncclCommInitRank(&mut comm, world_size, self.unique_id, rank)
        };
        
        if result != 0 {
            return Err(result);
        }
        
        let communicator = NCCLCommunicator {
            comm,
            rank,
            world_size,
        };
        
        self.communicators.lock().insert(rank, communicator);
        Ok(())
    }
    
    pub fn all_reduce(
        &self,
        rank: i32,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: usize,
        datatype: ncclDataType_t,
        op: ncclRedOp_t,
        stream: *mut c_void,
    ) -> Result<(), i32> {
        let comms = self.communicators.lock();
        let comm = comms.get(&rank).ok_or(-1)?;
        
        let result = unsafe {
            ncclAllReduce(sendbuf, recvbuf, count, datatype, op, comm.comm, stream)
        };
        
        if result == 0 { Ok(()) } else { Err(result) }
    }
    
    pub fn all_gather(
        &self,
        rank: i32,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        sendcount: usize,
        datatype: ncclDataType_t,
        stream: *mut c_void,
    ) -> Result<(), i32> {
        let comms = self.communicators.lock();
        let comm = comms.get(&rank).ok_or(-1)?;
        
        let result = unsafe {
            ncclAllGather(sendbuf, recvbuf, sendcount, datatype, comm.comm, stream)
        };
        
        if result == 0 { Ok(()) } else { Err(result) }
    }
}

impl Drop for NCCLContext {
    fn drop(&mut self) {
        let comms = self.communicators.lock();
        for comm in comms.values() {
            unsafe { ncclCommDestroy(comm.comm) };
        }
    }
}
