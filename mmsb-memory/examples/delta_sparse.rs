use mmsb_memory::{
    delta::{Delta, Source},
    epoch::Epoch,
};

use mmsb_primitives::{DeltaID, PageID};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let delta = Delta::new_sparse(
        DeltaID(2),
        PageID(42),
        Epoch(7),
        vec![false, true, false, true], // mask
        vec![0xAA, 0xBB],               // payload (only changed slots)
        Source("sparse-example".to_string()),
    )?;

    println!("Sparse delta constructed: {:?}", delta.delta_id);

    Ok(())
}
