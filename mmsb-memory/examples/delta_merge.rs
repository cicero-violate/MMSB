use mmsb_memory::{
    delta::{Delta, Source},
    epoch::Epoch,
};

use mmsb_primitives::{DeltaID, PageID};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base = Delta::new_dense(
        DeltaID(1),
        PageID(1),
        Epoch(1),
        vec![1, 2, 3],
        vec![true, true, true],
        Source("base".into()),
    )?;

    let overlay = Delta::new_dense(
        DeltaID(2),
        PageID(1),
        Epoch(2),
        vec![9, 9, 9],
        vec![false, true, false],
        Source("overlay".into()),
    )?;

    let merged = base.merge(&overlay)?;

    println!("Merged epoch: {:?}", merged.epoch);
    println!("Merged payload: {:?}", merged.to_dense());

    Ok(())
}
