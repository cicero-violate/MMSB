use mmsb_memory::{
    delta::{Delta, Source, ColumnarDeltaBatch},
    epoch::Epoch,
};

use mmsb_primitives::{DeltaID, PageID};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let deltas = vec![
        Delta::new_dense(
            DeltaID(1),
            PageID(1),
            Epoch(1),
            vec![1, 2],
            vec![true, true],
            Source("a".into()),
        )?,
        Delta::new_dense(
            DeltaID(2),
            PageID(2),
            Epoch(1),
            vec![3, 4],
            vec![true, true],
            Source("b".into()),
        )?,
    ];

    let batch = ColumnarDeltaBatch::from_rows(deltas);

    println!("Batch size: {}", batch.len());
    println!("Epoch=1 matches: {:?}", batch.filter_epoch_eq(Epoch(1)));

    Ok(())
}
