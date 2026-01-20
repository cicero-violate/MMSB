use mmsb_memory::{
    delta::{Delta, Source},
    epoch::Epoch,
    page::{Page, DeltaAppliable, PageAccess},
    page::types::PageLocation,
};

use mmsb_primitives::{DeltaID, PageID};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut page = Page::new(PageID(1), 3, PageLocation::Cpu)?;

    let delta = Delta::new_dense(
        DeltaID(1),
        PageID(1),
        Epoch(1),
        vec![10, 20, 30],
        vec![true, true, true],
        Source("apply".into()),
    )?;

    page.apply_delta(&delta)?;

    // Trait method: PageAccess must be in scope
    println!("Page data: {:?}", page.data_slice());

    Ok(())
}
