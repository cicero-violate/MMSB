use mmsb_memory::page::{Page, PageAccess};
use mmsb_memory::page::types::PageLocation;
use mmsb_primitives::PageID;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let page = Page::new(
        PageID(1),
        4096,
        PageLocation::Cpu,
    )?;

    // Trait method; trait must be in scope
    println!("Created page {:?}", page.id());

    Ok(())
}
