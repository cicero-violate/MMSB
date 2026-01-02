use crate::dead_code_doc_comment_parser::item_attrs;
use syn::Item;

pub fn is_cfg_test_item(item: &Item) -> bool {
    item_attrs(item).iter().any(|attr| {
        if !attr.path().is_ident("cfg") {
            return false;
        }
        let mut found = false;
        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("test") {
                found = true;
                return Ok(());
            }
            if meta.path.is_ident("any") {
                meta.parse_nested_meta(|nested| {
                    if nested.path.is_ident("test") {
                        found = true;
                    }
                    Ok(())
                })?;
            }
            Ok(())
        });
        found
    })
}
