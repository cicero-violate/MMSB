//! Advanced mutation operations for fine-grained transformations

use super::MutationOp;
use syn::{Item, FnArg, Pat};

/// Replace function parameters by name or position
#[derive(Debug, Clone)]
pub struct ParamReplaceOp {
    /// Parameter selector: either name or index
    pub selector: ParamSelector,
    /// New parameter definition (e.g., "new_name: NewType")
    pub replacement: String,
}

#[derive(Debug, Clone)]
pub enum ParamSelector {
    /// Match by parameter name
    ByName(String),
    /// Match by position (0-indexed)
    ByIndex(usize),
    /// Match all parameters
    All,
}

impl ParamReplaceOp {
    pub fn by_name(name: impl Into<String>, replacement: impl Into<String>) -> Self {
        Self {
            selector: ParamSelector::ByName(name.into()),
            replacement: replacement.into(),
        }
    }
    
    pub fn by_index(index: usize, replacement: impl Into<String>) -> Self {
        Self {
            selector: ParamSelector::ByIndex(index),
            replacement: replacement.into(),
        }
    }
}

impl MutationOp for ParamReplaceOp {
    fn apply(&self, item: &Item) -> String {
        match item {
            Item::Fn(func) => {
                let mut new_func = func.clone();
                
                // Parse replacement parameter
                let replacement_param = format!("fn dummy({}) {{}}", self.replacement);
                let Ok(parsed) = syn::parse_str::<syn::ItemFn>(&replacement_param) else {
                    // If parsing fails, return original
                    return quote::quote!(#func).to_string();
                };
                
                let new_param = if let Some(FnArg::Typed(pat_type)) = parsed.sig.inputs.first() {
                    FnArg::Typed(pat_type.clone())
                } else {
                    return quote::quote!(#func).to_string();
                };
                
                // Apply replacement based on selector
                match &self.selector {
                    ParamSelector::ByName(target_name) => {
                        // Replace parameter with matching name
                        for param in &mut new_func.sig.inputs {
                            if let FnArg::Typed(pat_type) = param {
                                if let Pat::Ident(ident) = &*pat_type.pat {
                                    if ident.ident == target_name {
                                        *param = new_param.clone();
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    ParamSelector::ByIndex(idx) => {
                        // Replace parameter at specific index
                        if *idx < new_func.sig.inputs.len() {
                            new_func.sig.inputs[*idx] = new_param;
                        }
                    }
                    ParamSelector::All => {
                        // Replace all parameters
                        new_func.sig.inputs.clear();
                        new_func.sig.inputs.push(new_param);
                    }
                }
                
                quote::quote!(#new_func).to_string()
            }
            _ => {
                // Not a function, return unchanged
                quote::quote!(#item).to_string()
            }
        }
    }
}

/// Replace expressions in function body using pattern matching
#[derive(Debug, Clone)]
pub struct BodyReplaceOp {
    /// Pattern to match (simplified - matches identifier usage)
    pub pattern: String,
    /// Replacement expression
    pub replacement: String,
}

impl BodyReplaceOp {
    pub fn new(pattern: impl Into<String>, replacement: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            replacement: replacement.into(),
        }
    }
}

impl MutationOp for BodyReplaceOp {
    fn apply(&self, item: &Item) -> String {
        match item {
           Item::Fn(func) => {
                let mut new_func = func.clone();
                
                // Get function body
                let block = &mut new_func.block;
                // Simple text-based replacement in body
                let body_str = quote::quote!(#block).to_string();
                let new_body_str = body_str.replace(&self.pattern, &self.replacement);
                
                // Parse back to block
                if let Ok(new_block) = syn::parse_str::<syn::Block>(&new_body_str) {
                    new_func.block = Box::new(new_block);
                }
                
                quote::quote!(#new_func).to_string()
            }
            _ => {
                // Not a function, return unchanged
                quote::quote!(#item).to_string()
            }
        }
    }
}

/// Add parameter to function signature
#[derive(Debug, Clone)]
pub struct AddParamOp {
    /// New parameter to add (e.g., "flag: bool")
    pub param: String,
    /// Where to insert
    pub position: ParamPosition,
}

#[derive(Debug, Clone)]
pub enum ParamPosition {
    First,
    Last,
    At(usize),
}

impl AddParamOp {
    pub fn at_end(param: impl Into<String>) -> Self {
        Self {
            param: param.into(),
            position: ParamPosition::Last,
        }
    }
    
    pub fn at_start(param: impl Into<String>) -> Self {
        Self {
            param: param.into(),
            position: ParamPosition::First,
        }
    }
}

impl MutationOp for AddParamOp {
    fn apply(&self, item: &Item) -> String {
        match item {
            Item::Fn(func) => {
                let mut new_func = func.clone();
                
                // Parse new parameter
                let param_fn = format!("fn dummy({}) {{}}", self.param);
                let Ok(parsed) = syn::parse_str::<syn::ItemFn>(&param_fn) else {
                    return quote::quote!(#func).to_string();
                };
                
                if let Some(FnArg::Typed(pat_type)) = parsed.sig.inputs.first() {
                    let new_param = FnArg::Typed(pat_type.clone());
                    
                    match self.position {
                        ParamPosition::First => {
                            new_func.sig.inputs.insert(0, new_param);
                        }
                        ParamPosition::Last => {
                            new_func.sig.inputs.push(new_param);
                        }
                        ParamPosition::At(idx) => {
                            if idx <= new_func.sig.inputs.len() {
                                new_func.sig.inputs.insert(idx, new_param);
                            } else {
                                new_func.sig.inputs.push(new_param);
                            }
                        }
                    }
                }
                
                quote::quote!(#new_func).to_string()
            }
            _ => {
                quote::quote!(#item).to_string()
            }
        }
    }
}

/// Remove parameter from function signature
#[derive(Debug, Clone)]
pub struct RemoveParamOp {
    pub selector: ParamSelector,
}

impl RemoveParamOp {
    pub fn by_name(name: impl Into<String>) -> Self {
        Self {
            selector: ParamSelector::ByName(name.into()),
        }
    }
    
    pub fn by_index(index: usize) -> Self {
        Self {
            selector: ParamSelector::ByIndex(index),
        }
    }
}

impl MutationOp for RemoveParamOp {
    fn apply(&self, item: &Item) -> String {
        match item {
            Item::Fn(func) => {
                let mut new_func = func.clone();
                
                match &self.selector {
                    ParamSelector::ByName(target_name) => {
                        new_func.sig.inputs = new_func.sig.inputs.into_iter()
                            .filter(|param| {
                                if let FnArg::Typed(pat_type) = param {
                                    if let Pat::Ident(ident) = &*pat_type.pat {
                                        return ident.ident != target_name;
                                    }
                                }
                                true
                            })
                            .collect();
                    }
                    ParamSelector::ByIndex(idx) => {
                        if *idx < new_func.sig.inputs.len() {
                            new_func.sig.inputs = new_func.sig.inputs.into_iter()
                                .enumerate()
                                .filter(|(i, _)| i != idx)
                                .map(|(_, param)| param)
                                .collect();
                        }
                    }
                    ParamSelector::All => {
                        new_func.sig.inputs.clear();
                    }
                }
                
                quote::quote!(#new_func).to_string()
            }
            _ => {
                quote::quote!(#item).to_string()
            }
        }
    }
}
