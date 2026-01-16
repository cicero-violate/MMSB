//! Advanced query predicates

use super::predicate::Predicate;
use syn::{Item, ItemFn, ItemStruct, ItemEnum, GenericParam};

/// Match items that have generic parameters
#[derive(Debug, Clone)]
pub struct GenericPredicate {
    /// If Some, match only items with this exact number of generics
    /// If None, match any item with at least one generic
    pub count: Option<usize>,
}

impl GenericPredicate {
    pub fn new() -> Self {
        Self { count: None }
    }
    
    pub fn with_count(count: usize) -> Self {
        Self { count: Some(count) }
    }
    
    pub fn any() -> Self {
        Self::new()
    }
}

impl Default for GenericPredicate {
    fn default() -> Self {
        Self::new()
    }
}

impl Predicate for GenericPredicate {
    fn matches(&self, item: &Item) -> bool {
        let generic_count = match item {
            Item::Fn(ItemFn { sig, .. }) => {
                sig.generics.params.len()
            }
            Item::Struct(ItemStruct { generics, .. }) => {
                generics.params.len()
            }
            Item::Enum(ItemEnum { generics, .. }) => {
                generics.params.len()
            }
            Item::Trait(t) => {
                t.generics.params.len()
            }
            Item::Impl(i) => {
                i.generics.params.len()
            }
            Item::Type(t) => {
                t.generics.params.len()
            }
            _ => return false,
        };
        
        match self.count {
            Some(expected) => generic_count == expected,
            None => generic_count > 0,
        }
    }
}

/// Match functions by signature properties
#[derive(Debug, Clone)]
pub struct SignaturePredicate {
    pub min_params: Option<usize>,
    pub max_params: Option<usize>,
    pub has_return_type: Option<bool>,
    pub is_async: Option<bool>,
    pub is_const: Option<bool>,
    pub is_unsafe: Option<bool>,
}

impl SignaturePredicate {
    pub fn new() -> Self {
        Self {
            min_params: None,
            max_params: None,
            has_return_type: None,
            is_async: None,
            is_const: None,
            is_unsafe: None,
        }
    }
    
    pub fn with_param_count(min: usize, max: usize) -> Self {
        Self {
            min_params: Some(min),
            max_params: Some(max),
            ..Self::new()
        }
    }
    
    pub fn min_params(mut self, min: usize) -> Self {
        self.min_params = Some(min);
        self
    }
    
    pub fn max_params(mut self, max: usize) -> Self {
        self.max_params = Some(max);
        self
    }
    
    pub fn returns(mut self, has_return: bool) -> Self {
        self.has_return_type = Some(has_return);
        self
    }
    
    pub fn is_async(mut self) -> Self {
        self.is_async = Some(true);
        self
    }
    
    pub fn is_const(mut self) -> Self {
        self.is_const = Some(true);
        self
    }
    
    pub fn is_unsafe(mut self) -> Self {
        self.is_unsafe = Some(true);
        self
    }
}

impl Default for SignaturePredicate {
    fn default() -> Self {
        Self::new()
    }
}

impl Predicate for SignaturePredicate {
    fn matches(&self, item: &Item) -> bool {
        let Item::Fn(func) = item else {
            return false;
        };
        
        let sig = &func.sig;
        
        // Check parameter count
        let param_count = sig.inputs.len();
        if let Some(min) = self.min_params {
            if param_count < min {
                return false;
            }
        }
        if let Some(max) = self.max_params {
            if param_count > max {
                return false;
            }
        }
        
        // Check return type
        if let Some(expects_return) = self.has_return_type {
            let has_return = !matches!(sig.output, syn::ReturnType::Default);
            if has_return != expects_return {
                return false;
            }
        }
        
        // Check async
        if let Some(expects_async) = self.is_async {
            let is_async = sig.asyncness.is_some();
            if is_async != expects_async {
                return false;
            }
        }
        
        // Check const
        if let Some(expects_const) = self.is_const {
            let is_const = sig.constness.is_some();
            if is_const != expects_const {
                return false;
            }
        }
        
        // Check unsafe
        if let Some(expects_unsafe) = self.is_unsafe {
            let is_unsafe = sig.unsafety.is_some();
            if is_unsafe != expects_unsafe {
                return false;
            }
        }
        
        true
    }
}
