//! Rust AST parser using syn

use crate::types::*;
use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use syn::visit::Visit;
use syn::{File, Item, ItemFn, ItemImpl, ItemStruct, ItemEnum, ItemTrait, ItemMod, ItemUse};

pub struct RustAnalyzer {
    root_path: String,
}

impl RustAnalyzer {
    pub fn new(root_path: String) -> Self {
        Self { root_path }
    }
    
    pub fn analyze_file(&self, file_path: &Path) -> Result<AnalysisResult> {
        let content = fs::read_to_string(file_path)
            .with_context(|| format!("Failed to read file: {:?}", file_path))?;
        
        let syntax_tree = syn::parse_file(&content)
            .with_context(|| format!("Failed to parse Rust file: {:?}", file_path))?;
        
        let mut result = AnalysisResult::new();
        let layer = self.extract_layer(file_path);
        let file_path_str = file_path.to_string_lossy().to_string();
        
        let mut visitor = RustVisitor {
            file_path: file_path_str.clone(),
            layer: layer.clone(),
            result: &mut result,
        };
        
        visitor.visit_file(&syntax_tree);
        
        Ok(result)
    }
    
    fn extract_layer(&self, path: &Path) -> String {
        for component in path.components() {
            if let Some(name) = component.as_os_str().to_str() {
                if name.chars().next().map_or(false, |c| c.is_ascii_digit()) {
                    if let Some(pos) = name.find('_') {
                        if name[..pos].chars().all(|c| c.is_ascii_digit()) {
                            return name.to_string();
                        }
                    }
                }
            }
        }
        "root".to_string()
    }
}

struct RustVisitor<'a> {
    file_path: String,
    layer: String,
    result: &'a mut AnalysisResult,
}

impl<'a> RustVisitor<'a> {
    fn get_visibility(&self, vis: &syn::Visibility) -> Visibility {
        match vis {
            syn::Visibility::Public(_) => Visibility::Public,
            syn::Visibility::Restricted(r) => {
                if let Some(first) = r.path.segments.first() {
                    if first.ident == "crate" {
                        return Visibility::Crate;
                    }
                }
                Visibility::Private
            }
            syn::Visibility::Inherited => Visibility::Private,
        }
    }
    
    fn extract_function_calls(&self, block: &syn::Block) -> Vec<String> {
        let mut calls = Vec::new();
        
        struct CallVisitor<'b> {
            calls: &'b mut Vec<String>,
        }
        
        impl<'b> Visit<'_> for CallVisitor<'b> {
            fn visit_expr_call(&mut self, node: &syn::ExprCall) {
                if let syn::Expr::Path(path_expr) = &*node.func {
                    let name = path_expr.path.segments.iter()
                        .map(|s| s.ident.to_string())
                        .collect::<Vec<_>>()
                        .join("::");
                    self.calls.push(name);
                }
                syn::visit::visit_expr_call(self, node);
            }
            
            fn visit_expr_method_call(&mut self, node: &syn::ExprMethodCall) {
                self.calls.push(node.method.to_string());
                syn::visit::visit_expr_method_call(self, node);
            }
        }
        
        let mut call_visitor = CallVisitor { calls: &mut calls };
        call_visitor.visit_block(block);
        
        calls
    }
}

impl<'a> Visit<'_> for RustVisitor<'a> {
    fn visit_item_struct(&mut self, node: &ItemStruct) {
        let generic_params = node.generics.params.iter()
            .filter_map(|p| match p {
                syn::GenericParam::Type(t) => Some(t.ident.to_string()),
                syn::GenericParam::Lifetime(l) => Some(l.lifetime.to_string()),
                _ => None,
            })
            .collect();
        
        self.result.add_element(CodeElement {
            element_type: ElementType::Struct,
            name: node.ident.to_string(),
            file_path: self.file_path.clone(),
            line_number: 0, // syn doesn't provide line numbers easily
            language: Language::Rust,
            layer: self.layer.clone(),
            signature: quote::quote!(#node).to_string(),
            calls: Vec::new(),
            visibility: self.get_visibility(&node.vis),
            generic_params,
        });
        
        syn::visit::visit_item_struct(self, node);
    }
    
    fn visit_item_enum(&mut self, node: &ItemEnum) {
        let generic_params = node.generics.params.iter()
            .filter_map(|p| match p {
                syn::GenericParam::Type(t) => Some(t.ident.to_string()),
                _ => None,
            })
            .collect();
        
        self.result.add_element(CodeElement {
            element_type: ElementType::Enum,
            name: node.ident.to_string(),
            file_path: self.file_path.clone(),
            line_number: 0,
            language: Language::Rust,
            layer: self.layer.clone(),
            signature: format!("enum {}", node.ident),
            calls: Vec::new(),
            visibility: self.get_visibility(&node.vis),
            generic_params,
        });
        
        syn::visit::visit_item_enum(self, node);
    }
    
    fn visit_item_trait(&mut self, node: &ItemTrait) {
        self.result.add_element(CodeElement {
            element_type: ElementType::Trait,
            name: node.ident.to_string(),
            file_path: self.file_path.clone(),
            line_number: 0,
            language: Language::Rust,
            layer: self.layer.clone(),
            signature: format!("trait {}", node.ident),
            calls: Vec::new(),
            visibility: self.get_visibility(&node.vis),
            generic_params: Vec::new(),
        });
        
        syn::visit::visit_item_trait(self, node);
    }
    
    fn visit_item_impl(&mut self, node: &ItemImpl) {
        let impl_name = if let Some((_, path, _)) = &node.trait_ {
            format!("{} for {}", 
                path.segments.last().unwrap().ident,
                quote::quote!(#node.self_ty))
        } else {
            quote::quote!(#node.self_ty).to_string()
        };
        
        self.result.add_element(CodeElement {
            element_type: ElementType::Impl,
            name: impl_name,
            file_path: self.file_path.clone(),
            line_number: 0,
            language: Language::Rust,
            layer: self.layer.clone(),
            signature: quote::quote!(#node).to_string(),
            calls: Vec::new(),
            visibility: Visibility::Private,
            generic_params: Vec::new(),
        });
        
        syn::visit::visit_item_impl(self, node);
    }
    
    fn visit_item_fn(&mut self, node: &ItemFn) {
        let calls = self.extract_function_calls(&node.block);
        
        let generic_params = node.sig.generics.params.iter()
            .filter_map(|p| match p {
                syn::GenericParam::Type(t) => Some(t.ident.to_string()),
                syn::GenericParam::Lifetime(l) => Some(l.lifetime.to_string()),
                _ => None,
            })
            .collect();
        
        self.result.add_element(CodeElement {
            element_type: ElementType::Function,
            name: node.sig.ident.to_string(),
            file_path: self.file_path.clone(),
            line_number: 0,
            language: Language::Rust,
            layer: self.layer.clone(),
            signature: quote::quote!(#node.sig).to_string(),
            calls,
            visibility: self.get_visibility(&node.vis),
            generic_params,
        });
        
        syn::visit::visit_item_fn(self, node);
    }
    
    fn visit_item_mod(&mut self, node: &ItemMod) {
        self.result.add_element(CodeElement {
            element_type: ElementType::Module,
            name: node.ident.to_string(),
            file_path: self.file_path.clone(),
            line_number: 0,
            language: Language::Rust,
            layer: self.layer.clone(),
            signature: format!("mod {}", node.ident),
            calls: Vec::new(),
            visibility: self.get_visibility(&node.vis),
            generic_params: Vec::new(),
        });
        
        syn::visit::visit_item_mod(self, node);
    }
}
