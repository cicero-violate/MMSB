use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use syn::visit::Visit;
use syn::{ItemEnum, ItemFn, ItemImpl, ItemMod, ItemStruct, ItemTrait, ItemUse};

use crate::types::{
    AnalysisResult, CfgEdge, CfgNode, CodeElement, ElementType, FunctionCfg, Language, ModuleInfo,
    NodeType, Visibility,
};

pub struct RustAnalyzer {
    root_path: PathBuf,
}

impl RustAnalyzer {
    pub fn new(root_path: String) -> Self {
        Self {
            root_path: PathBuf::from(root_path),
        }
    }

    pub fn analyze_file(&self, file_path: &Path) -> Result<AnalysisResult> {
        let content = fs::read_to_string(file_path)
            .with_context(|| format!("Failed to read file: {:?}", file_path))?;

        let syntax_tree = syn::parse_file(&content)
            .with_context(|| format!("Failed to parse Rust file: {:?}", file_path))?;

        let mut result = AnalysisResult::new();
        let layer = self.extract_layer(file_path);
        let file_path_str = relativize_path(file_path, &self.root_path);

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

fn relativize_path(path: &Path, root: &Path) -> String {
    if let Ok(stripped) = path.strip_prefix(root) {
        stripped.to_string_lossy().to_string()
    } else {
        path.to_string_lossy().to_string()
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
                    let name = path_expr
                        .path
                        .segments
                        .iter()
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

    fn ensure_module_entry(&mut self) -> usize {
        if let Some(idx) = self
            .result
            .modules
            .iter()
            .position(|m| m.file_path == self.file_path)
        {
            idx
        } else {
            let module = ModuleInfo {
                name: self
                    .file_path
                    .split('/')
                    .last()
                    .unwrap_or(&self.file_path)
                    .trim_end_matches(".rs")
                    .to_string(),
                file_path: self.file_path.clone(),
                imports: Vec::new(),
                exports: Vec::new(),
                submodules: Vec::new(),
            };
            self.result.modules.push(module);
            self.result.modules.len() - 1
        }
    }
}

impl<'a> Visit<'_> for RustVisitor<'a> {
    fn visit_item_struct(&mut self, node: &ItemStruct) {
        let generic_params = node
            .generics
            .params
            .iter()
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
        let generic_params = node
            .generics
            .params
            .iter()
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
            format!(
                "{} for {}",
                path.segments.last().unwrap().ident,
                quote::quote!(#node.self_ty)
            )
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

        let generic_params = node
            .sig
            .generics
            .params
            .iter()
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

        let cfg = CfgExtractor::from_function(
            node.sig.ident.to_string(),
            self.file_path.clone(),
            &node.block,
        );
        self.result.add_cfg(cfg);

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

        let submodule_name = node.ident.to_string();
        let idx = self.ensure_module_entry();
        if !self.result.modules[idx]
            .submodules
            .contains(&submodule_name)
        {
            self.result.modules[idx].submodules.push(submodule_name);
        }

        syn::visit::visit_item_mod(self, node);
    }

    fn visit_item_use(&mut self, node: &ItemUse) {
        let stmt = quote::quote!(#node).to_string();
        let idx = self.ensure_module_entry();

        match node.vis {
            syn::Visibility::Public(_) => self.result.modules[idx].exports.push(stmt),
            _ => self.result.modules[idx].imports.push(stmt),
        }

        syn::visit::visit_item_use(self, node);
    }
}

struct CfgExtractor {
    nodes: Vec<CfgNode>,
    edges: Vec<CfgEdge>,
    next_id: usize,
    branch_count: usize,
    loop_count: usize,
}

impl CfgExtractor {
    fn from_function(function: String, file_path: String, block: &syn::Block) -> FunctionCfg {
        let mut extractor = Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            next_id: 0,
            branch_count: 0,
            loop_count: 0,
        };

        let entry_id = extractor.add_node("ENTRY".to_string(), NodeType::Entry, vec![]);
        let body_exit = extractor.build_block(&block.stmts, entry_id);
        let exit_id = extractor.add_node("EXIT".to_string(), NodeType::Exit, vec![]);
        extractor.add_edge(body_exit, exit_id, None);

        FunctionCfg {
            function,
            file_path,
            entry_id,
            exit_id,
            nodes: extractor.nodes,
            edges: extractor.edges,
            branch_count: extractor.branch_count,
            loop_count: extractor.loop_count,
        }
    }

    fn add_node(&mut self, label: String, node_type: NodeType, lines: Vec<u32>) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(CfgNode { id, node_type, label, lines });
        id
    }

    fn add_edge(&mut self, from: usize, to: usize, condition: Option<bool>) {
        self.edges.push(CfgEdge { from, to, condition });
    }

    fn build_block(&mut self, stmts: &[syn::Stmt], from: usize) -> usize {
        let mut cursor = from;
        for stmt in stmts {
            cursor = self.process_stmt(stmt, cursor);
        }
        cursor
    }

    fn process_stmt(&mut self, stmt: &syn::Stmt, from: usize) -> usize {
        match stmt {
            syn::Stmt::Local(local) => {
                let mut label = format!("let {}", pat_snippet(&local.pat));
                if let Some(init) = &local.init {
                    label.push_str(" = ");
                    label.push_str(&expr_snippet(&init.expr));
                    if let Some((_else_token, diverge)) = &init.diverge {
                        label.push_str(" else ");
                        label.push_str(&expr_snippet(diverge));
                    }
                }
                let node_id = self.add_node(label, NodeType::BasicBlock, vec![]);
                self.add_edge(from, node_id, None);
                node_id
            }
            syn::Stmt::Item(item) => {
                let label = match item {
                    syn::Item::Struct(i) => format!("struct {}", i.ident),
                    syn::Item::Enum(i) => format!("enum {}", i.ident),
                    syn::Item::Trait(i) => format!("trait {}", i.ident),
                    syn::Item::Impl(_) => "impl block".to_string(),
                    syn::Item::Use(_) => "use".to_string(),
                    syn::Item::Mod(i) => format!("mod {}", i.ident),
                    _ => "item".to_string(),
                };
                let node_id = self.add_node(label, NodeType::BasicBlock, vec![]);
                self.add_edge(from, node_id, None);
                node_id
            }
            syn::Stmt::Expr(expr, _) => self.process_expr(expr, from),
            syn::Stmt::Macro(mac) => {
                let label = format!(
                    "macro {}",
                    mac.mac
                        .path
                        .segments
                        .last()
                        .map(|s| s.ident.to_string())
                        .unwrap_or_else(|| "?".into())
                );
                let node_id = self.add_node(label, NodeType::BasicBlock, vec![]);
                self.add_edge(from, node_id, None);
                node_id
            }
        }
    }

    fn process_expr(&mut self, expr: &syn::Expr, from: usize) -> usize {
        match expr {
            syn::Expr::If(expr_if) => {
                self.branch_count += 1;
                let cond_label = truncate_label(format!("if {}", expr_snippet(&expr_if.cond)));
                let branch_id = self.add_node(cond_label, NodeType::Branch, vec![]);
                self.add_edge(from, branch_id, None);

                // Then branch
                let then_start = self.add_node("THEN BB".to_string(), NodeType::BasicBlock, vec![]);
                self.add_edge(branch_id, then_start, Some(true));
                let then_exit = self.build_block(&expr_if.then_branch.stmts, then_start);

                // Else branch (or direct from branch if no else)
                let else_exit = if let Some((_, else_branch)) = &expr_if.else_branch {
                    let else_start = self.add_node("ELSE BB".to_string(), NodeType::BasicBlock, vec![]);
                    self.add_edge(branch_id, else_start, Some(false));
                    self.process_expr(else_branch, else_start)
                } else {
                    let empty_else = self.add_node("EMPTY ELSE".to_string(), NodeType::BasicBlock, vec![]);
                    self.add_edge(branch_id, empty_else, Some(false));
                    empty_else
                };

                // Join
                let join_id = self.add_node("IF JOIN".to_string(), NodeType::BasicBlock, vec![]);
                self.add_edge(then_exit, join_id, None);
                self.add_edge(else_exit, join_id, None);
                join_id
            }
            syn::Expr::Loop(expr_loop) => {
                self.loop_count += 1;
                let loop_label = expr_loop.label.as_ref().map_or("LOOP".to_string(), |l| format!("LOOP {}", l.name.ident));
                let header_id = self.add_node(loop_label, NodeType::LoopHeader, vec![]);
                self.add_edge(from, header_id, None);

                let body_start = self.add_node("LOOP BB".to_string(), NodeType::BasicBlock, vec![]);
                self.add_edge(header_id, body_start, None);
                let body_exit = self.build_block(&expr_loop.body.stmts, body_start);

                // Back edge
                self.add_edge(body_exit, header_id, None);

                // Exit after loop
                let after_id = self.add_node("AFTER LOOP".to_string(), NodeType::BasicBlock, vec![]);
                self.add_edge(body_exit, after_id, None);  // Break would go here, but simplified
                after_id
            }
            // Add more for while, for, match, etc., as needed
            _ => {
                let label = truncate_label(expr_snippet(expr));
                let node_id = self.add_node(label, NodeType::BasicBlock, vec![]);
                self.add_edge(from, node_id, None);
                node_id
            }
        }
    }
}

fn expr_snippet(expr: &syn::Expr) -> String {
    truncate_label(quote::quote!(#expr).to_string())
}

fn pat_snippet(pat: &syn::Pat) -> String {
    truncate_label(quote::quote!(#pat).to_string())
}

fn truncate_label(text: String) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut label = collapsed;
    if label.len() > 80 {
        label.truncate(77);
        label.push_str("...");
    }
    label
}
