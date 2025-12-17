//! Rust AST parser using syn

use crate::types::*;
use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use syn::visit::Visit;
use syn::{ItemEnum, ItemFn, ItemImpl, ItemMod, ItemStruct, ItemTrait, ItemUse};

pub struct RustAnalyzer {
    _root_path: String,
}

impl RustAnalyzer {
    pub fn new(root_path: String) -> Self {
        Self {
            _root_path: root_path,
        }
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
    edges: Vec<(usize, usize)>,
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

        let entry = extractor.add_node("ENTRY".to_string());
        let exit_anchor = extractor.build_block(&block.stmts, entry);
        let exit = extractor.add_node("EXIT".to_string());
        extractor.add_edge(exit_anchor, exit);

        FunctionCfg {
            function,
            file_path,
            nodes: extractor.nodes,
            edges: extractor.edges,
            branch_count: extractor.branch_count,
            loop_count: extractor.loop_count,
        }
    }

    fn add_node(&mut self, label: String) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(CfgNode { id, label });
        id
    }

    fn add_edge(&mut self, from: usize, to: usize) {
        self.edges.push((from, to));
    }

    fn build_block(&mut self, stmts: &[syn::Stmt], current: usize) -> usize {
        let mut cursor = current;
        for stmt in stmts {
            cursor = self.process_stmt(stmt, cursor);
        }
        cursor
    }

    fn process_stmt(&mut self, stmt: &syn::Stmt, current: usize) -> usize {
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
                let node = self.add_node(label);
                self.add_edge(current, node);
                node
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
                let node = self.add_node(label);
                self.add_edge(current, node);
                node
            }
            syn::Stmt::Expr(expr, _) => self.process_expr(expr, current),
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
                let node = self.add_node(label);
                self.add_edge(current, node);
                node
            }
        }
    }

    fn process_expr(&mut self, expr: &syn::Expr, current: usize) -> usize {
        match expr {
            syn::Expr::If(expr_if) => {
                self.branch_count += 1;
                let cond_label = truncate_label(format!("if {}", expr_snippet(&expr_if.cond)));
                let cond_id = self.add_node(cond_label);
                self.add_edge(current, cond_id);

                let then_exit = self.build_block(&expr_if.then_branch.stmts, cond_id);
                let mut exits = vec![then_exit];

                if let Some((_, else_branch)) = &expr_if.else_branch {
                    exits.push(self.process_expr(else_branch, cond_id));
                } else {
                    exits.push(cond_id);
                }

                let join = self.add_node("if join".to_string());
                for exit in exits {
                    self.add_edge(exit, join);
                }
                join
            }
            syn::Expr::ForLoop(expr_for) => {
                self.loop_count += 1;
                let label = truncate_label(format!(
                    "for {} in {}",
                    pat_snippet(&expr_for.pat),
                    expr_snippet(&expr_for.expr)
                ));
                let loop_node = self.add_node(label);
                self.add_edge(current, loop_node);
                let body_exit = self.build_block(&expr_for.body.stmts, loop_node);
                self.add_edge(body_exit, loop_node);
                let exit = self.add_node("after for".to_string());
                self.add_edge(loop_node, exit);
                exit
            }
            syn::Expr::While(expr_while) => {
                self.loop_count += 1;
                let cond_label =
                    truncate_label(format!("while {}", expr_snippet(&expr_while.cond)));
                let cond_node = self.add_node(cond_label);
                self.add_edge(current, cond_node);
                let body_exit = self.build_block(&expr_while.body.stmts, cond_node);
                self.add_edge(body_exit, cond_node);
                let exit = self.add_node("after while".to_string());
                self.add_edge(cond_node, exit);
                exit
            }
            syn::Expr::Loop(expr_loop) => {
                self.loop_count += 1;
                let loop_node = self.add_node("loop".to_string());
                self.add_edge(current, loop_node);
                let body_exit = self.build_block(&expr_loop.body.stmts, loop_node);
                self.add_edge(body_exit, loop_node);
                let exit = self.add_node("loop break".to_string());
                self.add_edge(loop_node, exit);
                exit
            }
            syn::Expr::Match(expr_match) => {
                self.branch_count += expr_match.arms.len();
                let match_label =
                    truncate_label(format!("match {}", expr_snippet(&expr_match.expr)));
                let match_node = self.add_node(match_label);
                self.add_edge(current, match_node);

                let mut exits = Vec::new();
                for arm in &expr_match.arms {
                    let mut arm_label = format!("arm {}", pat_snippet(&arm.pat));
                    if arm.guard.is_some() {
                        arm_label.push_str(" if guard");
                    }
                    let arm_node = self.add_node(truncate_label(arm_label));
                    self.add_edge(match_node, arm_node);
                    let arm_exit = self.process_expr(&arm.body, arm_node);
                    exits.push(arm_exit);
                }

                let join = self.add_node("match join".to_string());
                for exit in exits {
                    self.add_edge(exit, join);
                }
                join
            }
            syn::Expr::Block(expr_block) => self.build_block(&expr_block.block.stmts, current),
            syn::Expr::Return(ret) => {
                let mut label = "return".to_string();
                if let Some(expr) = &ret.expr {
                    label.push(' ');
                    label.push_str(&expr_snippet(expr));
                }
                let node = self.add_node(label);
                self.add_edge(current, node);
                node
            }
            syn::Expr::Break(expr_break) => {
                let mut label = "break".to_string();
                if let Some(expr) = &expr_break.expr {
                    label.push(' ');
                    label.push_str(&expr_snippet(expr));
                }
                let node = self.add_node(label);
                self.add_edge(current, node);
                node
            }
            syn::Expr::Continue(expr_continue) => {
                let mut label = "continue".to_string();
                if let Some(label_token) = &expr_continue.label {
                    label.push_str(&format!(" '{}", label_token.ident));
                }
                let node = self.add_node(label);
                self.add_edge(current, node);
                node
            }
            _ => {
                let node = self.add_node(truncate_label(expr_snippet(expr)));
                self.add_edge(current, node);
                node
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
