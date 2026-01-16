use syn::{Item, Ident as SynIdent};

// TODO: Advanced query predicates
// - VisibilityPredicate (pub vs private)
// - GenericPredicate (has generics, specific type params)
// - AttributePredicate (has #[derive], #[cfg], etc.)
// - SignaturePredicate (param count, return type matching)
// - BodyPredicate (AST pattern matching in function bodies)
// - DocPredicate (doc comment content matching)

/// Base trait for all query predicates
pub trait Predicate: std::fmt::Debug + PredicateClone {
    fn matches(&self, item: &Item) -> bool;
}

/// Helper trait for cloning Predicate trait objects
pub trait PredicateClone {
    fn clone_box(&self) -> Box<dyn Predicate>;
}

impl<T> PredicateClone for T
where
    T: 'static + Predicate + Clone,
{
    fn clone_box(&self) -> Box<dyn Predicate> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Predicate> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Match items by kind (e.g., Function, Struct, Enum)
#[derive(Debug, Clone)]
pub enum ItemKind {
    Function,
    Struct,
    Enum,
    Trait,
    Impl,
    Mod,
    Use,
    Const,
    Static,
    Type,
}

#[derive(Debug, Clone)]
pub struct KindPredicate {
    pub kind: ItemKind,
}

impl KindPredicate {
    pub fn new(kind: ItemKind) -> Self {
        Self { kind }
    }
}

impl Predicate for KindPredicate {
    fn matches(&self, item: &Item) -> bool {
        match (&self.kind, item) {
            (ItemKind::Function, Item::Fn(_)) => true,
            (ItemKind::Struct, Item::Struct(_)) => true,
            (ItemKind::Enum, Item::Enum(_)) => true,
            (ItemKind::Trait, Item::Trait(_)) => true,
            (ItemKind::Impl, Item::Impl(_)) => true,
            (ItemKind::Mod, Item::Mod(_)) => true,
            (ItemKind::Use, Item::Use(_)) => true,
            (ItemKind::Const, Item::Const(_)) => true,
            (ItemKind::Static, Item::Static(_)) => true,
            (ItemKind::Type, Item::Type(_)) => true,
            _ => false,
        }
    }
}

/// Match items by name
#[derive(Debug, Clone)]
pub struct NamePredicate {
    pub name: String,
}

impl NamePredicate {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl Predicate for NamePredicate {
    fn matches(&self, item: &Item) -> bool {
        let item_name: Option<&SynIdent> = match item {
            Item::Fn(f) => Some(&f.sig.ident),
            Item::Struct(s) => Some(&s.ident),
            Item::Enum(e) => Some(&e.ident),
            Item::Trait(t) => Some(&t.ident),
            Item::Mod(m) => Some(&m.ident),
            Item::Const(c) => Some(&c.ident),
            Item::Static(s) => Some(&s.ident),
            Item::Type(t) => Some(&t.ident),
            _ => None,
        };

        item_name.map(|ident: &SynIdent| ident.to_string() == self.name).unwrap_or(false)
    }
}

/// Custom predicate with arbitrary closure
#[derive(Clone)]
pub struct CustomPredicate {
    predicate: std::sync::Arc<dyn Fn(&Item) -> bool + Send + Sync>,
}

impl CustomPredicate {
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&Item) -> bool + Send + Sync + 'static,
    {
        Self {
            predicate: std::sync::Arc::new(f),
        }
    }
}

impl Predicate for CustomPredicate {
    fn matches(&self, item: &Item) -> bool {
        (self.predicate)(item)
    }
}

impl std::fmt::Debug for CustomPredicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomPredicate").finish()
    }
}
