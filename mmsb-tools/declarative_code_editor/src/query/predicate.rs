use syn::{Item, Ident as SynIdent};
use syn::Visibility;

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

/// Visibility level for matching
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisibilityLevel {
    /// pub
    Public,
    /// pub(crate)
    Crate,
    /// pub(super)
    Super,
    /// pub(self) or no visibility modifier (private)
    Private,
}

/// Match items by visibility
#[derive(Debug, Clone)]
pub struct VisibilityPredicate {
    pub level: VisibilityLevel,
}

impl VisibilityPredicate {
    pub fn new(level: VisibilityLevel) -> Self {
        Self { level }
    }
    
    pub fn public() -> Self {
        Self::new(VisibilityLevel::Public)
    }
    
    pub fn private() -> Self {
        Self::new(VisibilityLevel::Private)
    }
    
    pub fn crate_visible() -> Self {
        Self::new(VisibilityLevel::Crate)
    }
}

impl Predicate for VisibilityPredicate {
    fn matches(&self, item: &Item) -> bool {
        let vis = match item {
            Item::Fn(f) => Some(&f.vis),
            Item::Struct(s) => Some(&s.vis),
            Item::Enum(e) => Some(&e.vis),
            Item::Trait(t) => Some(&t.vis),
            Item::Mod(m) => Some(&m.vis),
            Item::Const(c) => Some(&c.vis),
            Item::Static(s) => Some(&s.vis),
            Item::Type(t) => Some(&t.vis),
            Item::Use(u) => Some(&u.vis),
            _ => None,
        };
        
        vis.map(|v| visibility_matches(v, &self.level)).unwrap_or(false)
    }
}

fn visibility_matches(vis: &Visibility, level: &VisibilityLevel) -> bool {
    match (vis, level) {
        (Visibility::Public(_), VisibilityLevel::Public) => true,
        (Visibility::Restricted(r), VisibilityLevel::Crate) => {
            r.path.is_ident("crate")
        }
        (Visibility::Restricted(r), VisibilityLevel::Super) => {
            r.path.is_ident("super")
        }
        (Visibility::Inherited, VisibilityLevel::Private) => true,
        (Visibility::Restricted(r), VisibilityLevel::Private) if r.path.is_ident("self") => true,
        _ => false,
    }
}

/// Match items by attribute presence
#[derive(Debug, Clone)]
pub struct AttributePredicate {
    pub attribute_name: String,
}

impl AttributePredicate {
    pub fn new(attribute_name: impl Into<String>) -> Self {
        Self {
            attribute_name: attribute_name.into(),
        }
    }
    
    pub fn derive() -> Self {
        Self::new("derive")
    }
    
    pub fn cfg() -> Self {
        Self::new("cfg")
    }
    
    pub fn test() -> Self {
        Self::new("test")
    }
}

impl Predicate for AttributePredicate {
    fn matches(&self, item: &Item) -> bool {
        let attrs = match item {
            Item::Fn(f) => &f.attrs,
            Item::Struct(s) => &s.attrs,
            Item::Enum(e) => &e.attrs,
            Item::Trait(t) => &t.attrs,
            Item::Impl(i) => &i.attrs,
            Item::Mod(m) => &m.attrs,
            Item::Const(c) => &c.attrs,
            Item::Static(s) => &s.attrs,
            Item::Type(t) => &t.attrs,
            Item::Use(u) => &u.attrs,
            _ => return false,
        };
        
        attrs.iter().any(|attr| {
            attr.path()
                .get_ident()
                .map(|id| id == &self.attribute_name)
                .unwrap_or(false)
        })
    }
}
