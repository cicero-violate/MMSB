# Structure Group: src/01_types

## File: MMSB/src/01_types/delta_types.rs

- Layer(s): 01_types
- Language coverage: Rust (3)
- Element types: Enum (1), Struct (2)
- Total elements: 3

### Elements

- [Rust | Enum] `DeltaError` (line 0, pub)
- [Rust | Struct] `DeltaID` (line 0, pub)
  - Signature: `# [repr (transparent)] # [derive (Debug , Clone , Copy , PartialEq , Eq , Hash)] pub struct DeltaID (pub u64) ;`
- [Rust | Struct] `Source` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct Source (pub String) ;`

## File: MMSB/src/01_types/epoch_types.rs

- Layer(s): 01_types
- Language coverage: Rust (4)
- Element types: Impl (2), Struct (2)
- Total elements: 4

### Elements

- [Rust | Struct] `Epoch` (line 0, pub)
  - Signature: `# [doc = " Page epoch (monotonic counter)"] # [repr (transparent)] # [derive (Debug , Clone , Copy , PartialEq , Eq ,...`
- [Rust | Struct] `EpochCell` (line 0, pub)
  - Signature: `# [doc = " Thread-safe epoch cell reused inside the Rust Page structure."] # [derive (Debug)] pub struct EpochCell { ...`
- [Rust | Impl] `impl Epoch { pub fn new (value : u32) -> Self { Epoch (value) } } . self_ty` (line 0, priv)
- [Rust | Impl] `impl EpochCell { pub fn new (value : u32) -> Self { Self { inner : AtomicU32 :: new (value) , } } # [inline] pub fn load (& self) -> Epoch { Epoch (self . inner . load (Ordering :: Acquire)) } # [inline] pub fn store (& self , value : Epoch) { self . inner . store (value . 0 , Ordering :: Release) ; } # [inline] pub fn increment (& self) -> Epoch { let old = self . inner . fetch_add (1 , Ordering :: AcqRel) ; println ! ("EPOCH_INCREMENT: was {} → now {}" , old , old + 1) ; Epoch (old) } } . self_ty` (line 0, priv)

## File: MMSB/src/01_types/gc.rs

- Layer(s): 01_types
- Language coverage: Rust (2)
- Element types: Struct (1), Trait (1)
- Total elements: 2

### Elements

- [Rust | Struct] `GCMetrics` (line 0, pub)
  - Signature: `# [doc = " Metrics returned by memory pressure handlers after a GC pass."] # [derive (Debug , Clone , Copy)] pub stru...`
- [Rust | Trait] `MemoryPressureHandler` (line 0, pub)

## File: MMSB/src/01_types/mod.rs

- Layer(s): 01_types
- Language coverage: Rust (4)
- Element types: Module (4)
- Total elements: 4

### Elements

- [Rust | Module] `delta_types` (line 0, priv)
- [Rust | Module] `epoch_types` (line 0, priv)
- [Rust | Module] `gc` (line 0, priv)
- [Rust | Module] `page_types` (line 0, priv)

## File: MMSB/src/01_types/page_types.rs

- Layer(s): 01_types
- Language coverage: Rust (5)
- Element types: Enum (2), Impl (2), Struct (1)
- Total elements: 5

### Elements

- [Rust | Impl] `Display for impl fmt :: Display for PageID { fn fmt (& self , f : & mut fmt :: Formatter < '_ >) -> fmt :: Result { write ! (f , "{}" , self . 0) } } . self_ty` (line 0, priv)
- [Rust | Enum] `PageError` (line 0, pub)
- [Rust | Struct] `PageID` (line 0, pub)
  - Signature: `# [doc = " Globally unique identifier for pages"] # [repr (transparent)] # [derive (Debug , Clone , Copy , PartialEq ...`
- [Rust | Enum] `PageLocation` (line 0, pub)
- [Rust | Impl] `impl PageLocation { pub fn from_tag (tag : i32) -> Result < Self , PageError > { match tag { 0 => Ok (PageLocation :: Cpu) , 1 => Ok (PageLocation :: Gpu) , 2 => Ok (PageLocation :: Unified) , other => Err (PageError :: InvalidLocation (other)) , } } } . self_ty` (line 0, priv)

