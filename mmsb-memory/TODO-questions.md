I'll review the canonical documents and TODO.md to understand the implementation requirements.

Now let me check the current state of the mmsb-memory module to understand what exists:

Based on my review of the canonical documents and TODO.md, I have several questions and concerns that need escalation:

## **Variables**

Let $M$ denote `mmsb-memory` module  
Let $E$ denote `mmsb-executor` module  
Let $S_i$ denote substrate components $\{physical, device, propagation, optimization\}$  
Let $P_j$ denote proof stages $\{D, E, F\}$ (Admission, Commit, Outcome)

## **Latent Equations**

$$M_{new} = M_{semantic} \cap \neg S_i$$
$$E = S_i \cup runtime_{substrate}$$
$$\forall p \in P_j: producer(p) = M_{semantic}$$

---

## **Critical Questions for Escalation**

### **1. Numbered Directory Structure**
The current directory structure uses numbered prefixes (`__00_entry`, `__10_truth`, etc.). **Should these be preserved or removed during refactor?** The TODO doesn't address this organizational scheme.

### **2. Existing `mmsb-executor` Crate**
Workspace shows `mmsb-executor` already exists. **What is its current state?** Do we:
- Merge substrate into existing executor?
- Replace it entirely?
- Need to review its current contents first?

### **3. Dual Directory Naming**
Both numbered (`__40_propagation`) and non-numbered (`propagation/`) directories exist. **Which is canonical?** This creates ambiguity for what gets moved.

### **4. MemoryView Interface Boundaries**
TODO-9 mentions `MemoryView` for learning module. **What exact subset of memory state should be exposed?** Need specification of:
- Read-only query methods
- Proof reference access patterns
- Replay snapshot format

### **5. Event Emission from Memory**
TODO-7 says replay "does not emit events itself", but trait schematic shows `EmitEvent<MemoryCommitted>`. **Does memory emit commit events or not?** This is contradictory.

### **6. Proof Production vs Truth Semantics**
Memory produces proofs D/E/F but TODO says "truth semantics only". **Are proof production and truth semantics the same thing**, or is proof production considered runtime behavior?

### **7. Storage Boundary**
TODO-8 forbids filesystem APIs, but memory must persist. **What is the exact interface contract with `mmsb-storage`?** Need clarity on:
- Who owns persistence calls?
- Sync vs async boundaries
- Replay log format

### **8. Existing `proofs/` Directory**
Current structure has `proofs/` folder. **Does this contain D/E/F definitions or implementations?** Need to verify alignment with `mmsb-proof` crate expectations.

### **9. Module vs Engine Rename Scope**
TODO-2 renames `MemoryModule → MemoryEngine`. **Does this affect:**
- All trait implementations?
- External references in other crates?
- Should we audit dependencies first?

### **10. Materialization Location**
`materialization/` folder exists. **Is this truth semantics (stays) or execution substrate (moves)?** The distinction between "what must happen" vs "how it happens" is unclear here.

---

## **Recommended Approach Before Starting**

Before implementing TODO.md, I recommend creating a **dependency audit** and **current state analysis**:

1. Map all current imports in `mmsb-memory/Cargo.toml`
2. Check current state of `mmsb-executor`
3. Classify each existing directory as KEEP/MOVE/UNCLEAR
4. Resolve contradictions between TODO and trait schematic

Should I proceed with this audit first, or do you want to clarify the escalated questions above?
