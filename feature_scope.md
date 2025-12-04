Understood.
You want **a clean, finished, unified “Feature Scope Specification”** — without rambling, without metaphors, without comparisons.
Just the *scope*, written like an engineer documenting a subsystem.

Here it is.

---
# **MMSB — FEATURE SCOPE (FINAL SPEC)**

**Memory-Mapped State Bus**

This is the exact scope of what the MMSB *is* and *is not*.
This is the list to complete.
This is the contract.

---

# **1. CORE CAPABILITIES**

## **1.1 Versioning**

* Every page has a monotonically increasing version (epoch).
* Every delta increments version.
* Enables ordering, conflict detection, and synchronization.

## **1.2 Delta Updates (Byte-Level)**

* State changes are represented as *deltas*, not full replacements.
* Deltas contain:

  * byte mask (which bytes changed)
  * new byte values
* No semantic interpretation required.

## **1.3 Logging (Internal TLog)**

* Append-only log of deltas.
* Used for:

  * state reconstruction
  * temporal replay
  * crash recovery
  * GPU ↔ CPU coherence

## **1.4 Dependency Graph**

* Graph mapping page → dependent pages.
* When a page updates, affected dependents receive notifications.
* Supports multi-device propagation (CPU, GPU, agents).

## **1.5 Event Notifications**

* Broadcast events when:

  * a page changes
  * a delta is applied
  * a dependency triggers
* Not a messaging system; events carry internal state-coherence meaning only.

## **1.6 Deterministic Replay**

* Reconstruct exact historical states from:

  * base page
  * sequence of deltas
* Enables deterministic debugging, reproducibility, and time travel.

---

# **2. WHAT MMSB IS (TRUE SCOPE)**

## **2.1 Structured RAM**

* Operates entirely in memory (DRAM or Unified Memory).
* Page-based state fabric with explicit update rules.

## **2.2 Raw Byte Storage**

* Pages store raw bytes, not:

  * objects
  * structs
  * documents
  * JSON
  * key-value pairs

This ensures device-agnostic, type-agnostic, zero-overhead storage.

## **2.3 State Consistency Layer**

* MMSB ensures all views (CPU, GPU, agents) see coherent state.
* Not for messaging.
* Not for business events.
* All operations revolve around memory coherence.

## **2.4 Application-Level State Engine**

* Handles:

  * page updates
  * delta propagation
  * device sync
  * dependency execution
  * state replay

## **2.5 Multi-Device Logic**

* Coordinates memory interactions across:

  * CPU
  * GPU
  * browser WASM
  * agents
  * subsystems

All using the same page/delta model.

## **2.6 RAM-Speed Diff Engine**

* Applies deltas at memory throughput.
* Delta-based, not replace-based.
* Versioning baked in.

---

# **3. WHAT MMSB IS *NOT***

## **3.1 Not a Database**

* No queries.
* No indexes.
* No schema.
* No query language.
* No predicates.
* No transactions.

## **3.2 Not Distributed (By Default)**

* Single-machine by default.
* Distribution requires explicit replication layers built on top.

## **3.3 Not a Cache**

* No eviction.
* No TTL.
* No key-value semantics.

## **3.4 Not an Event Stream**

* Does not store arbitrary event messages.
* Logs *deltas*, not domain events.

## **3.5 Not Semantic**

* Does not understand:

  * types
  * structs
  * JSON
  * graphs
  * functions

Everything is just bytes.

---

# **4. EXECUTION ROLE IN SYSTEM**

## **4.1 The Lowest-Level State Substrate**

* Sits *above* raw memory (DRAM / Unified Memory).
* Sits *below* databases, CRDTs, agents, UI frameworks, GPU layouts.

## **4.2 Unified Memory Fabric**

* Single coherent memory model across devices.
* Everything is represented as Pages + Deltas.

## **4.3 Deterministic Operating Layer**

* Provides:

  * version-controlled memory
  * reproducibility
  * propagation
  * scheduled updates
  * exact state auditing

---

# **5. ONE SENTENCE DEFINITION**

**MMSB is a structured, versioned, delta-driven shared-memory substrate providing deterministic state coherence across CPU, GPU, and program components, without semantics, schemas, or database features.**

---
