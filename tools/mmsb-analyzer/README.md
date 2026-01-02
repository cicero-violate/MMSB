MMSB Analyzer
=============

Purpose
-------
MMSB Analyzer is a structural intelligence tool for Rust and Julia codebases. It scans the project,
builds dependency graphs, detects invariants, and emits structured refactor intelligence that can be
consumed by the MMSB executor.

Highlights
----------
- Dependency-ordered scanning of Rust and Julia sources
- Call graph and cohesion analysis
- Structural and semantic invariant detection
- Refactor planning outputs (clustered, phased, and ordered)
- Correction intelligence JSON for the executor

Quick Start
-----------
Run analysis from the project root:

  ./run_analysis.sh

This writes reports under:

  docs/

Common outputs:
- docs/00_refactoring_plan/
- docs/10_structure/
- docs/97_correction_intelligence/

Run the executor
----------------
Use the executor for apply or dry-run modes. The script is configurable by commenting/uncommenting
blocks:

  ./run_executor.sh

Key Inputs/Outputs
------------------
Inputs:
- Project root (Rust/Julia sources)
- analyzer_config.toml (behavior tuning)

Outputs:
- Markdown reports under docs/
- JSON correction intelligence under docs/97_correction_intelligence/

How It Works (High Level)
-------------------------
1) Scan and order files by dependency
2) Build graphs (call, module, layer)
3) Detect invariants and violations
4) Produce reports and refactoring plans
5) Emit correction intelligence for execution

Notes
-----
- The executor is intentionally guarded and allowlisted; most changes are report-only by default.
- Diff reports are produced when the executor records file mutations.

License
-------
Internal project use.
