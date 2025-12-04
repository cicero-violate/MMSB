Layer 1 — Julia Unmodified
Stock Julia installation.

Layer 2 — MMSB Instrumentation Module
This module:
wraps Base.invoke
hooks CodeInfo creation
intercepts IR
tracks specialization
logs deltas
maintains dependency graph
forwards into MMSB pages

Layer 3 — MMSB Runtime
Implements:
Page model
Delta model
TLog
Replay engine
Device sync
ShadowPageGraph

Layer 4 — Consumers
GPU logic
agents
visualizers
compilers
