kind = "doc"
title = "Roles Summary - High-Level View"

prompt = """# Roles Summary - High-Level View

This table provides a compressed operational overview.

Roles Overview

| Role               | Function          | Code Exec    | Design | Decisions | Patch Apply | Typical Interface |
|--------------------+-------------------+--------------+--------+-----------+-------------+-------------------|
| R1 Technical Lead  | Scope & decisions | no           | yes    | yes       | no          | ChatGPT           |
| R2 Systems Analyst | Inventory         | no           | no     | no        | no          | Claude            |
| R4 Implementer     | Implementation    | yes (bounded)| no     | no        | yes         | Codex             |
| R5 Patch Generator | Patch syntax      | no           | no     | no        | no          | Codex             |
| R6 Verifier        | Review            | no           | no     | no        | no          | ChatGPT           |
| R7 Steward         | Process           | no           | no     | no        | no          | ChatGPT           |

Invariant

If a capability is not explicitly granted:

It does not exist.
"""
