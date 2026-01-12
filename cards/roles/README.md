kind = "doc"
title = "Role Cards - Canonical Prompt Contract"

prompt = """# Role Cards - Canonical Prompt Contract

Role cards define WHAT an agent may do.

They constrain:
- Decision-making
- Output types
- Responsibility boundaries

Prompt Composition Order (STRICT)

When initializing an agent:

1. Identity Card
2. Interface Card
3. One or more Role Cards

Precedence:

Interface > Role > Identity

Enforcement Rule

If a behavior is not explicitly allowed by the role card:

It is forbidden.

Role cards are binding.
"""
