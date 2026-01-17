# mmsb-policy

## Purpose
Classify intents under explicit policy rules and emit non-authoritative policy assessments.

`mmsb-policy` determines *how an intent is categorized*, not whether it is allowed.

---

## Responsibilities

- Consume canonical intents emitted by mmsb-intent
- Evaluate intents against declared policy rules
- Classify intent scope, category, and risk
- Attach applicable constraints and conditions
- Produce PolicyProof (B) as a witness of classification
- Emit `PolicyEvaluated` events with attached proofs

---

## Owns

- Policy rule definitions
- Policy schemas and versions
- Classification logic
- PolicyProof (B) creation

---

## Does NOT Do

- No approval or denial
- No judgment or authority
- No execution logic
- No state mutation
- No persistence
- No learning or feedback
- No proof verification

---

## Inputs

- Event: `IntentCreated`
  - Intent
  - IntentHash
  - IntentProof (A)

---

## Outputs

- Event: `PolicyEvaluated`
  - Intent
  - PolicyResult
  - PolicyProof (B)

All outputs are emitted to the event bus only.

---

## Guarantees

- Every classified intent has a corresponding PolicyProof
- Policy classification is deterministic
- Policy results are reproducible from inputs
- Policy rules are explicit and inspectable

---

## Proof Relationship

- PolicyProof (B) MUST include a hash of IntentProof (A)
- PolicyProof (B) MUST NOT grant permission or authority
- Missing or invalid IntentProof invalidates PolicyProof

---

## Authority

- NONE

`mmsb-policy` explains *what category an intent falls into*, not whether it may execute.
