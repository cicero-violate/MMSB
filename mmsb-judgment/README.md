# mmsb-judgment CLI

## Judgment Ritual

The `mmsb-judgment` CLI is a judgment ritual, not a convenience utility.
It is designed to be slow, interruptive, and content-bound.

### Required Behavior

- Shows the full intent content before authorization
- Prints the intent hash and requires the full hash to be typed
- Forces an irreversibility acknowledgment
- Emits the token to stdout only

### Forbidden Patterns

These are constitutional violations and must never be added:

- `--yes` / `--force`
- Environment-variable confirmation
- Non-interactive mode
- Batch issuance
- Default confirmations

### Irreversibility Warning

The CLI must display this warning (or equivalent severity):

THIS ACTION IS IRREVERSIBLE.
By authorizing this judgment, you accept full responsibility
for the resulting state transition.
This authorization cannot be reused.
This authorization cannot be automated.
