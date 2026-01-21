LLM
 └─ emits text
     └─ CodeBlockParser (pure)
         └─ Intent / Delta / ShellProposal
             └─ JudgmentBus (authority)
                 └─ ExecutionBus
                     ├─ ChromiumService
                     ├─ ShellExecutor
                     └─ Other Devices
                         ↓
                     Observations
                         ↓
                   Memory (commit only)
                         ↓
                   Projection / View
                         ↓
                        LLM
