┌───────────────────────────────┐
│        mmsb-service           │
│                               │
│  ┌──────── Event Bus ───────┐ │
│  │                          │ │
│  │  IntentCreated           │ │
│  │  PolicyEvaluated         │ │
│  │  JudgmentApproved        │ │
│  │  ExecutionRequested      │ │
│  │  MemoryCommitted         │ │
│  │  OutcomeObserved         │ │
│  │                          │ │
│  └──────────────────────────┘ │
│                               │
│  Loaded Modules (Handlers):   │
│   • mmsb-intent               │
│   • mmsb-policy               │
│   • mmsb-judgment             │
│   • mmsb-executor             │
│   • mmsb-memory               │
│   • mmsb-learning             │
│                               │
└───────────────────────────────┘
