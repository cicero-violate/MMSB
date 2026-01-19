### **Rust intuition shortcut**

| Concept | Role      |
| ------- | --------- |
| struct  | state     |
| trait   | rule      |
| impl    | binding   |
| fn      | execution |

### **Declare variables**

Let

* S = **struct**
* T = **trait**
* I = **impl**
* f = **function (fn)**
* D = data
* B = behavior
* C = constraint
* X = a concrete type

---

### **Latent equations**

1. **Struct**
   S = D

2. **Trait**
   T = C(B)

3. **Impl**
   I(X, T) \Rightarrow X \in T

4. **Function**
   f : D \rightarrow D

---

### **Composition view**

[
X = S + I(T)
]

[
\text{Executable behavior} = f \in I
]

---

### **Explanation (plain English)**

**Struct ((S))**
A struct is **data**.
It defines *what exists* in memory.

* Fields
* Layout
* No behavior by itself

Think: **noun**

---

**Trait ((T))**
A trait is a **constraint on behavior**.
It says:

> “Any type that claims this trait must provide these functions.”

Traits do **not** store data.
They define **capabilities**.

Think: **interface / contract / law**

---

**Impl ((I))**
An impl **binds behavior to data**.

It answers:

> “How does *this specific struct* satisfy *this trait*?”

This is where traits become real.

Think: **wiring**

---

**Function ((f))**
A function is a **transformation**.
It takes inputs and produces outputs.

* May be free-standing
* Or attached to a struct via `impl`

Think: **action**

---

### **Mental model (very important)**

* **Struct** → *What is it made of?*
* **Trait** → *What must it be able to do?*
* **Impl** → *How does it do that?*
* **Fn** → *The actual step-by-step logic*

---

### **Rust intuition shortcut**

| Concept | Role      |
| ------- | --------- |
| struct  | state     |
| trait   | rule      |
| impl    | binding   |
| fn      | execution |

---

### **Optimization principle**

[
\max(\text{intelligence},\ \text{efficiency},\ \text{correctness},\ \text{alignment}) = \text{good}
]

