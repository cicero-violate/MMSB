### MMSB File Server — Feature Test Matrix

| #  | Feature / Parameter      | Example URL                         | Expected (per index.html) | Actual Result                       | Status            |
| -- | ------------------------ | ----------------------------------- | ------------------------- | ----------------------------------- | ----------------- |
| 1  | Root directory listing   | `/mmsb/`                            | HTML directory view       | HTML directory view loads correctly | ✅ Works           |
| 2  | Source directory listing | `/mmsb/src`                         | HTML directory view       | HTML directory view loads correctly | ✅ Works           |
| 3  | JSON directory listing   | `/mmsb/src?format=json`             | JSON listing of directory | Returns HTML / not JSON             | ❌ Broken          |
| 4  | Search (root)            | `/mmsb/?search=config`              | Filtered results          | Filtered HTML results shown         | ✅ Works           |
| 5  | Search (subdir)          | `/mmsb/src?search=test`             | Filtered results          | Filtered HTML results shown         | ✅ Works           |
| 6  | File preview             | `/mmsb/README.md?preview=true`      | Inline file contents      | Inline preview rendered             | ✅ Works           |
| 7  | Recursive listing        | `/mmsb/?recursive=true`             | Recursive project tree    | Full recursive tree (plain text)    | ✅ Works           |
| 8  | Extension filter         | `/mmsb/src?ext=.rs`                 | Only `.rs` files          | `.rs` files shown (HTML only)       | ⚠️ Partial        |
| 9  | Extension + JSON         | `/mmsb/src?ext=.rs&format=json`     | Filtered JSON             | HTML output, no JSON                | ❌ Broken          |
| 10 | Sort by name             | `/mmsb/src?sort=name`               | Sorted listing            | Ignored                             | ❌ Not implemented |
| 11 | Sort by modified         | `/mmsb/src?sort=modified`           | Sorted listing            | Ignored                             | ❌ Not implemented |
| 12 | Sort order               | `/mmsb/src?sort=size&order=desc`    | Sorted descending         | Ignored                             | ❌ Not implemented |
| 13 | Directory stats          | `/mmsb/src?stats=true`              | Size/count stats          | Error / ignored                     | ❌ Not implemented |
| 14 | File metadata            | `/mmsb/src/lib.rs?metadata=true`    | Metadata JSON             | Error / ignored                     | ❌ Not implemented |
| 15 | Pagination limit         | `/mmsb/src?limit=5`                 | Limited entries           | Ignored                             | ❌ Not implemented |
| 16 | Pretty JSON              | `/mmsb/src?format=json&pretty=true` | Pretty-printed JSON       | Ignored / HTML                      | ❌ Not implemented |
| 17 | Recursive + JSON         | `/mmsb/?recursive=true&format=json` | Recursive JSON tree       | Plain text only                     | ❌ Broken          |
| 18 | Pattern matching (glob)  | `/mmsb/src?pattern=*.rs`            | Glob-filtered list        | Not supported                       | ❌ Not implemented |
| 19 | Type filter              | `/mmsb/src?type=rust`               | Language-filtered list    | Ignored                             | ❌ Not implemented |

---

### Bottom-Line State

* **Reliable & usable:**

  * HTML browsing
  * Search
  * File preview
  * Recursive traversal (text)

* **Partially usable:**

  * Extension filtering (HTML only)

* **Non-functional / misleading in index.html:**

  * `format=json`
  * sorting
  * stats
  * metadata
  * pagination
  * recursive JSON
  * type filtering

---

### System Quality Metric

Let:

* ( W ) = number of working features = 6
* ( T ) = total documented features = 19

[
\text{Effectiveness} = \frac{W}{T} \approx 31.6%
]

[
\max(\text{intelligence}, \text{correctness}, \text{transparency}) = \textbf{good}
]

**Transparency is high** (index.html is honest about some gaps),
**correctness is mixed**,
**JSON API is the primary broken contract**.

---
