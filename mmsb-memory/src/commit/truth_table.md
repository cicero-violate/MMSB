| Module         | Changes graph | Writes disk | Writes memory |
| -------------- | ------------- | ----------- | ------------- |
| `dag_commit`   | ✅ yes        | ✅ log      | ❌ no         |
| `commit_delta` | ❌ no         | ✅ log      | ❌ no         |
| `page_commit`  | ❌ no         | ❌ no       | ✅ yes        |
