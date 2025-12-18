# MMSB File Server - Enhanced Query API

A Node.js file server with advanced querying capabilities for the MMSB project.

## Quick Start

```bash
cd tools/website
node file-server.js
```

Server runs on `http://127.0.0.1:8888/mmsb`

## Query Parameters

### Filtering

**Filter by extension:**
```
GET /mmsb/src?ext=.rs
GET /mmsb/src?ext=rs,toml
```

**Filter by type:**
```
GET /mmsb/src?type=rust
GET /mmsb/docs?type=markdown,text
```

Supported types: `rust`, `javascript`, `typescript`, `python`, `julia`, `markdown`, `config`, `shell`, `text`, `image`, `document`

**Search by filename:**
```
GET /mmsb/src?search=test
GET /mmsb/?search=config
```

**Pattern matching (glob):**
```
GET /mmsb/src?pattern=*.rs
GET /mmsb/test?pattern=test_*
```

### Sorting

**Sort by field:**
```
GET /mmsb/src?sort=name         # Default
GET /mmsb/src?sort=size
GET /mmsb/src?sort=modified
GET /mmsb/src?sort=type
```

**Sort order:**
```
GET /mmsb/src?sort=size&order=desc
GET /mmsb/src?sort=modified&order=asc
```

### Output Format

**Response format:**
```
GET /mmsb/src?format=json       # JSON output
GET /mmsb/src?format=html       # HTML directory listing (default)
GET /mmsb/src?format=text       # Plain text
```

**Pretty print JSON:**
```
GET /mmsb/src?format=json&pretty=true
```

### Pagination

**Limit and offset:**
```
GET /mmsb/src?limit=50
GET /mmsb/src?limit=50&offset=100
```

**Page-based:**
```
GET /mmsb/src?page=2&pagesize=25
```

### Recursive Listing

**Enable recursion:**
```
GET /mmsb/src?recursive=true
GET /mmsb/src?recursive=true&depth=2
```

### Metadata & Statistics

**File metadata only:**
```
GET /mmsb/src/lib.rs?metadata=true
```

Response:
```json
{
  "name": "lib.rs",
  "path": "/src/lib.rs",
  "size": 15420,
  "size_formatted": "15.06 KB",
  "modified": "2025-12-18T10:30:00.000Z",
  "extension": ".rs",
  "is_file": true
}
```

**Directory statistics:**
```
GET /mmsb/src?stats=true&format=json
```

Response:
```json
{
  "path": "/src",
  "total_files": 45,
  "total_dirs": 8,
  "total_size": 256000,
  "total_size_formatted": "250 KB",
  "file_types": {
    ".rs": 40,
    ".toml": 3,
    ".md": 2
  }
}
```

### Content Preview

**Preview first N lines:**
```
GET /mmsb/README.md?preview=true
GET /mmsb/README.md?preview=true&lines=50
```

## Combined Queries

Combine multiple parameters for powerful queries:

```
# Recent Rust files, largest first
GET /mmsb/src?ext=.rs&sort=modified&order=desc&limit=10&format=json

# All test files recursively
GET /mmsb/?pattern=test_*&recursive=true&depth=3&format=json

# Markdown files with "architecture" in name
GET /mmsb/docs?type=markdown&search=architecture&sort=modified

# Statistics on config files
GET /mmsb/?type=config&stats=true&format=json
```

## Response Examples

### JSON Directory Listing

```json
{
  "path": "/src",
  "entries": [
    {
      "name": "lib.rs",
      "type": "file",
      "path": "/src/lib.rs",
      "url": "/mmsb/src/lib.rs",
      "size": 15420,
      "size_formatted": "15.06 KB",
      "extension": ".rs",
      "modified": "2025-12-18T10:30:00.000Z"
    },
    {
      "name": "00_physical",
      "type": "directory",
      "path": "/src/00_physical",
      "url": "/mmsb/src/00_physical",
      "modified": "2025-12-18T09:15:00.000Z"
    }
  ],
  "pagination": {
    "total": 45,
    "offset": 0,
    "limit": 50,
    "returned": 45,
    "hasMore": false
  }
}
```

### HTML Directory Listing

Visual directory browser with:
- Dark theme optimized for code browsing
- Active filter display
- Pagination information
- Sortable columns
- Clickable file/directory links

### Text Directory Listing

```
Directory: /src

TYPE        SIZE           MODIFIED              NAME
--------------------------------------------------------------------------------
[DIR]       -              2025-12-18 09:15:00   00_physical/
[DIR]       -              2025-12-18 09:20:00   01_page/
[FILE]      15.06 KB       2025-12-18 10:30:00   lib.rs
[FILE]      2.45 KB        2025-12-18 08:45:00   ffi.rs
```

## Error Handling

All errors return appropriate HTTP status codes with formatted error messages:

```json
{
  "error": true,
  "statusCode": 404,
  "message": "File or directory not found"
}
```

Common status codes:
- `400` - Bad Request (invalid query parameters)
- `403` - Forbidden (path traversal attempt)
- `404` - Not Found
- `500` - Internal Server Error

## Configuration

Edit `config/server-config.json` to customize:

- Server port and host
- URL prefix
- Feature toggles (recursive, pagination, preview)
- Security settings
- Rate limiting
- Caching policy

## Use with Grok or AI Agents

The JSON API is optimized for programmatic access:

```javascript
// Get all Rust files modified today
const url = 'http://127.0.0.1:8889/mmsb/src?ext=.rs&sort=modified&order=desc&format=json';
const response = await fetch(url);
const data = await response.json();

for (const file of data.entries) {
  console.log(`${file.name}: ${file.size_formatted}`);
}
```

## Architecture

```
Request → Query Parser → Filter → Sort → Paginate → Format → Response
```

Middleware components:
- `middleware/query-parser.js` - Parse and validate URL parameters
- `middleware/filter.js` - Filter files by extension/type/pattern
- `middleware/sort.js` - Sort and paginate results
- `middleware/recursive.js` - Recursive directory traversal
- `routes/api.js` - Format responses (JSON/HTML/text)

## Examples for Common Tasks

**Find all configuration files:**
```
/mmsb/?type=config&recursive=true&format=json
```

**Get project structure:**
```
/mmsb/?recursive=true&depth=3&format=text
```

**Find large files:**
```
/mmsb/?sort=size&order=desc&limit=20&format=json
```

**Search documentation:**
```
/mmsb/docs?type=markdown&search=architecture&format=json
```

**Get file metadata:**
```
/mmsb/Cargo.toml?metadata=true&format=json
```
