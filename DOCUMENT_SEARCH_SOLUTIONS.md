# Document Search Solutions for Multi-PDF Query System

## Problem Statement

When users upload multiple PDFs and ask questions, the system retrieves chunks from all documents, leading to poor responses due to mixed context from different sources.

## Solutions Implemented

### 1. **List Available Documents API**

**Endpoint:** `GET /documents/`

Lists all uploaded documents so users can see what's available.

**Response:**

```json
{
  "documents": ["document1.pdf", "document2.pdf", "document3.pdf"],
  "count": 3
}
```

### 2. **Query Specific Document API**

**Endpoint:** `POST /query_document/`

Allows users to query a specific document by name.

**Request:**

```json
{
  "query": "What is the company policy on remote work?",
  "document_name": "employee_handbook.pdf",
  "k": 3
}
```

**Response:**

```json
{
  "answer": "According to the employee handbook...",
  "document_queried": "employee_handbook.pdf",
  "chunks_found": 3
}
```

### 3. **Smart Query API (Recommended)**

**Endpoint:** `POST /smart_query/`

Intelligently determines the best document(s) to search based on the query.

**Request:**

```json
{
  "query": "What are the safety protocols?",
  "k": 3,
  "strategy": "best_match"
}
```

**Strategies Available:**

#### a) `best_match` (Recommended)

- Finds the most relevant document based on average similarity scores
- Returns chunks only from the most relevant document
- Provides document relevance scores for transparency

#### b) `multi_doc`

- Returns relevant chunks from multiple documents
- Balances results across top 2-3 most relevant documents
- Good for queries that might span multiple documents

#### c) `single_source`

- Original approach - filters to most common source in top results
- Simpler but less sophisticated than `best_match`

**Response:**

```json
{
  "answer": "The safety protocols include...",
  "strategy_used": "best_match",
  "chunks_used": 3,
  "primary_document": "safety_manual.pdf",
  "document_relevance_scores": {
    "safety_manual.pdf": 0.15,
    "employee_handbook.pdf": 0.45,
    "training_guide.pdf": 0.67
  }
}
```

## Frontend Implementation Suggestions

### Option 1: Document Selector UI

```javascript
// First, get available documents
const documents = await fetch("/documents/").then((r) => r.json());

// Show dropdown for document selection
<select>
  <option value="">All Documents (Smart Search)</option>
  {documents.documents.map((doc) => (
    <option value={doc}>{doc}</option>
  ))}
</select>;

// Query based on selection
if (selectedDocument) {
  // Query specific document
  await fetch("/query_document/", {
    method: "POST",
    body: JSON.stringify({
      query: userQuery,
      document_name: selectedDocument,
    }),
  });
} else {
  // Use smart search
  await fetch("/smart_query/", {
    method: "POST",
    body: JSON.stringify({
      query: userQuery,
      strategy: "best_match",
    }),
  });
}
```

### Option 2: Automatic Smart Search (Simplest for Users)

```javascript
// Just use smart search - no user selection needed
const response = await fetch("/smart_query/", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: userQuery,
    strategy: "best_match",
    k: 3,
  }),
});

const result = await response.json();
// Show answer and which document(s) were used
console.log(`Answer from: ${result.primary_document}`);
```

## Advanced Features

### Document Metadata Enhancement

You can enhance the metadata to include more information:

```python
# In split_text_into_chunks function
metadata = {
    "source": pdf_filename,
    "upload_date": datetime.now().isoformat(),
    "file_size": len(text),
    "chunk_index": i,
    "document_type": "pdf"  # or "youtube", etc.
}
```

### Query Classification

Implement query classification to automatically determine the best strategy:

```python
def classify_query(query: str) -> str:
    """Classify query to determine best search strategy"""
    if any(word in query.lower() for word in ['compare', 'difference', 'versus']):
        return "multi_doc"
    elif any(word in query.lower() for word in ['specific', 'particular', 'exact']):
        return "best_match"
    else:
        return "best_match"  # default
```

## Migration from Current System

Your existing `/query_qdrant/` endpoint remains unchanged for backward compatibility. To migrate:

1. **Immediate**: Use `/smart_query/` with `best_match` strategy as drop-in replacement
2. **Enhanced**: Add document listing and selection UI
3. **Advanced**: Implement query classification and automatic strategy selection

## Performance Considerations

- **Document Listing**: Cached for better performance
- **Search Optimization**: `best_match` strategy reduces token usage by focusing on single document
- **Memory Usage**: More efficient than processing mixed context from multiple documents

## Testing the New APIs

```bash
# List documents
curl -X GET "http://localhost:8000/documents/"

# Query specific document
curl -X POST "http://localhost:8000/query_document/" \
     -H "Content-Type: application/json" \
     -d '{"query": "test query", "document_name": "test.pdf"}'

# Smart query
curl -X POST "http://localhost:8000/smart_query/" \
     -H "Content-Type: application/json" \
     -d '{"query": "test query", "strategy": "best_match"}'
```
