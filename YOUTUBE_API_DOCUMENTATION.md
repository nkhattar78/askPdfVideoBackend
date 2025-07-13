# 🎥 YouTube Video Query API Documentation

## 🎯 New YouTube Video APIs

Your FastAPI application now supports comprehensive YouTube video transcript processing and querying! Here are all the new endpoints:

---

## 📤 **Upload YouTube Video**

### `POST /upload-youtube/`

Process and store YouTube video transcript for later querying.

**Request:**

```json
{
  "video_url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

**Response:**

```json
{
  "message": "YouTube transcript processed and embedded successfully",
  "video_id": "VIDEO_ID",
  "source": "video_VIDEO_ID.txt",
  "num_chunks": 45,
  "video_url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

---

## 🎥 **Query Specific YouTube Video**

### `POST /query_youtube/`

Ask questions about a specific YouTube video (processes transcript on-the-fly).

**Request:**

```json
{
  "query": "What are the main points discussed in this video?",
  "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "k": 3
}
```

**Response:**

```json
{
  "answer": "The main points discussed include...",
  "video_id": "VIDEO_ID",
  "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "source": "database", // or "direct_transcript"
  "chunks_found": 3
}
```

---

## 📺 **List Available Videos**

### `GET /videos/`

Get all processed YouTube videos in the database.

**Response:**

```json
{
  "videos": [
    {
      "video_id": "ABC123",
      "source_name": "video_ABC123.txt",
      "youtube_url": "https://www.youtube.com/watch?v=ABC123"
    }
  ],
  "count": 1
}
```

---

## 🧠 **Smart Query Across All Content**

### `POST /smart_query_all/`

Intelligent search across PDFs, YouTube videos, or both with automatic content type detection.

**Request:**

```json
{
  "query": "Explain machine learning concepts",
  "k": 3,
  "strategy": "best_match",
  "content_type": "all" // "all", "pdf", "video"
}
```

**Response:**

```json
{
  "answer": "Machine learning concepts include...",
  "strategy_used": "best_match",
  "chunks_used": 3,
  "content_type": "all",
  "primary_source": "video_ABC123.txt",
  "source_type": "video",
  "sources_used": ["video_ABC123.txt"],
  "source_relevance_scores": {
    "video_ABC123.txt": 0.15,
    "ml_guide.pdf": 0.45
  }
}
```

---

## 📊 **Content Summary**

### `GET /content_summary/`

Get overview of all available content.

**Response:**

```json
{
  "total_documents": 5,
  "videos": {
    "count": 2,
    "sources": ["video_ABC123.txt", "video_XYZ789.txt"]
  },
  "pdfs": {
    "count": 3,
    "sources": ["manual.pdf", "guide.pdf", "handbook.pdf"]
  }
}
```

---

## 🚀 **Usage Examples**

### 1. **Upload and Query a Video**

```bash
# Upload video
curl -X POST "http://localhost:8000/upload-youtube/" \
     -H "Content-Type: application/json" \
     -d '{"video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'

# Query the video
curl -X POST "http://localhost:8000/query_youtube/" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is this video about?",
       "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
     }'
```

### 2. **Smart Search Across All Content**

```bash
curl -X POST "http://localhost:8000/smart_query_all/" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Explain the concept of neural networks",
       "content_type": "all",
       "strategy": "best_match"
     }'
```

### 3. **Search Only Videos**

```bash
curl -X POST "http://localhost:8000/smart_query_all/" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What was mentioned about AI?",
       "content_type": "video"
     }'
```

---

## 🎯 **Content Type Strategies**

### **`content_type: "all"`** (Default)

- Searches across both PDFs and YouTube videos
- Best for general queries
- Returns most relevant content regardless of type

### **`content_type: "pdf"`**

- Searches only PDF documents
- Best for formal documentation queries
- Excludes all video content

### **`content_type: "video"`**

- Searches only YouTube video transcripts
- Best for lecture/tutorial content
- Excludes all PDF content

---

## 🔄 **Search Strategies**

All video APIs support the same intelligent search strategies:

- **`best_match`**: Finds single most relevant source
- **`multi_doc`**: Returns from multiple relevant sources
- **`single_source`**: Original filtering approach

---

## 🎨 **Frontend Integration**

### React/JavaScript Example:

```javascript
// Upload video
const uploadVideo = async (videoUrl) => {
  const response = await fetch("/upload-youtube/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_url: videoUrl }),
  });
  return response.json();
};

// Query across all content
const smartQuery = async (query, contentType = "all") => {
  const response = await fetch("/smart_query_all/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      content_type: contentType,
      strategy: "best_match",
    }),
  });
  return response.json();
};

// Get content summary
const getContentSummary = async () => {
  const response = await fetch("/content_summary/");
  return response.json();
};
```

---

## 🛠 **Technical Details**

### **Video ID Extraction**

Supports multiple YouTube URL formats:

- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://youtube.com/watch?v=VIDEO_ID`

### **Transcript Processing**

- Automatic transcript retrieval using `youtube-transcript-api`
- Text chunking for optimal embedding storage
- Metadata includes video ID and source URL

### **Database Storage**

- Videos stored with prefix `video_` in filename
- Same embedding database as PDFs
- Unified search across all content types

### **Error Handling**

- Handles videos without transcripts
- Manages private/restricted videos
- Fallback to direct transcript processing

---

## 🧪 **Testing**

Test the new endpoints:

```bash
# Test all video endpoints
python -c "
import requests

# Upload video
r1 = requests.post('http://localhost:8000/upload-youtube/',
                   json={'video_url': 'YOUR_VIDEO_URL'})
print('Upload:', r1.json())

# Query video
r2 = requests.post('http://localhost:8000/query_youtube/',
                   json={'query': 'What is this about?', 'video_url': 'YOUR_VIDEO_URL'})
print('Query:', r2.json())

# List videos
r3 = requests.get('http://localhost:8000/videos/')
print('Videos:', r3.json())

# Smart query all
r4 = requests.post('http://localhost:8000/smart_query_all/',
                   json={'query': 'Test query', 'content_type': 'all'})
print('Smart Query:', r4.json())
"
```

---

## 🎉 **Complete API Endpoints Summary**

| Endpoint            | Method | Purpose                              |
| ------------------- | ------ | ------------------------------------ |
| `/upload-pdf/`      | POST   | Upload PDF documents                 |
| `/upload-youtube/`  | POST   | Upload YouTube videos                |
| `/documents/`       | GET    | List PDF documents                   |
| `/videos/`          | GET    | List YouTube videos                  |
| `/content_summary/` | GET    | Overview of all content              |
| `/query_qdrant/`    | POST   | Original query (backward compatible) |
| `/query_document/`  | POST   | Query specific PDF                   |
| `/query_youtube/`   | POST   | Query specific video                 |
| `/smart_query/`     | POST   | Smart PDF search                     |
| `/smart_query_all/` | POST   | Smart search across all content      |

Your application now supports a complete multimedia knowledge base! 🚀
