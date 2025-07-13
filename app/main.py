# Create Virtual Environment 
    #python -m venv venv
# Activate virtual Environment
    # .\venv\Scripts\activate
# install all the required lib either directly or add in requirements.txt file
# pip install -r requirements.txt
# Start the server
    #uvicorn app.main:app --reload

# Debugging FastAPI code
# From run and debug options from Run button towards top-right, choose option "Python Debugger: Debug suing launch.json"
# choose Python Debugger - Current file from seach drop down which comes in focus 
# Choose Python File
# Update launch.json file with below this content
# {
#   "version": "0.2.0",
#   "configurations": [
#     {
#       "name": "Debug FastAPI (Uvicorn)",
#       "type": "python",
#       "request": "launch",
#       "module": "uvicorn",
#       "args": [
#         "app.main:app",          // Change 'main' to your filename (no .py)
#         "--reload"
#       ],
#       "jinja": true,
#       "justMyCode": true
#     }
#   ]
# }


from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.docstore.document import Document
# Qdrant utilities moved to separate file
from app.qdrant_utils import (
    create_embeddings_and_store_qdrant, 
    qdrant_similarity_search, 
    get_available_documents,
    search_specific_document,
    smart_document_search,
    search_videos_only,
    search_pdfs_only,
    get_content_type_summary
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Query
from youtube_transcript_api import YouTubeTranscriptApi
import re
from collections import Counter 
from typing import List, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import os
genai.configure(api_key=os.getenv("GOOGLE_API_KEY2"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    k: int = 3  # number of top results to return

class DocumentQueryRequest(BaseModel):
    query: str
    document_name: str
    k: int = 3

class SmartQueryRequest(BaseModel):
    query: str
    k: int = 3
    strategy: str = "best_match"  # "best_match", "multi_doc", "single_source"

class VideoRequest(BaseModel):
    video_url: str

class VideoRequestWithTranscript(BaseModel):
    video_url: str
    manual_transcript: Optional[str] = None
    use_fallback: bool = True

class VideoQueryRequest(BaseModel):
    query: str
    video_url: str
    k: int = 3
    manual_transcript: Optional[str] = None

class SmartVideoQueryRequest(BaseModel):
    query: str
    k: int = 3
    strategy: str = "best_match"  # "best_match", "multi_doc", "single_source"
    content_type: str = "all"  # "all", "pdf", "video"

# ---- Helper Functions ---- #

def validate_pdf(file: UploadFile):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF files are allowed."
        )


async def extract_text_from_pdf(file: UploadFile) -> str:
    file_bytes = await file.read()
    try:
        pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in pdf_doc:
            text += page.get_text()
        pdf_doc.close()

        if not text.strip():
            raise ValueError("PDF has no extractable text.")
        return text
    except Exception as e:
        raise RuntimeError(f"PDF text extraction failed: {str(e)}")


def split_text_into_chunks(text: str, pdf_path: str, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    doc = Document(page_content=text, metadata={"source": pdf_path.split("/")[-1]})
    return splitter.split_documents([doc])



    



def ask_gemini(query: str, context_chunks: list) -> str:
    load_dotenv()

    # Configure the API
    key = os.getenv("GOOGLE_API_KEY2")
    print("key:", key)

    genai.configure(api_key=key)

    context = "\n\n".join([doc.page_content for doc in context_chunks])

    prompt = f"""You are an assistant that answers questions based on the context provided.

Context:
{context}

Question:
{query}

Answer:"""
    print(prompt)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

def get_youtube_transcript(video_url: str, use_fallback: bool = True) -> str:
    """
    Get YouTube transcript with multiple fallback strategies for cloud deployment
    """
    try:
        video_id = extract_video_id(video_url)
        
        # Primary method: Direct transcript API
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            full_text = " ".join([entry['text'] for entry in transcript])
            return full_text
        except Exception as primary_error:
            if not use_fallback:
                raise RuntimeError(f"Primary transcript retrieval failed: {str(primary_error)}")
            
            # Fallback method 1: Try with different language codes
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'en-GB'])
                full_text = " ".join([entry['text'] for entry in transcript])
                return full_text
            except Exception as fallback1_error:
                pass
            
            # Fallback method 2: Try to get any available transcript
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                for transcript in transcript_list:
                    try:
                        transcript_data = transcript.fetch()
                        full_text = " ".join([entry['text'] for entry in transcript_data])
                        return full_text
                    except:
                        continue
                raise Exception("No accessible transcripts found")
            except Exception as fallback2_error:
                pass
            
            # If all transcript methods fail, provide helpful error message
            error_message = f"""Transcript retrieval failed for video {video_id}. 

Primary Error: {str(primary_error)}

This is likely due to:
1. YouTube blocking cloud provider IPs (Azure/AWS/GCP)
2. Video has no captions/transcripts
3. Video is private or restricted
4. Geographic restrictions

For cloud deployment, consider:
1. Using a proxy service
2. Asking users to provide transcript text manually
3. Using a third-party YouTube API service
4. Processing videos locally before deployment"""

            raise RuntimeError(error_message)
            
    except Exception as e:
        if "extract_video_id" in str(e):
            raise RuntimeError(f"Invalid YouTube URL format: {video_url}")
        raise RuntimeError(str(e))

def get_youtube_transcript_with_proxy(video_url: str, proxy_url: str = None) -> str:
    """
    Get YouTube transcript using proxy (for cloud deployment)
    This is a placeholder for proxy implementation
    """
    try:
        video_id = extract_video_id(video_url)
        
        if proxy_url:
            # Implement proxy logic here if needed
            # This would require additional libraries like requests with proxy support
            pass
        
        # For now, try the standard method
        return get_youtube_transcript(video_url, use_fallback=True)
    except Exception as e:
        raise RuntimeError(f"Proxy transcript retrieval failed: {str(e)}")

def handle_manual_transcript(video_url: str, manual_transcript: str) -> str:
    """
    Handle manually provided transcript text
    """
    if not manual_transcript or not manual_transcript.strip():
        raise ValueError("Manual transcript text is required")
    
    return manual_transcript.strip()

def extract_video_id(url: str) -> str:
    from urllib.parse import urlparse, parse_qs
    query = urlparse(url).query
    params = parse_qs(query)
    if "v" in params:
        return params["v"][0]
    else:
        # Handle youtu.be/VIDEO_ID format
        return url.split("/")[-1]


# ---- API Endpoint ---- #

@app.get("/CallLLM")
def read_root():
    # Load environment variables from .env
    load_dotenv()

    # Configure the API
    key = os.getenv("GOOGLE_API_KEY2")
    print("key:", key)

    genai.configure(api_key=key)

    # for m in genai.list_models():
    #     print(m.name, "->", m.supported_generation_methods)

    # Create a generative model (use "gemini-pro" for text)
    # model = genai.GenerativeModel(model_name="models/gemini-pro")
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Define a prompt
    prompt = "Give a short note on Dynamic Programming"

    # Generate content
    response = model.generate_content(prompt)

    # Print the result
    print(response.text)

    return {"message": response.text}

@app.get("/")
def read_root():
    load_dotenv()   
    return {"message": os.getenv("SHREYA_NAME")}

@app.post("/upload-pdf/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    all_chunks = []
    upload_summaries = []

    try:
        for file in files:  
            validate_pdf(file)

            pdf_filename = file.filename
            text = await extract_text_from_pdf(file) 
            docs = split_text_into_chunks(text, pdf_path=pdf_filename)
            all_chunks.extend(docs)

            upload_summaries.append({
                "filename": pdf_filename,
                "num_chunks": len(docs)
            })

        num_chunks = create_embeddings_and_store_qdrant(docs)

        return JSONResponse(content={
            "message": f"Embeddings stored in Qdrant",
            "summary": upload_summaries
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


    
@app.post("/query_qdrant/")
async def query_vector_db_qdrant(request: QueryRequest):
    try:
        # Use Qdrant utility for similarity search
        results = qdrant_similarity_search(request.query, k=request.k)
        gemini_answer = ask_gemini(request.query, results)
        return {
            "answer": gemini_answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/")
async def list_available_documents():
    """Get list of all uploaded documents"""
    try:
        documents = get_available_documents()
        return {
            "documents": documents,
            "count": len(documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query_document/")
async def query_specific_document(request: DocumentQueryRequest):
    """Query a specific document by name"""
    try:
        results = search_specific_document(request.query, request.document_name, k=request.k)
        if not results:
            return {
                "answer": f"No relevant information found in document '{request.document_name}' for your query.",
                "document_queried": request.document_name
            }
        
        gemini_answer = ask_gemini(request.query, results)
        return {
            "answer": gemini_answer,
            "document_queried": request.document_name,
            "chunks_found": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/smart_query/")
async def smart_query(request: SmartQueryRequest):
    """Intelligent query that automatically determines the best search strategy"""
    try:
        search_result = smart_document_search(request.query, k=request.k, strategy=request.strategy)
        
        if not search_result["chunks"]:
            return {
                "answer": "No relevant information found for your query.",
                "strategy_used": request.strategy
            }
        
        gemini_answer = ask_gemini(request.query, search_result["chunks"])
        
        response = {
            "answer": gemini_answer,
            "strategy_used": search_result["strategy_used"],
            "chunks_used": len(search_result["chunks"])
        }
        
        # Add strategy-specific metadata
        if "primary_source" in search_result:
            response["primary_document"] = search_result["primary_source"]
        if "sources_used" in search_result:
            response["documents_used"] = search_result["sources_used"]
        if "document_scores" in search_result:
            response["document_relevance_scores"] = search_result["document_scores"]
            
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---- YouTube Video APIs ---- #

@app.post("/upload-youtube/")
async def upload_youtube(req: VideoRequestWithTranscript):
    """Upload and process YouTube video transcript with fallback support"""
    try:
        video_id = extract_video_id(req.video_url)
        
        # Try to get transcript
        if req.manual_transcript:
            # Use manually provided transcript
            transcript = handle_manual_transcript(req.video_url, req.manual_transcript)
            source_method = "manual"
        else:
            # Try automatic transcript retrieval
            try:
                transcript = get_youtube_transcript(req.video_url, use_fallback=req.use_fallback)
                source_method = "automatic"
            except Exception as transcript_error:
                # If automatic fails, provide helpful response
                return {
                    "success": False,
                    "error": "transcript_blocked",
                    "message": "YouTube transcript blocked by cloud provider",
                    "details": str(transcript_error),
                    "video_id": video_id,
                    "video_url": req.video_url,
                    "solutions": [
                        "Provide manual transcript text in 'manual_transcript' field",
                        "Use a proxy service",
                        "Process video locally before cloud deployment",
                        "Use alternative transcript services"
                    ],
                    "manual_upload_example": {
                        "video_url": req.video_url,
                        "manual_transcript": "Your transcript text here..."
                    }
                }
        
        source_name = f"video_{video_id}.txt"
        chunks = split_text_into_chunks(transcript, pdf_path=source_name)

        create_embeddings_and_store_qdrant(chunks)

        return {
            "success": True,
            "message": "YouTube transcript processed and embedded successfully",
            "video_id": video_id,
            "source": source_name,
            "num_chunks": len(chunks),
            "video_url": req.video_url,
            "transcript_method": source_method,
            "transcript_length": len(transcript)
        }

    except Exception as e:
        return {
            "success": False,
            "error": "processing_failed",
            "message": "Failed to process YouTube video",
            "details": str(e),
            "video_url": req.video_url if hasattr(req, 'video_url') else "unknown"
        }

@app.post("/query_youtube/")
async def query_youtube_video(request: VideoQueryRequest):
    """Query a specific YouTube video transcript with fallback support"""
    try:
        video_id = extract_video_id(request.video_url)
        source_name = f"video_{video_id}.txt"
        
        # First, try to search in existing database
        try:
            results = search_specific_document(request.query, source_name, k=request.k)
            if results:
                gemini_answer = ask_gemini(request.query, results)
                return {
                    "success": True,
                    "answer": gemini_answer,
                    "video_id": video_id,
                    "video_url": request.video_url,
                    "source": "database",
                    "chunks_found": len(results)
                }
        except:
            pass
        
        # If not in database, try to get transcript
        if request.manual_transcript:
            # Use manually provided transcript
            transcript = handle_manual_transcript(request.video_url, request.manual_transcript)
            source_method = "manual"
        else:
            # Try automatic transcript retrieval
            try:
                transcript = get_youtube_transcript(request.video_url, use_fallback=True)
                source_method = "automatic"
            except Exception as transcript_error:
                # If automatic fails, provide helpful response
                return {
                    "success": False,
                    "error": "transcript_blocked",
                    "message": "YouTube transcript blocked and no manual transcript provided",
                    "details": str(transcript_error),
                    "video_id": video_id,
                    "video_url": request.video_url,
                    "solutions": [
                        "Provide manual transcript in 'manual_transcript' field",
                        "Upload video first using /upload-youtube/ with manual transcript",
                        "Use a proxy service",
                        "Process video locally"
                    ]
                }
        
        # Create chunks for context
        chunks = split_text_into_chunks(transcript, pdf_path=source_name)
        
        # Limit chunks for better performance
        relevant_chunks = chunks[:request.k] if len(chunks) > request.k else chunks
        gemini_answer = ask_gemini(request.query, relevant_chunks)
        
        return {
            "success": True,
            "answer": gemini_answer,
            "video_id": video_id,
            "video_url": request.video_url,
            "source": f"direct_transcript_{source_method}",
            "chunks_used": len(relevant_chunks)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": "query_failed",
            "message": "Failed to query YouTube video",
            "details": str(e),
            "video_url": request.video_url if hasattr(request, 'video_url') else "unknown"
        }

@app.get("/videos/")
async def list_available_videos():
    """Get list of all uploaded YouTube videos"""
    try:
        documents = get_available_documents()
        # Filter for video sources
        videos = [doc for doc in documents if doc.startswith("video_") and doc.endswith(".txt")]
        
        # Extract video IDs
        video_info = []
        for video in videos:
            video_id = video.replace("video_", "").replace(".txt", "")
            video_info.append({
                "video_id": video_id,
                "source_name": video,
                "youtube_url": f"https://www.youtube.com/watch?v={video_id}"
            })
        
        return {
            "videos": video_info,
            "count": len(video_info)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/smart_query_all/")
async def smart_query_all_content(request: SmartVideoQueryRequest):
    """Intelligent query across PDFs, YouTube videos, or both"""
    try:
        documents = get_available_documents()
        
        # Filter by content type
        if request.content_type == "pdf":
            filtered_docs = [doc for doc in documents if not doc.startswith("video_")]
        elif request.content_type == "video":
            filtered_docs = [doc for doc in documents if doc.startswith("video_")]
        else:  # "all"
            filtered_docs = documents
        
        if not filtered_docs:
            return {
                "answer": f"No {request.content_type} content found in the database.",
                "strategy_used": request.strategy,
                "content_type": request.content_type
            }
        
        # Use the existing smart search but specify we want all content types
        search_result = smart_document_search(request.query, k=request.k, strategy=request.strategy)
        
        if not search_result["chunks"]:
            return {
                "answer": f"No relevant information found in {request.content_type} content for your query.",
                "strategy_used": request.strategy,
                "content_type": request.content_type
            }
        
        gemini_answer = ask_gemini(request.query, search_result["chunks"])
        
        response = {
            "answer": gemini_answer,
            "strategy_used": search_result["strategy_used"],
            "chunks_used": len(search_result["chunks"]),
            "content_type": request.content_type
        }
        
        # Add strategy-specific metadata
        if "primary_source" in search_result:
            source = search_result["primary_source"]
            response["primary_source"] = source
            response["source_type"] = "video" if source.startswith("video_") else "pdf"
            
        if "sources_used" in search_result:
            response["sources_used"] = search_result["sources_used"]
            
        if "document_scores" in search_result:
            response["source_relevance_scores"] = search_result["document_scores"]
            
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/content_summary/")
async def get_content_summary():
    """Get summary of all available content (PDFs and videos)"""
    try:
        summary = get_content_type_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/youtube_status/")
async def check_youtube_api_status():
    """Check if YouTube transcript API is accessible from current environment"""
    test_video_id = "dQw4w9WgXcQ"  # Rick Roll - commonly available video
    test_url = f"https://www.youtube.com/watch?v={test_video_id}"
    
    try:
        # Try to get transcript for a well-known video
        transcript = get_youtube_transcript(test_url, use_fallback=True)
        return {
            "status": "accessible",
            "message": "YouTube transcript API is working",
            "test_video": test_url,
            "environment": "local_or_compatible",
            "recommendations": [
                "You can use automatic transcript retrieval",
                "All YouTube endpoints should work normally"
            ]
        }
    except Exception as e:
        error_str = str(e)
        
        if "cloud provider" in error_str.lower() or "ip" in error_str.lower():
            return {
                "status": "blocked_cloud",
                "message": "YouTube API blocked - Cloud provider IP detected",
                "test_video": test_url,
                "environment": "cloud_deployment",
                "error_details": error_str,
                "solutions": [
                    "Use manual transcript upload",
                    "Implement proxy service",
                    "Process videos locally before deployment",
                    "Use alternative transcript services"
                ],
                "workaround_endpoints": [
                    "POST /upload-youtube/ with manual_transcript field",
                    "POST /query_youtube/ with manual_transcript field"
                ]
            }
        else:
            return {
                "status": "error_other",
                "message": "YouTube API error - Not cloud-related",
                "test_video": test_url,
                "error_details": error_str,
                "recommendations": [
                    "Check internet connectivity",
                    "Verify video accessibility",
                    "Check for API rate limits"
                ]
            }


