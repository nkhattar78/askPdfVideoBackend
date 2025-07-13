from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain.schema import Document
from collections import Counter
from typing import List, Optional, Dict, Any
import numpy as np

# # Qdrant configuration: Shreya
QDRANT_URL = "https://a26f47f6-04bd-4433-bc1d-30c41f48f0cd.europe-west3-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.ZHDktyBPFyC7qCHT-zmPwr8URuKZ3_PwB-D4xuHCwj0"

# # Qdrant configuration: Naveen
# QDRANT_URL = "https://3f8cfe0a-f10c-4f98-ba5f-3ebef081925b.eu-central-1-0.aws.cloud.qdrant.io"
# QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.o7iGfac_jEvGDJG8JwXSz9fh_WOV_FhFfJ7VZUu9niA"
QDRANT_COLLECTION_PDF_EMBEDDINGS = "pdfs_embeddings"
QDRANT_COLLECTION_PDF_SUMMARY = "pdfs_summary"


Document(
    page_content="This is a chunk of text from a PDF.",
    metadata={"source": "employee_handbook.pdf"}
)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def create_embeddings_and_store_qdrant(docs):
    Qdrant.from_documents(
        documents=docs,
        embedding=embedding_model,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION_PDF_EMBEDDINGS
    )

def qdrant_similarity_search(query, k=3):
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vectorstore = Qdrant(
        client=client,
        collection_name=QDRANT_COLLECTION_PDF_EMBEDDINGS,
        embeddings=embedding_model
    )
    retrieved_chunks = vectorstore.similarity_search(query, k=k)

    sources = [doc.metadata.get("source") for doc in retrieved_chunks]
    most_common_source = Counter(sources).most_common(1)[0][0]
    filtered_chunks = [
        doc for doc in retrieved_chunks
        if doc.metadata.get("source") == most_common_source
    ]
    return filtered_chunks

def get_available_documents() -> List[str]:
    """Get list of all available documents in the database"""
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        vectorstore = Qdrant(
            client=client,
            collection_name=QDRANT_COLLECTION_PDF_EMBEDDINGS,
            embeddings=embedding_model
        )
        
        # Get all documents to extract unique sources
        all_docs = vectorstore.similarity_search("", k=1000)  # Large k to get many docs
        sources = set([doc.metadata.get("source") for doc in all_docs if doc.metadata.get("source")])
        return sorted(list(sources))
    except Exception as e:
        print(f"Error getting documents: {e}")
        return []

def search_specific_document(query: str, document_name: str, k: int = 3) -> List[Document]:
    """Search within a specific document only"""
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vectorstore = Qdrant(
        client=client,
        collection_name=QDRANT_COLLECTION_PDF_EMBEDDINGS,
        embeddings=embedding_model
    )
    
    # Get more results to filter by document
    retrieved_chunks = vectorstore.similarity_search(query, k=k*10)
    
    # Filter by specific document
    filtered_chunks = [
        doc for doc in retrieved_chunks
        if doc.metadata.get("source") == document_name
    ][:k]
    
    return filtered_chunks

def smart_document_search(query: str, k: int = 3, strategy: str = "best_match") -> Dict[str, Any]:
    """
    Intelligent search that determines the best document(s) to query
    
    Strategies:
    - "best_match": Find most relevant document and search within it
    - "multi_doc": Search across all docs but group by relevance
    - "single_source": Original approach - filter to most common source
    """
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vectorstore = Qdrant(
        client=client,
        collection_name=QDRANT_COLLECTION_PDF_EMBEDDINGS,
        embeddings=embedding_model
    )
    
    if strategy == "best_match":
        # Get top results and find the most relevant document
        retrieved_chunks = vectorstore.similarity_search_with_score(query, k=k*5)
        
        # Calculate average score per document
        doc_scores = {}
        doc_chunks = {}
        
        for doc, score in retrieved_chunks:
            source = doc.metadata.get("source", "unknown")
            if source not in doc_scores:
                doc_scores[source] = []
                doc_chunks[source] = []
            doc_scores[source].append(score)
            doc_chunks[source].append(doc)
        
        # Find document with best average score
        best_doc = min(doc_scores.keys(), key=lambda x: np.mean(doc_scores[x]))
        best_chunks = doc_chunks[best_doc][:k]
        
        return {
            "chunks": best_chunks,
            "primary_source": best_doc,
            "strategy_used": "best_match",
            "document_scores": {doc: np.mean(scores) for doc, scores in doc_scores.items()}
        }
    
    elif strategy == "multi_doc":
        # Return results from multiple documents with source grouping
        retrieved_chunks = vectorstore.similarity_search_with_score(query, k=k*3)
        
        # Group by source and take top chunks from each
        doc_groups = {}
        for doc, score in retrieved_chunks:
            source = doc.metadata.get("source", "unknown")
            if source not in doc_groups:
                doc_groups[source] = []
            doc_groups[source].append((doc, score))
        
        # Take top 2-3 chunks from top 2-3 documents
        final_chunks = []
        sources_used = []
        
        for source in sorted(doc_groups.keys(), key=lambda x: min([score for _, score in doc_groups[x]])):
            if len(sources_used) < 3:  # Max 3 different sources
                chunks_from_source = [doc for doc, _ in sorted(doc_groups[source], key=lambda x: x[1])[:2]]
                final_chunks.extend(chunks_from_source)
                sources_used.append(source)
        
        return {
            "chunks": final_chunks[:k],
            "sources_used": sources_used,
            "strategy_used": "multi_doc"
        }
    
    else:  # "single_source" - original approach
        retrieved_chunks = vectorstore.similarity_search(query, k=k)
        sources = [doc.metadata.get("source") for doc in retrieved_chunks]
        most_common_source = Counter(sources).most_common(1)[0][0]
        filtered_chunks = [
            doc for doc in retrieved_chunks
            if doc.metadata.get("source") == most_common_source
        ]
        
        return {
            "chunks": filtered_chunks,
            "primary_source": most_common_source,
            "strategy_used": "single_source"
        }

def search_videos_only(query: str, k: int = 3) -> List[Document]:
    """Search only within YouTube video transcripts"""
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vectorstore = Qdrant(
        client=client,
        collection_name=QDRANT_COLLECTION_PDF_EMBEDDINGS,
        embeddings=embedding_model
    )
    
    # Get more results to filter by video sources
    retrieved_chunks = vectorstore.similarity_search(query, k=k*10)
    
    # Filter by video sources only
    video_chunks = [
        doc for doc in retrieved_chunks
        if doc.metadata.get("source", "").startswith("video_")
    ][:k]
    
    return video_chunks

def search_pdfs_only(query: str, k: int = 3) -> List[Document]:
    """Search only within PDF documents"""
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vectorstore = Qdrant(
        client=client,
        collection_name=QDRANT_COLLECTION_PDF_EMBEDDINGS,
        embeddings=embedding_model
    )
    
    # Get more results to filter by PDF sources
    retrieved_chunks = vectorstore.similarity_search(query, k=k*10)
    
    # Filter by PDF sources only (exclude video sources)
    pdf_chunks = [
        doc for doc in retrieved_chunks
        if not doc.metadata.get("source", "").startswith("video_")
    ][:k]
    
    return pdf_chunks

def get_content_type_summary() -> Dict[str, Any]:
    """Get summary of available content types"""
    try:
        documents = get_available_documents()
        
        videos = [doc for doc in documents if doc.startswith("video_")]
        pdfs = [doc for doc in documents if not doc.startswith("video_")]
        
        return {
            "total_documents": len(documents),
            "videos": {
                "count": len(videos),
                "sources": videos
            },
            "pdfs": {
                "count": len(pdfs),
                "sources": pdfs
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "total_documents": 0,
            "videos": {"count": 0, "sources": []},
            "pdfs": {"count": 0, "sources": []}
        }
