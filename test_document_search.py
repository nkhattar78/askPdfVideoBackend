"""
Test script for the enhanced document search functionality
Run this after starting the FastAPI server with: uvicorn app.main:app --reload
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_document_apis():
    """Test the new document search APIs"""
    
    print("🧪 Testing Enhanced Document Search APIs\n")
    
    # Test 1: List available documents
    print("1. 📋 Testing document listing...")
    try:
        response = requests.get(f"{BASE_URL}/documents/")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Found {data['count']} documents:")
            for doc in data['documents']:
                print(f"      - {doc}")
        else:
            print(f"   ❌ Error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
    
    print()
    
    # Test 2: Smart query
    print("2. 🧠 Testing smart query...")
    test_query = "What is this document about?"
    
    try:
        response = requests.post(f"{BASE_URL}/smart_query/", json={
            "query": test_query,
            "strategy": "best_match",
            "k": 3
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Smart query successful!")
            print(f"      Strategy used: {data.get('strategy_used')}")
            print(f"      Primary document: {data.get('primary_document', 'N/A')}")
            print(f"      Chunks used: {data.get('chunks_used', 0)}")
            print(f"      Answer: {data.get('answer', 'No answer')[:100]}...")
        else:
            print(f"   ❌ Error: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
    
    print()
    
    # Test 3: Query specific document (if documents exist)
    print("3. 📄 Testing document-specific query...")
    try:
        # First get documents
        docs_response = requests.get(f"{BASE_URL}/documents/")
        if docs_response.status_code == 200:
            docs = docs_response.json()['documents']
            if docs:
                first_doc = docs[0]
                response = requests.post(f"{BASE_URL}/query_document/", json={
                    "query": test_query,
                    "document_name": first_doc,
                    "k": 3
                })
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"   ✅ Document query successful!")
                    print(f"      Document queried: {data.get('document_queried')}")
                    print(f"      Chunks found: {data.get('chunks_found', 0)}")
                    print(f"      Answer: {data.get('answer', 'No answer')[:100]}...")
                else:
                    print(f"   ❌ Error: {response.status_code}")
            else:
                print("   ⚠️  No documents available to test")
        else:
            print("   ❌ Could not get document list")
    except Exception as e:
        print(f"   ❌ Connection error: {e}")

if __name__ == "__main__":
    print("Make sure your FastAPI server is running with:")
    print("uvicorn app.main:app --reload")
    print("\nPress Enter to start testing...")
    input()
    
    test_document_apis()
    
    print("\n🎉 Testing complete!")
    print("\nNext steps:")
    print("1. Upload some PDFs using /upload-pdf/")
    print("2. Try different search strategies: 'best_match', 'multi_doc', 'single_source'")
    print("3. Use /documents/ to see all available documents")
    print("4. Use /query_document/ to search specific documents")
