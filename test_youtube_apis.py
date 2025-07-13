"""
Test script for YouTube Video Query APIs
Run this after starting the FastAPI server with: uvicorn app.main:app --reload
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_youtube_apis():
    """Test the new YouTube video APIs"""
    
    print("🎥 Testing YouTube Video Query APIs\n")
    
    # Example video URL (replace with any public YouTube video)
    test_video_url = "https://www.youtube.com/watch?v=kqtD5dpn9C8"  # Python tutorial
    test_query = "What is this video about?"
    
    print(f"Using test video: {test_video_url}")
    print(f"Using test query: '{test_query}'\n")
    
    # Test 1: Content Summary
    print("1. 📊 Testing content summary...")
    try:
        response = requests.get(f"{BASE_URL}/content_summary/")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Content Summary:")
            print(f"      Total documents: {data.get('total_documents', 0)}")
            print(f"      PDFs: {data.get('pdfs', {}).get('count', 0)}")
            print(f"      Videos: {data.get('videos', {}).get('count', 0)}")
        else:
            print(f"   ❌ Error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
    
    print()
    
    # Test 2: Upload YouTube Video
    print("2. 📤 Testing YouTube video upload...")
    try:
        response = requests.post(f"{BASE_URL}/upload-youtube/", json={
            "video_url": test_video_url
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Video upload successful!")
            print(f"      Video ID: {data.get('video_id')}")
            print(f"      Source: {data.get('source')}")
            print(f"      Chunks: {data.get('num_chunks')}")
        else:
            print(f"   ❌ Error: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
    
    print()
    
    # Test 3: List Videos
    print("3. 📺 Testing video listing...")
    try:
        response = requests.get(f"{BASE_URL}/videos/")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Found {data['count']} videos:")
            for video in data['videos']:
                print(f"      - Video ID: {video['video_id']}")
                print(f"        URL: {video['youtube_url']}")
        else:
            print(f"   ❌ Error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
    
    print()
    
    # Test 4: Query Specific Video
    print("4. 🎥 Testing specific video query...")
    try:
        response = requests.post(f"{BASE_URL}/query_youtube/", json={
            "query": test_query,
            "video_url": test_video_url,
            "k": 3
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Video query successful!")
            print(f"      Video ID: {data.get('video_id')}")
            print(f"      Source: {data.get('source')}")
            print(f"      Chunks used: {data.get('chunks_used', 0)}")
            print(f"      Answer: {data.get('answer', 'No answer')[:150]}...")
        else:
            print(f"   ❌ Error: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
    
    print()
    
    # Test 5: Smart Query All Content Types
    print("5. 🧠 Testing smart query across all content...")
    for content_type in ["all", "video", "pdf"]:
        try:
            response = requests.post(f"{BASE_URL}/smart_query_all/", json={
                "query": test_query,
                "strategy": "best_match",
                "content_type": content_type,
                "k": 3
            })
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Smart query ({content_type}) successful!")
                print(f"      Strategy: {data.get('strategy_used')}")
                print(f"      Content type: {data.get('content_type')}")
                print(f"      Primary source: {data.get('primary_source', 'N/A')}")
                print(f"      Source type: {data.get('source_type', 'N/A')}")
                print(f"      Chunks used: {data.get('chunks_used', 0)}")
                print(f"      Answer: {data.get('answer', 'No answer')[:100]}...")
            else:
                print(f"   ❌ Error ({content_type}): {response.status_code}")
        except Exception as e:
            print(f"   ❌ Connection error ({content_type}): {e}")
        print()

def test_integration_scenarios():
    """Test real-world integration scenarios"""
    
    print("🔄 Testing Integration Scenarios\n")
    
    # Scenario 1: Mixed content search
    print("Scenario 1: 🔍 Mixed content intelligent search")
    try:
        response = requests.post(f"{BASE_URL}/smart_query_all/", json={
            "query": "What are the key concepts?",
            "strategy": "multi_doc", 
            "content_type": "all",
            "k": 5
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Found content from {len(data.get('sources_used', []))} sources")
            print(f"   Sources: {data.get('sources_used', [])}")
        else:
            print(f"   ❌ Error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # Scenario 2: Content type comparison
    print("Scenario 2: 📊 Content type comparison")
    query = "Explain the main topic"
    
    for content_type in ["pdf", "video"]:
        try:
            response = requests.post(f"{BASE_URL}/smart_query_all/", json={
                "query": query,
                "content_type": content_type,
                "k": 2
            })
            
            if response.status_code == 200:
                data = response.json()
                answer_preview = data.get('answer', 'No answer')[:80] + "..."
                print(f"   {content_type.upper()}: {answer_preview}")
            else:
                print(f"   {content_type.upper()}: No content available")
        except Exception as e:
            print(f"   {content_type.upper()}: Error - {e}")

if __name__ == "__main__":
    print("🚀 YouTube Video API Testing Suite")
    print("=" * 50)
    print("Make sure your FastAPI server is running with:")
    print("uvicorn app.main:app --reload")
    print("\nPress Enter to start testing...")
    input()
    
    # Run basic API tests
    test_youtube_apis()
    
    print("\n" + "=" * 50)
    
    # Run integration tests
    test_integration_scenarios()
    
    print("\n🎉 Testing complete!")
    print("\nNext steps:")
    print("1. Try uploading different YouTube videos")
    print("2. Test with videos + PDFs for mixed content search")
    print("3. Experiment with different search strategies")
    print("4. Use /content_summary/ to monitor your knowledge base")
    print("5. Build a React frontend to use these APIs!")
    
    print("\n📝 API Endpoints Available:")
    endpoints = [
        "GET  /content_summary/     - Overview of all content",
        "POST /upload-youtube/      - Upload YouTube video",
        "GET  /videos/              - List all videos", 
        "POST /query_youtube/       - Query specific video",
        "POST /smart_query_all/     - Smart search across all content",
        "POST /upload-pdf/          - Upload PDF (existing)",
        "POST /smart_query/         - Smart PDF search (existing)"
    ]
    
    for endpoint in endpoints:
        print(f"   {endpoint}")
