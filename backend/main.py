import os
import ijson
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)
index_name = "tweet-index"
index = pc.Index(index_name)

class QueryRequest(BaseModel):
    query: str

def tweet_generator():
    file_path = "tweets.json"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found in {os.getcwd()}")
        return
    with open(file_path, "r", encoding="utf-8") as f:
        for record in ijson.items(f, "item"):
            yield {
                "id": record["tweet"]["id_str"],
                "text": record["tweet"]["full_text"],
                "date": record["tweet"]["created_at"]
            }

@app.get("/")
def health_check():
    return {"status": "online", "message": "Search engine is live"}

@app.get("/force-ingest")
async def force_ingest():
    count = 0
    try:
        batch_size = 50 
        current_batch = []
        print("Starting manual ingestion...")
        
        for tweet_data in tweet_generator():
        
            current_batch.append({
                "id": tweet_data["id"],
                "text": tweet_data["text"],
                "date": tweet_data["date"]
            })
            count += 1
            if len(current_batch) >= batch_size:
                index.upsert_records(namespace="tweets", records=current_batch)
                current_batch = []
                print(f"Uploaded {count} tweets...")

        if current_batch:
            index.upsert_records(namespace="tweets", records=current_batch)
            
        return {"status": "success", "tweets_uploaded": count}
    except Exception as e:
        print(f"Ingestion Error: {e}")
        return {"status": "failed", "error": str(e)}

@app.post("/search")
async def search(request: QueryRequest):

    results = index.search(
        namespace="tweets",
        query={
            "inputs": {"text": request.query},
            "top_k": 10
        }
    )
    
    return [
        {
            "text": hit["fields"]["text"],
            "date": hit["fields"]["date"]
        }
        for hit in results["result"]["hits"]
    ]