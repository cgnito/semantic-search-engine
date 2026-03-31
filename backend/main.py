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

# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)
index_name = "tweet-index"
index = pc.Index(index_name)

class QueryRequest(BaseModel):
    query: str

def tweet_generator():
    file_path = "tweets.json"
    if not os.path.exists(file_path):
        return
    with open(file_path, "r", encoding="utf-8") as f:
        for record in ijson.items(f, "item"):
            yield {
                "text": record["tweet"]["full_text"],
                "date": record["tweet"]["created_at"]
            }

@app.on_event("startup")
async def startup_event():
    try:
        stats = index.describe_index_stats()
        if stats['total_vector_count'] == 0:
            batch_size = 50 
            current_batch = []
            count = 0
            for tweet_data in tweet_generator():
                current_batch.append({
                    "id": f"tweet_{count}",
                    "metadata": {
                        "text": tweet_data["text"],
                        "date": tweet_data["date"]
                    }
                })
                count += 1
                if len(current_batch) >= batch_size:
                    index.upsert(vectors=current_batch)
                    current_batch = []
            if current_batch:
                index.upsert(vectors=current_batch)
    except Exception:
        pass

@app.post("/search")
async def search(request: QueryRequest):
    results = index.query(
        top_k=10,
        include_metadata=True,
        inputs={"text": request.query} 
    )
    
    return [
        {
            "text": match["metadata"]["text"],
            "date": match["metadata"]["date"]
        }
        for match in results["matches"]
    ]