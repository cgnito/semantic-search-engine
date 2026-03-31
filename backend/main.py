import os
import ijson
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "tweet-index"
index = pc.Index(index_name)

model = SentenceTransformer("all-MiniLM-L6-v2")

class QueryRequest(BaseModel):
    query: str

def tweet_generator():
    if not os.path.exists("tweets.json"):
        return
    with open("tweets.json", "r", encoding="utf-8") as f:
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
            batch_size = 100 
            current_batch = []
            count = 0

            for tweet_data in tweet_generator():
                tweet_id = f"tweet_{count}"
                vector = model.encode(tweet_data["text"]).tolist()
                current_batch.append({
                    "id": tweet_id,
                    "values": vector,
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
    except Exception as e:
        pass

@app.post("/search")
async def search(request: QueryRequest):
    query_vector = model.encode(request.query).tolist()
    results = index.query(
        vector=query_vector, 
        top_k=10, 
        include_metadata=True
    )
    
    return [
        {
            "text": match["metadata"]["text"],
            "date": match["metadata"]["date"]
        }
        for match in results["matches"]
    ]