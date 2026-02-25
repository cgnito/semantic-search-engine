from sentence_transformers import SentenceTransformer
import chromadb
import ijson
import os

# initialize model and db
DB_PATH = "./chroma_db"
COLLECTION_NAME = "abduls_tweets"
model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# get tweets from json in a generator way, cuz of size
if not os.path.exists(DB_PATH):
    print("First time. Starting the slow embedding process (this takes a few minutes)...")
    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    def tweet_generator():
        with open("tweets.json", "r") as f:
            for record in ijson.items(f, "item"):
                yield {
                    "text": record["tweet"]["full_text"],
                    "date": record["tweet"]["created_at"]
                }

    batch_size = 256
    current_texts = []
    current_metadatas = []
    count = 0

    # add tweets and metadatas to list
    for tweet_data in tweet_generator(): # each result from generator
        current_texts.append(tweet_data["text"]) 
        current_metadatas.append({"date": tweet_data["date"]})

        # when list gets to 256 embed them
        if len(current_texts) >= batch_size:
            embeddings = model.encode(current_texts).tolist()
            ids = [f"id_{i}" for i in range(count, count+len(current_texts))]
            # add embeded tweets to collection
            collection.add(
                documents=current_texts,
                embeddings=embeddings,
                metadatas=current_metadatas,
                ids=ids
            )
            count += len(current_texts)
            print(f"Indexed {count} tweets...")
            
            # clear the batch to start fresh
            current_texts = []
            current_metadatas = []

    # remaining tweets
    if current_texts:
        embeddings = model.encode(current_texts).tolist()
        ids = [f"id_{i}" for i in range(count, count+len(current_texts))]

        collection.add(
            documents=current_texts,
            embeddings=embeddings,
            metadatas=current_metadatas,
            ids=ids
        )
        count += len(current_texts)
        print(f"Finished! Total stored:{count} tweets...")

# if db is available
else:
    print("Loading existing collection...")
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

print("\n--- Semantic Search Engine ---")
user_query = input("Search Abdul tweets: ")

# query
results = collection.query(
    query_texts=[user_query],
    n_results=5
)

# results
print("\nTop matches found in your history:")
for tweet, meta in zip(results['documents'][0], results['metadatas'][0]):
    print(f"📅 {meta['date']}")
    print(f"💬 {tweet}")
    print("-" * 30)