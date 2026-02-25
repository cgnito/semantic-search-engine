__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from sentence_transformers import SentenceTransformer
import chromadb
import ijson
import os
import streamlit as st

# initialize model, db and page
st.set_page_config(page_title="Abdulrahmon's Tweet Search", page_icon="🐦")
DB_PATH = "./chroma_db"
COLLECTION_NAME = "abduls_tweets"

# cache model and db
@st.cache_resource
def get_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    return model, chroma_client

model, chroma_client = get_resources()

def build_db_if_needed():
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    # check if the collection is empty
    if collection.count() == 0:
        st.info("First time setup. Starting the embedding process...")

        def tweet_generator():
            with open("tweets.json", "r") as f:
                for record in ijson.items(f, "item"):
                    yield {
                        "text": record["tweet"]["full_text"],
                        "date": record["tweet"]["created_at"]
                    }
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        batch_size = 256
        current_texts, current_metadatas, count = [], [], 0

        for tweet_data in tweet_generator():
            current_texts.append(tweet_data["text"]) 
            current_metadatas.append({"date": tweet_data["date"]})

            if len(current_texts) >= batch_size:
                embeddings = model.encode(current_texts).tolist()
                ids = [f"id_{i}" for i in range(count, count + len(current_texts))]
                collection.add(
                    documents=current_texts,
                    embeddings=embeddings,
                    metadatas=current_metadatas,
                    ids=ids
                )
                count += len(current_texts)
                status_text.text(f"Indexed {count} tweets...")
                progress_bar.progress(min(count / 3500, 1.0))
                
                current_texts, current_metadatas = [], []

        # handle leftovers
        if current_texts:
            embeddings = model.encode(current_texts).tolist()
            ids = [f"id_{i}" for i in range(count, count + len(current_texts))]
            collection.add(
                documents=current_texts, 
                embeddings=embeddings, 
                metadatas=current_metadatas, 
                ids=ids
            )

        st.success("Indexing complete!")
    
    return collection

collection = build_db_if_needed()

st.title("Abdulrahmon's Tweets Semantic Search")
st.markdown("Enter a topic or phrase to search through my tweet history using AI.")

user_query = st.text_input("Search Abdul's tweets:", placeholder="e.g., coding tips, football, startup life...")

if user_query:
    with st.spinner("Searching through history..."):
        # query database
        results = collection.query(query_texts=[user_query], n_results=10)
        
        # results
        st.subheader(f"Top matches for: '{user_query}'")
        
        if results['documents'][0]:
            for tweet, meta in zip(results['documents'][0], results['metadatas'][0]):
                with st.expander(f"📅 {meta['date']}"):
                    st.write(tweet)
        else:
            st.warning("No matches found. Try a different search term.")