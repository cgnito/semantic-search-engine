import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

st.set_page_config(page_title="Semantic Search Engine for Elon Musk Tweets", page_icon="🧠")
st.title("Search Elon Musk’s Mind")
st.write("""
This project uses Natural Language Processing (NLP) and embeddings 
to build a semantic search engine over Elon Musk’s tweet dataset.

Rather than simple keyword matching, this app understands meaning. 
Type in a topic or idea, and it will surface the most relevant tweets 
—even if they don’t contain the exact same words.
""")

# setup model and database
@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="elon_tweets")
    return model, collection

model, collection = load_resources()

# if db empty; index tweeets
if collection.count()==0:
    with st.spinner("Indexing tweets... this only happens once"):
        df = pd.read_csv("elon_musk_tweets.csv")
        sentences = df["text"].astype("str").tolist()
        embeddings = model.encode(sentences).tolist() # convert tweets to embeddings
        # add data to the chromadb collection
        collection.add(
            documents=sentences,
            embeddings=embeddings,
            metadatas=[{"date": str(d)} for d in df["date"]],
            ids=[f"id_{i}" for i in range(len(sentences))]
            )
    st.success("Database ready")

# input query, conert to embeddings and specify results
query = st.text_input("Search Elon's tweets: ", placeholder="e.g. mars or tesla")
if query:
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    # print results
    st.subheader("Results")
    for tweet, meta in zip(results['documents'][0], results['metadatas'][0]):
        with st.chat_message("user"):
            st.write(f"**{meta['date']}**")
            st.write(tweet)
         