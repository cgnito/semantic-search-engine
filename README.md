# 🐦 abdulrahmon's tweet semantic search engine

this engine implements a vector-based retrieval system that performs asymmetric semantic search across a local twitter archive using sentence-level embeddings. unlike traditional inverted-index keyword searches, this system utilizes a transformer-based encoder to map tweets and queries into a shared 384-dimensional dense vector space, enabling discovery based on latent topical similarity and intent.

### what it does (simply put)

most search engines just match words. this app uses **machine learning** to convert tweets into numerical "embeddings." when you search for a concept like "productivity," the system calculates which tweets are mathematically closest to that idea, even if they don't contain the specific word "productivity."

### the stack

- **ui framework:** streamlit for an interactive, reactive python web interface.

- **vector database:** chromadb for persistent storage and fast nearest-neighbor (knn) lookups.

- **embedding model:** `all-minilm-l6-v2` via sentence-transformers (sbert), providing high-quality semantic mapping at low latency.

- **data processing:** ijson for iterative json parsing to handle large archives with minimal memory overhead.

### how it works

1. memory-efficient ingestion: the app streams the `tweets.json` archive using a generator, ensuring the system doesn't crash regardless of archive size.

2. vectorization: tweets are processed in batches of 256. the sbert model encodes each tweet into a vector representing its semantic meaning.

3. local storage: chromadb stores the document text, date metadata, and the high-dimensional vectors.

4. semantic retrieval: user queries are encoded on-the-fly, and a cosine similarity search is performed against the database to find the top 10 most relevant matches.

5. sqlite compatibility: uses a `pysqlite3` injection to ensure compatibility with modern chromadb requirements in various environments.

### setup & installation

```bash
# clone the repository
git clone https://github.com/cgnito/semantic-search-engine.git
cd semantic-search-engine
```

```bash
#create a virtual environment
python -m venv venv
source venv/bin/activate  # on windows: venv\scripts\activate
```

```bash
# install dependencies
pip install -r requirements.txt
```

```bash
# run the app
streamlit run eng.py
```

### future improvements

- better ui: transition to a more polished, custom-branded interface.

- api-based embeddings: moving to cloud-hosted embedding apis for higher dimensionality and improved retrieval quality.

- proper top-k search tuning: refining retrieval parameters for better precision and recall.

- live tweet indexing: replacing the static json archive with real-time x api integration.

---

this is just an mvp of a bigger idea... more soon.