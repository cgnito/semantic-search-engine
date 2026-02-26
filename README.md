## 🐦 Abdulrahmon's Tweet Semantic Search Engine

An AI-powered search tool that allows you to search through my Twitter history using **meanings** rather than just keywords. Unlike "standard" twitter search, this engine understands context—searching e.g "coding" will find tweets about Python, JavaScript, or debugging even if the word "coding" isn't in the tweet.

### Simple Explanation
Most search engines look for exact words. This app uses **Machine Learning** to turn my tweets into mathematical "vectors" (embeddings). When you type a query, the app calculates which tweets are mathematically closest to the meaning of your search and shows you the top 10 matches instantly.

### The Stack
* **UI Framework:** [Streamlit](https://streamlit.io/) for a responsive, Python-based web interface.
* **Vector Database:** [ChromaDB](https://www.trychroma.com/) (Persistent) to store and query embeddings locally.
* **Embedding Model:** `all-MiniLM-L6-v2` via **Sentence-Transformers**. It's a lightweight yet powerful model that converts text into 384-dimensional vectors.
* **Data Processing:** `ijson` for iterative JSON parsing, allowing the app to handle large Twitter archives without crashing the system's RAM.

### How it Works
1.  **Ingestion:** The app reads the raw `tweets.json` using a generator to keep memory usage low.
2.  **Vectorization:** Tweets are processed in batches of 256. The model generates embeddings for each batch.
3.  **Storage:** ChromaDB stores the text, metadata (date), and the vector.
4.  **Semantic Querying:** When a user enters a query, the search term is vectorized using the same model, and a **Cosine Similarity** search is performed against the database.
5.  **Caching:** Uses `@st.cache_resource` to keep the 100MB+ AI model in memory, ensuring search results appear in milliseconds.

## ⚙️ Setup & Installation

### Clone the Repository

```bash
git clone https://github.com/cgnito/semantic-search-engine.git
cd semantic-search-engine
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Run the App
```bash
streamlit run eng.py
```
---

## Future Improvements

- [ ] **Incremental Sync:** Update the logic to embed only new tweets instead of re-indexing the entire dataset when embeddings are missing.

- [ ] **Media Preview:** Enhance metadata to include image URLs or video thumbnails from tweets.

- [ ] **Advanced Filtering:** Add a sidebar UI to filter results by year, month, or tweet length.

- [ ] **Topic Clustering:** Use unsupervised learning (e.g., K-Means) to automatically group tweets into categories such as "Tech," "Life," or "Sports."

- [ ] **X API Integration:** Replace manual JSON uploads with real-time authentication using the X API, allowing users to search their own timelines.