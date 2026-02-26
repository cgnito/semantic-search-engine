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

1. **Clone the repo:**
   ```bash
  git clone https://github.com/cgnito/semantic-search-engine.git
  cd semantic-search-engine
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
3. **Run the app:**
   ```bash
   streamlit run eng.py   
## Future Improvements, hopefully

- [ ] **Incremental Sync:** Update the logic to only embed new tweets instead of re-indexing the whole file if it's missing.
- [ ] **Media Preview:** Enhance metadata to include image links or video thumbnails from the tweets.
- [ ] **Advanced Filtering:** Add a sidebar UI to filter results by year, month, or tweet length.
- [ ] **Topic Clustering:** Use Unsupervised Learning (K-Means) to automatically group tweets into categories like "Tech," "Life," or "Sports."
- [ ] **X API Integration:** Move away from manual JSON uploads and allow users to login with X to search their own timeline in real-time.