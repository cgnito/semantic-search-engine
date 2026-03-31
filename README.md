# 🐦 abdurahmon's tweet semantic search engine (v1.0)

a high-performance retrieval system that performs asymmetric semantic search across a twitter archive. instead of basic keyword matching, this system uses pinecone's serverless vector database and integrated cloud embeddings to map tweets and queries into a shared 384-dimensional dense vector space.

## what it does (simply put)

most search engines match exact strings. this application uses machine learning embeddings to represent the "idea" of a tweet as a set of coordinates (vectors). when you search for "productivity", the system calculates which tweets are mathematically closest to that concept via pinecone's cloud inference, even if the specific word "productivity" never appears in the text.

## the stack

- **backend:** fastapi (python 3.12) - high-performance asynchronous api
- **frontend:** next.js 15 + tailwind css - modern, responsive search interface
- **vector database:** pinecone - serverless cloud vector storage and metadata retrieval
- **model:** llama-text-embed-v2 (pinecone integrated) - 384-dimensional cloud-based embeddings
- **data processing:** ijson - memory-efficient json streaming for large archives

## how it works

- **memory-efficient ingestion:** streams your local `tweets.json` using a python generator, handling large archives without memory issues.
- **cloud vectorization:** uses pinecone's integrated embedding pipeline to convert tweet text into vectors automatically during upsert.
- **serverless storage:** stores embeddings and tweet metadata in a pinecone index for fast, scalable retrieval.
- **semantic retrieval:** converts user queries into vectors in the cloud and performs cosine similarity search to return the top 10 most relevant matches.
- The system utilizes Pinecone Integrated Inference via the llama-text-embed-v2 model, offloading all embedding generation to the cloud. This architectural decision ensures the backend remains lightweight and responsive on resource-constrained environments by eliminating the need for local vectorization libraries and heavy machine learning models. Additionally, a dedicated ingestion synchronization endpoint was implemented to manage high-volume data migration, ensuring stable uploads that bypass standard serverless startup timeouts.

## project structure & data handling

important: to protect privacy and stay within github file limits, the raw data and environment keys are not included in the repository.

```plaintext
semantic-search-engine/
├── backend/
│   ├── main.py            # fastapi server logic & pinecone integration
│   ├── requirements.txt   # python dependencies
│   ├── .env               # [local only] pinecone api keys
│   └── tweets.json        # [local only] your personal twitter archive
├── frontend/
│   ├── src/app/           # next.js application router
│   └── src/lib/api.ts     # frontend api client
└── .gitignore             # ignores .venv, .env, and tweets.json
```

## setup & installation

### 1. clone & configure

```bash
git clone https://github.com/cgnito/semantic-search-engine.git
cd semantic-search-engine
```

### 2. pinecone setup

- create a free account at pinecone.io.
- create a new index named `tweet-index`.
- select serverless (aws us-east-1).
- set dimensions to 384 and metric to cosine.
- enable integrated embedding and select `llama-text-embed-v2`.
- set the field map source to `text`.
- copy your api key into `backend/.env`:

```env
PINECONE_API_KEY=your_key
```

### 3. backend setup

place your `tweets.json` inside the `backend/` folder.

```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate  # windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### 4. frontend setup

```bash
cd frontend
npm install
npm run dev
```

## api reference

### search tweets

- **endpoint:** `POST /search`
- **request body:**
```json
{"query": "startup growth tips"}
```

- **response:**
```json
[{"text": "...", "date": "..."}]
```  