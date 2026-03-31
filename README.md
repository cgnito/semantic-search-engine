# 🐦 abdurahmon's tweet semantic search engine (v1.0)

a high-performance retrieval system that performs asymmetric semantic search across a twitter archive. instead of basic keyword matching, this system uses pinecone's serverless vector database and integrated embeddings to map tweets and queries into a shared 384-dimensional dense vector space, enabling discovery based on latent topical similarity and intent.

## what it does (simply put)

most search engines match exact strings. this application uses machine learning embeddings to represent the "idea" of a tweet as a set of coordinates (vectors). when you search for "productivity", the system calculates which tweets are mathematically closest to that concept via pinecone, even if the specific word "productivity" never appears in the text.

## the stack

- **backend:** fastapi (python 3.12) - high-performance asynchronous api  
- **frontend:** next.js 15 + tailwind css - modern, responsive search interface  
- **vector database:** pinecone - serverless cloud vector storage and metadata retrieval  
- **model:** all-minilm-l6-v2 (sbert) - high-quality semantic mapping with low latency 
- **data processing:** ijson - memory-efficient json streaming for large archives  

## how it works

- **memory-efficient ingestion:** streams your local `tweets.json` using a python generator, handling large archives without memory issues  
- **cloud vectorization:** uses pinecone's integrated embedding pipeline to convert tweet text into vectors automatically during upsert  
- **serverless storage:** stores embeddings and tweet metadata in a pinecone index for fast, scalable retrieval  
- **semantic retrieval:** converts user queries into vectors and performs cosine similarity search to return the top 10 most relevant matches  

## project structure & data handling

**important:** to protect privacy and stay within github file limits, the raw data and environment keys are not included in the repository.

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

create a free account at pinecone.io.

- create a new index named `tweet-index`
- SELECT CUSTOM
- set dimensions to `384` and metric to `cosine`
- index Name: tweet-index, dimensions: 384, metric: cosine, capacity code: serverless

copy your api key into a `.env` file inside the `backend/` folder:

```env
PINECONE_API_KEY=your_key_here
```

### 3. backend setup

place your `tweets.json` (exported from twitter) inside the `backend/` folder.

initialize the environment:

```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate  # windows: .venv\Scripts\activate
pip install -r requirements.txt
```

start the api:

```bash
uvicorn main:app --reload
```

the first run will stream your `tweets.json` and upsert the embeddings to your pinecone cloud index.

### 4. frontend setup

create a `.env.local` file inside the `frontend/` folder:
```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

```bash
cd frontend
npm install
npm run dev
```

## api reference

### search tweets

**endpoint:** `POST /search`

**request body:**

```json
{
  "query": "startup growth tips"
}
```

**successful response:**

```json
[
  {
    "text": "the hardest part of a startup is the first 10 customers...",
    "date": "wed oct 12 14:20:01 +0000 2022"
  }
]
```

## future improvements

- footprint analysis: automatic flagging of "risky" historical tweets for digital footprint cleanup  
- hybrid search: combining bm25 keyword ranking with semantic search for perfect precision  