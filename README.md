# luki-memory-service  
*Unified memory layer for LUKi: ELR® ingestion, vector search, session/KV memory*  
**PRIVATE / PROPRIETARY – Internal use only**

---

## 1. Overview  
`luki-memory-service` provides the persistence and retrieval layer for the LUKi agent and modules. It unifies:

- **Long‑term semantic memory** (vector DB of ELR® snippets, activity results, forum posts)  
- **Structured/KV memory** (facts, preferences, flags, consents)  
- **Ephemeral session memory** (short-term chat summaries)  
- **Ingestion pipelines** (chunking, embedding, enrichment, redaction)

It exposes a clean API (gRPC/HTTP) so any service (agent, reporting, engagement) can **store, search, and update** user memory safely.

---

## 2. Core Capabilities  
- **ELR Ingestion Pipeline** – Chunking, metadata tagging, sensitivity labels, embedding generation.  
- **Vector Retrieval** – KNN / hybrid search, filter by tags, time, consent scopes.  
- **KV Store** – Fast access to key facts (favorite music, mobility level) and agent state.  
- **Session Summaries** – Rolling conversation summaries & last-message buffers.  
- **Redaction & Access Control** – Strip PII, enforce per-field consent and role-based access.  
- **Audit & Versioning** – Immutable log of writes; soft delete & restore.

---

## 3. Tech Stack  
- **Vector DB:** ChromaDB (default) or FAISS; adapter for managed services (Pinecone, Qdrant)  
- **KV / Document Store:** PostgreSQL + pgvector or Redis/Mongo optional backends  
- **Embeddings:** sentence-transformers (local) or model-as-a-service via internal endpoint  
- **API Layer:** FastAPI (HTTP+JSON) & gRPC (optional)  
- **Schemas:** pydantic models; JSON Schema export  
- **ETL / Workers:** Celery / Dramatiq for async ingestion jobs

---

## 4. Repository Structure  
~~~text
luki_memory_service/
├── README.md
├── pyproject.toml
├── luki_memory/
│   ├── __init__.py
│   ├── config.py                    # env, DB urls, embedding model choice
│   ├── schemas/
│   │   ├── elr.py                   # ELR item, consent, sensitivity enums
│   │   ├── kv.py                    # key/value models
│   │   └── query.py                 # search requests/responses
│   ├── ingestion/
│   │   ├── chunker.py               # text/media chunking
│   │   ├── embedder.py              # embedding calls
│   │   ├── pipeline.py              # orchestration
│   │   └── redact.py                # PII/sensitive-field removal
│   ├── storage/
│   │   ├── vector_store.py          # Chroma/FAISS adapters
│   │   ├── kv_store.py              # Postgres/Redis adapters
│   │   └── session_store.py         # short-term memory
│   ├── api/
│   │   ├── http.py                  # FastAPI routes
│   │   └── grpc.proto               # gRPC definitions (optional)
│   ├── auth/
│   │   ├── rbac.py                  # role-based access checks
│   │   └── consent.py               # consent enforcement
│   ├── audit/
│   │   ├── logger.py
│   │   └── versions.py
│   └── utils/
│       └── ids.py                   # id generation, hashing, etc.
├── migrations/                      # DB migrations (Alembic)
├── scripts/
│   ├── run_dev.sh
│   └── load_demo_data.py
└── tests/
    ├── unit/
    └── integration/
~~~

---

## 5. Quick Start (Internal Dev)  
~~~bash
git clone git@github.com:REMELife/luki-memory-service.git
cd luki-memory-service
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# start local services (example: docker-compose)
docker compose up -d   # launches postgres + chroma containers
alembic upgrade head    # run DB migrations
python scripts/load_demo_data.py
uvicorn luki_memory.api.http:app --reload --port 8002
~~~

### Example: Ingest ELR text  
~~~python
import requests

payload = {
  "user_id": "user_123",
  "text": "Alice loves gardening and jazz. Wedding in 1975 in Madrid.",
  "tags": ["interests", "life_event"],
  "sensitivity": "normal"
}
r = requests.post("http://localhost:8002/v1/elr/ingest_text", json=payload, headers={"Authorization":"Bearer devtoken"})
print(r.json())
~~~

### Example: Vector search  
~~~python
q = {
  "user_id": "user_123",
  "query": "music she enjoys",
  "k": 3,
  "filters": {"tags": ["interests"]}
}
r = requests.post("http://localhost:8002/v1/elr/search", json=q, headers={"Authorization":"Bearer devtoken"})
for hit in r.json()["results"]:
    print(hit["score"], hit["text"])
~~~

### Example: KV set/get  
~~~python
# set favorite_artist
requests.post("http://localhost:8002/v1/kv/set", json={
  "user_id": "user_123",
  "key": "favorite_artist",
  "value": "Miles Davis"
}, headers={"Authorization":"Bearer devtoken"})

# get favorite_artist
g = requests.get("http://localhost:8002/v1/kv/get", params={
  "user_id": "user_123",
  "key": "favorite_artist"
}, headers={"Authorization":"Bearer devtoken"})
print(g.json()["value"])
~~~

---

## 6. Access Control & Consent  
- All endpoints require a service token (checked in `auth/rbac.py`).  
- Each ELR item carries a `consent_scope` & `sensitivity`; queries must specify role/need.  
- Redaction runs **before** embedding to avoid leaking PII into vectors.  
- Audit every write/delete; keep version hash for integrity checks.

---

## 7. Backups & Migration  
- Nightly snapshot of Postgres & vector store indexes to encrypted S3 bucket.  
- `migrations/` contains Alembic scripts—never change old migrations, add new ones.  
- Provide `export_user_memory(user_id)` for GDPR export/delete flows.

---

## 8. Monitoring & Metrics  
- Prometheus endpoint `/metrics` for query counts, latency, failures.  
- Log trace IDs to correlate with agent requests.  
- Alerts on ingestion failures, high latency, index corruption.

---

## 9. Roadmap  
- Hybrid retrieval (BM25 + vector fusion)  
- Multimedia embeddings (audio/image snippets)  
- Federated ingestion mode (Flower / PySyft integration)  
- Differential privacy noise for cohort analytics  
- TTL policies for session memory & stale KV entries

---

## 10. Contributing (Internal Only)  
- Branch naming: `feat/memory-...`, `fix/ingest-...`  
- Add unit tests for new storage adapters or consent logic.  
- Never commit real ELR data; use `tests/fixtures/demo_elr.json`.  
- PR requires 1 reviewer + green CI.

---

## 11. License  
**Proprietary – All Rights Reserved**  
Copyright © 2025 Singularities Ltd / ReMeLife.  
Unauthorized copying, modification, distribution, or disclosure is strictly prohibited.

---

**Memory with meaning. Privacy with power.**
