import os, glob, hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

POLICIES_DIR = "data"
os.makedirs("data/chroma_db", exist_ok=True)

_model = SentenceTransformer("all-MiniLM-L6-v2")
_collection = chromadb.PersistentClient(path="data/chroma_db").get_or_create_collection(
    name="hr_policies", metadata={"hnsw:space": "cosine"}
)

def load_and_index_policies():
    pdf_files = glob.glob(os.path.join(POLICIES_DIR, "*.pdf"))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    indexed_summary = []

    for path in pdf_files:
        filename = os.path.basename(path)
        with open(path, "rb") as f:
            f_hash = hashlib.sha256(f.read()).hexdigest()

        pages = PyPDFLoader(path).load()
        texts, metadatas, ids = [], [], []

        for page in pages:
            page_num = page.metadata.get("page", 0) + 1
            chunks = splitter.split_text(page.page_content)
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 20: 
                    continue
                texts.append(chunk)
                metadatas.append({"source": filename, "page": page_num})
                ids.append(f"{f_hash}_p{page_num}_{i}")

        if texts:
            embeddings = _model.encode(texts).tolist()
            _collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

        indexed_summary.append({"filename": filename, "chunks": len(texts), "hash": f_hash})

    return indexed_summary

def retrieve_relevant_chunks(query, top_k=3):
    if _collection.count() == 0: 
        return []
    embedding = _model.encode([query]).tolist()
    r = _collection.query(query_embeddings=embedding, n_results=min(top_k, _collection.count()))
    return [
        {"text": r["documents"][0][i], "source": r["metadatas"][0][i]["source"],
         "page": r["metadatas"][0][i]["page"], "score": round(1 - r["distances"][0][i], 4)}
        for i in range(len(r["ids"][0]))
    ]

def collection_count():
    return _collection.count()