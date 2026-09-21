import hashlib, os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

os.makedirs("data/chroma_db", exist_ok=True)
model = SentenceTransformer("all-MiniLM-L6-v2")
collection = chromadb.PersistentClient(path="data/chroma_db").get_or_create_collection(
    name="hr_policies", metadata={"hnsw:space": "cosine"}
)

#For duplicate files
def compute_file_hash(file_bytes): 
    return hashlib.sha256(file_bytes).hexdigest()


def process_uploaded_pdf(file_path, filename, file_bytes):
    file_hash = compute_file_hash(file_bytes)
    pages = PyPDFLoader(file_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts, metadatas, ids = [], [], []

    for page in pages:
        for i, chunk in enumerate(splitter.split_text(page.page_content)):
            if len(chunk.strip()) < 20:
                continue
            texts.append(chunk)
            metadatas.append({"source": filename, "page": page.metadata.get("page", 0) + 1})
            ids.append(f"{file_hash}_{len(ids)}")

    if texts:
        embeddings = model.encode(texts).tolist()
        collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    return len(texts), file_hash

#If old policy file is deleted, delete chunks related to them
def delete_document_vectors(filename):
    results = collection.get(where={"source": filename})
    if results["ids"]:
        collection.delete(ids=results["ids"])


def retrieve_relevant_chunks(query, top_k=5):
    if collection.count() == 0:
        return []
    embedding = model.encode([query]).tolist()
    r = collection.query(query_embeddings=embedding, n_results=min(top_k, collection.count()))
    return [
        {"text": r["documents"][0][i], "source": r["metadatas"][0][i]["source"],
         "page": r["metadatas"][0][i]["page"], "score": round(1 - r["distances"][0][i], 4)}
        for i in range(len(r["ids"][0]))
    ]

def collection_count():
    return collection.count()