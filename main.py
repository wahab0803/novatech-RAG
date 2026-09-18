import os
import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_PATH = os.path.join(BASE_DIR, "documents")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")


# --------------------------------------------------
# 2. LOAD EMBEDDING MODEL
# --------------------------------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# 3. LOAD DOCUMENTS
# --------------------------------------------------

def load_documents():

    documents = []

    for filename in os.listdir(DOCUMENTS_PATH):

        file_path = os.path.join(DOCUMENTS_PATH, filename)

        if filename.endswith(".txt"):

            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            documents.append({
                "filename": filename,
                "content": content
            })

    return documents


# --------------------------------------------------
# 4. SPLIT DOCUMENTS INTO CHUNKS
# --------------------------------------------------

def chunk_text(text, chunk_size=500, overlap=100):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# --------------------------------------------------
# 5. BUILD VECTOR DATABASE
# --------------------------------------------------

def build_vector_database():

    documents = load_documents()

    print(f"Loaded {len(documents)} documents\n")

    all_chunks = []

    for document in documents:

        chunks = chunk_text(document["content"])

        for chunk in chunks:

            all_chunks.append({
                "filename": document["filename"],
                "content": chunk
            })

    print(f"Created {len(all_chunks)} chunks\n")

    texts = [chunk["content"] for chunk in all_chunks]

    embeddings = embedding_model.encode(texts)

    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimensions: {len(embeddings[0])}\n")

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = chroma_client.get_or_create_collection(
        name="novatech_knowledge"
    )

    ids = []
    documents_for_db = []
    metadatas = []

    for i, chunk in enumerate(all_chunks):

        ids.append(f"chunk_{i}")

        documents_for_db.append(chunk["content"])

        metadatas.append({
            "source": chunk["filename"]
        })

    collection.upsert(
        ids=ids,
        documents=documents_for_db,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print(f"Stored {len(all_chunks)} chunks in ChromaDB")
    print(f"Documents in ChromaDB: {collection.count()}\n")

    return collection


# --------------------------------------------------
# 6. RETRIEVE RELEVANT CHUNKS
# --------------------------------------------------

def retrieve_documents(collection, query, number_of_results=3):

    query_embedding = embedding_model.encode([query])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=number_of_results
    )

    return results


# --------------------------------------------------
# 7. MAIN PROGRAM
# --------------------------------------------------

collection = build_vector_database()


query = input("Ask a question about NovaTech Solutions: ")

results = retrieve_documents(
    collection,
    query,
    number_of_results=3
)


# --------------------------------------------------
# 8. DISPLAY RETRIEVED RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("RETRIEVED INFORMATION")
print("=" * 60)

for i in range(len(results["documents"][0])):

    print(f"\nResult {i + 1}")

    print("-" * 60)

    print(f"Source: {results['metadatas'][0][i]}")

    print()

    print(results["documents"][0][i])