import os

import numpy as np
import ollama


# ==================================================
# CONFIGURATION
# ==================================================

KNOWLEDGE_FOLDER = "knowledge_base"
EMBEDDING_MODEL = "nomic-embed-text"


# ==================================================
# 1. LOAD KNOWLEDGE BASE
# ==================================================

def load_documents():

    documents = []

    for filename in os.listdir(KNOWLEDGE_FOLDER):

        if filename.endswith(".txt"):

            path = os.path.join(KNOWLEDGE_FOLDER, filename)

            with open(path, "r", encoding="utf-8") as file:
                text = file.read()

            documents.append({
                "source": filename,
                "text": text
            })

    return documents


# ==================================================
# 2. CREATE DOCUMENT EMBEDDINGS
# ==================================================

def create_embeddings(documents):

    for document in documents:

        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=document["text"]
        )

        document["embedding"] = response["embeddings"][0]

    return documents


# ==================================================
# 3. COSINE SIMILARITY
# ==================================================

def cosine_similarity(vector_a, vector_b):

    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )


# ==================================================
# 4. RETRIEVE KNOWLEDGE
# ==================================================

def retrieve_knowledge(query, documents, top_k=2):

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query
    )

    query_embedding = response["embeddings"][0]

    results = []

    for document in documents:

        similarity = cosine_similarity(
            query_embedding,
            document["embedding"]
        )

        results.append({
            "source": document["source"],
            "similarity": float(similarity),
            "text": document["text"]
        })

    results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return results[:top_k]


# ==================================================
# 5. INITIALIZE KNOWLEDGE BASE
# ==================================================

documents = load_documents()
documents = create_embeddings(documents)