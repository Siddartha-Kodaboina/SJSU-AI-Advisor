import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("INDEX_NAME")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Get the index
index = pc.Index(INDEX_NAME)

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = embeddings.embed_query(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [result['metadata']['text'] for result in results['matches']]