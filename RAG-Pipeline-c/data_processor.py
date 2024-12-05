# Script 1: data_processor.py
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pinecone
from dotenv import load_dotenv
load_dotenv()
# api_key = os.getenv("HF_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("INDEX_NAME")

def init_pinecone():
    print("IN init_pinecone", flush=True)
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(INDEX_NAME, dimension=384)  # 384 for all-MiniLM-L6-v2
    return pinecone.Index(INDEX_NAME)

def process_files(directory_path):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    documents = []
    for filename in os.listdir(directory_path):
        print("Start Processing: ", filename, flush=True)
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), 'r') as file:
                text = file.read()
                chunks = text_splitter.split_text(text)
                documents.extend([(chunk, filename) for chunk in chunks])
        print("Done Processing: ", filename, flush=True)
    
    return documents

def create_embeddings(documents):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = []
    for doc, filename in documents:
        print("Before embeddings : ", filename, flush=True)
        embedding = model.encode(doc)
        embeddings.append({
            'id': str(hash(doc)),
            'values': embedding.tolist(),
            'metadata': {
                'text': doc,
                'source': filename
            }
        })
        print("After embeddings : ", filename, flush=True)
    return embeddings

def main():
    directory_path = "/Users/sid/Desktop/Personel/SJSU/Semester III/CMPE 259- NLP/Project/streamlit_fastapi_togetherai_llama/src/processed_data2"
    
    # Initialize Pinecone
    index = init_pinecone()
    print("pine cone init is successful!", flush=True)
    
    # Process documents
    documents = process_files(directory_path)
    print("documents chunking is successful!", flush=True)
    
    # Create and store embeddings
    embeddings = create_embeddings(documents)
    print("embeddings generation is successful!", flush=True)
    
    # Upload to Pinecone in batches
    batch_size = 100
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i + batch_size]
        print(i, "upsert before")
        index.upsert(vectors=batch)
        print(i, "upsert after")
    print("embeddings upserting is successful!", flush=True)
        

if __name__ == "__main__":
    main()
