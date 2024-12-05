import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
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
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Function to read and chunk text files
def process_files(directory):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as file:
                text = file.read()
                chunks = text_splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    embedding = embeddings.embed_query(chunk)
                    index.upsert(vectors=[(f"{filename}_{i}", embedding, {"text": chunk})])

# Process files
process_files("/Users/sid/Desktop/Personel/SJSU/Semester III/CMPE 259- NLP/Project/streamlit_fastapi_togetherai_llama/src/processed_data")