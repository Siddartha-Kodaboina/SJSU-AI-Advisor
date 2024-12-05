
# Script 2: retriever.py
import pinecone
import os
from sentence_transformers import SentenceTransformer

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("INDEX_NAME")

class DocumentRetriever:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        self.index = pinecone.Index(INDEX_NAME)
    
    def get_relevant_chunks(self, query, top_k=3):
        query_embedding = self.model.encode(query)
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        
        return [match.metadata['text'] for match in results.matches]

# Script 3: generator.py
import ollama
# from retriever import DocumentRetriever

class ResponseGenerator:
    def __init__(self):
        self.retriever = DocumentRetriever()
    
    def generate_response(self, user_query):
        # Get relevant context
        context_chunks = self.retriever.get_relevant_chunks(user_query)
        print(context_chunks)
        # Prepare prompt
        prompt = f"""Context information is below:
        
{' '.join(context_chunks)}

Given the context information and no other information, answer the following query:
{user_query}
"""
        print("The promt is here: ", prompt)
        
        # Generate response using Llama
        response = ollama.generate(
            model='llama2:3.2', 
            prompt=prompt,
            temperature=0.7,
            max_tokens=500
        )
        
        return response['response']

# Usage example
if __name__ == "__main__":
    generator = ResponseGenerator()
    response = generator.generate_response("What are the required specialization core classes for the MS in Software Engineering - Enterprise Software Technologies program?")
    print(response)