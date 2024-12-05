
# Script 2: retriever.py
import pinecone
from sentence_transformers import SentenceTransformer

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
from retriever import DocumentRetriever

class ResponseGenerator:
    def __init__(self):
        self.retriever = DocumentRetriever()
    
    def generate_response(self, user_query):
        # Get relevant context
        context_chunks = self.retriever.get_relevant_chunks(user_query)
        
        # Prepare prompt
        prompt = f"""Context information is below:
        
{' '.join(context_chunks)}

Given the context information and no other information, answer the following query:
{user_query}
"""
        
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
    response = generator.generate_response("Your question here")
    print(response)