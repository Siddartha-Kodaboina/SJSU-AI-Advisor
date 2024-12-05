from langchain_ollama import OllamaLLM
from retriever import retrieve_relevant_chunks
import json
import subprocess
import csv

# Initialize Ollama LLM
llama = OllamaLLM(model="llama3.2")
finetuned_llama = OllamaLLM(model="llama_32_1b_advisor")

def generate_model_response_with_rag(llm, user_query):
    relevant_chunks = retrieve_relevant_chunks(user_query)
    context = "\n".join(relevant_chunks)

    prompt = f"""Context: {context}

Student Query: {user_query}

please provide a response based on the given context and to a Student Query as if it is texted by the SJSU Computer Engineering Advisor :"""

    response = llm.invoke(prompt)
    return response

def get_model_response(model_name, question):
    result = subprocess.run(
        ['ollama', 'run', model_name, question],
        capture_output=True,
        text=True
    )
    response = result.stdout.strip()
    return response

# Example usage
# user_input = "What is the maximum number of credits allowed per semester in the MSSE program?"
# response = generate_response(user_input)
# print(response)

def generate_responses_from_questions(input_file, output_file):
    # Load questions from the JSON file
    with open(input_file, 'r', encoding='utf-8') as json_file:
        questions = json.load(json_file)

    # Prepare the output data
    output_data = []

    # Iterate through each question
    for item in questions:
        question = item['question']
        print("Question: ", question)
        # Generate responses using both models
        llama_3_2_response = get_model_response('llama3.2', question)
        print("llama_3_2_response: ", llama_3_2_response)
        llama_32_1b_advisor_response = get_model_response('llama_32_1b_advisor', question)
        print("llama_32_1b_advisor_response: ", llama_32_1b_advisor_response)
        
        # Generate responses using RAG approach
        rag_llama_3_2_response = generate_model_response_with_rag(llama, question)  # Adjust as needed for RAG
        print("rag_llama_3_2_response: ", rag_llama_3_2_response)
        rag_llama_32_1b_advisor_response = generate_model_response_with_rag(finetuned_llama, question)  # Adjust as needed for RAG
        print("rag_llama_3_2_response: ", rag_llama_3_2_response)
        # Append the results to the output data
        output_data.append({
            'question': question,
            'llama_3_2_response': llama_3_2_response,
            'llama_32_1b_advisor_response': llama_32_1b_advisor_response,
            'rag_llama_3_2_response': rag_llama_3_2_response,
            'rag_llama_32_1b_advisor_response': rag_llama_32_1b_advisor_response
        })

    # Write the output data to a new CSV file
    with open(output_file, mode='w', encoding='utf-8', newline='') as csv_file:
        fieldnames = ['question', 'llama_3_2_response', 'llama_32_1b_advisor_response', 'rag_llama_3_2_response', 'rag_llama_32_1b_advisor_response']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        writer.writeheader()
        for data in output_data:
            writer.writerow(data)

    print(f"Responses have been successfully saved to {output_file}.")

if __name__ == "__main__":
    input_file = '/Users/sid/Desktop/Personel/SJSU/Semester III/CMPE 259- NLP/Project/streamlit_fastapi_togetherai_llama/test_questions.json'  # Path to your test questions JSON file
    output_file = '/Users/sid/Desktop/Personel/SJSU/Semester III/CMPE 259- NLP/Project/streamlit_fastapi_togetherai_llama/model_responses_from_questions.csv'  # Output CSV file
    generate_responses_from_questions(input_file, output_file)