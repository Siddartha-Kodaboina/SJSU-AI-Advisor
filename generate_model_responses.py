import csv
import subprocess
import json

# Define the input and output file paths
input_file = '/Users/sid/Desktop/Personel/SJSU/Semester III/CMPE 259- NLP/Project/streamlit_fastapi_togetherai_llama/qna_dataset.csv'
output_file = '/Users/sid/Desktop/Personel/SJSU/Semester III/CMPE 259- NLP/Project/streamlit_fastapi_togetherai_llama/model_responses.csv'

# Function to call the Ollama model and get the response
def get_model_response(model_name, question):
    # Call the Ollama model using subprocess
    result = subprocess.run(
        ['ollama', 'run', model_name, question],
        capture_output=True,
        text=True
    )
    # Parse the output
    response = result.stdout.strip()
    return response

# Read the input CSV file
with open(input_file, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    
    # Prepare the output data
    output_data = []
    
    # Process each row (skip the first row)
    for index, row in enumerate(csv_reader):
        if index>=11:
            break
        question = row['question']
        training_output = row['answer']  # Assuming the training output is the answer in the CSV
        print("Before : ", question, training_output)
        # Get responses from both models
        llama_3_2_output = get_model_response('llama3.2', question)
        llama_32_1b_advisor_output = get_model_response('llama_32_1b_advisor', question)
        print("After : ", question, training_output, llama_3_2_output, llama_32_1b_advisor_output, flush=True)
        
        # Append the results to the output data
        output_data.append({
            'input': question,
            'training_output': training_output,
            'llama_3_2_output': llama_3_2_output,
            'llama_32_1b_advisor_output': llama_32_1b_advisor_output
        })

# Write the output data to a new CSV file
with open(output_file, mode='w', encoding='utf-8', newline='') as csv_file:
    fieldnames = ['input', 'training_output', 'llama_3_2_output', 'llama_32_1b_advisor_output']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    writer.writeheader()
    for data in output_data:
        writer.writerow(data)

print(f"Responses have been successfully saved to {output_file}.")