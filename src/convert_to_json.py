import csv
import json

# Define the input and output file paths
input_file = '/Users/sid/Desktop/Personel/SJSU/Semester III/CMPE 259- NLP/Project/streamlit_fastapi_togetherai_llama/qna_dataset.csv'
output_file = '/Users/sid/Desktop/Personel/SJSU/Semester III/CMPE 259- NLP/Project/streamlit_fastapi_togetherai_llama/qna_dataset.json'

# Initialize a list to hold the JSON data
data = []

# Read the CSV file
with open(input_file, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    
    # Skip the first record
    next(csv_reader)
    
    # Convert each row to the desired JSON format
    for row in csv_reader:
        data.append({'question': row['question'], 'answer': row['answer']})

# Write the JSON data to a file
with open(output_file, mode='w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

print(f"Data has been successfully converted to JSON and saved to {output_file}.")