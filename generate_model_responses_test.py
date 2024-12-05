import csv
import subprocess
import json

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

question = "What is the MS in Software Engineering - Enterprise Software Technologies program about?"
training_output = "The MS in Software Engineering - Enterprise Software Technologies program prepares students to be technical leaders in software development, focusing on distributed N-Tier Client/Server architectures and the latest technologies and trends in Enterprise software development."

print("Before : ", question, training_output)
# Get responses from both models
# llama_3_2_output = get_model_response('llama3.2', question)
# print("After : ", question, training_output, llama_3_2_output, flush=True)

llama_32_1b_advisor_output = get_model_response('llama_32_1b_advisor', question)
print("After : ", question, training_output, llama_32_1b_advisor_output, flush=True)