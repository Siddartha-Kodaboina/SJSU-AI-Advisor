import csv
import json
import pandas as pd

def convert_qna_format_csv():
    try:
        # Read the original QnA dataset
        qna_pairs = []
        with open('qna_dataset.csv', 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                qna_pairs.append({
                    'question': row['question'],
                    'answer': row['answer']
                })
        
        # Convert to new format
        converted_data = []
        for qna in qna_pairs:
            conversation = [
                {
                    'from': 'human',
                    'value': qna['question']
                },
                {
                    'from': 'gpt',
                    'value': qna['answer']
                }
            ]
            converted_data.append({
                'conversations': conversation,
                'source': 'Siddartha-49'
            })
        
        # Write to new CSV file with headers
        with open('conversations_dataset.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(['conversations', 'source'])
            # Write data
            for row in converted_data:
                writer.writerow([
                    json.dumps(row['conversations']),  # Keep as JSON string for CSV
                    row['source']
                ])
        print("CSV file created successfully")
        
    except Exception as e:
        print(f"Error in CSV conversion: {str(e)}")

def convert_qna_format_parquet():
    try:
        # Read the original QnA dataset
        qna_pairs = []
        with open('qna_dataset.csv', 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                qna_pairs.append({
                    'question': row['question'],
                    'answer': row['answer']
                })
        
        # Convert to new format
        converted_data = []
        for qna in qna_pairs:
            conversation = [
                {
                    'from': 'human',
                    'value': qna['question']
                },
                {
                    'from': 'gpt',
                    'value': qna['answer']
                }
            ]
            converted_data.append({
                'conversations': conversation,  # Store as native list of dicts for parquet
                'source': 'Siddartha-49'
            })
        
        # Create DataFrame and save as parquet
        df = pd.DataFrame(converted_data)
        df.to_parquet('conversations_dataset.parquet', index=False)
        print("Parquet file created successfully")
        
    except Exception as e:
        print(f"Error in Parquet conversion: {str(e)}")


if __name__ == "__main__":
    convert_qna_format_csv()    # Creates CSV with JSON string
    convert_qna_format_parquet() # Creates parquet with native Python objects 