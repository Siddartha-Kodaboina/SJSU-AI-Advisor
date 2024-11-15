from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dataclasses import dataclass
import time
from restack_ai import Restack
import uvicorn
import os
from restack_ai.function import function, log, FunctionFailure, log
import json

# Define request model
@dataclass
class PromptRequest:
    prompt: str

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def home():
    return "Welcome to the TogetherAI LlamaIndex FastAPI App!"

@app.post("/api/schedule")
async def schedule_workflow(request: PromptRequest):
    try:
        client = Restack()
        workflow_id = f"{int(time.time() * 1000)}-llm_complete_workflow"
        
        runId = await client.schedule_workflow(
            workflow_name="llm_complete_workflow",
            workflow_id=workflow_id,
            input={"prompt": request.prompt}
        )
        print("Scheduled workflow", runId)
        
        result = await client.get_workflow_result(
            workflow_id=workflow_id,
            run_id=runId
        )
        
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/data/extract")
async def extract_dataset(request: PromptRequest):
    try:
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urls_file_path = os.path.join(current_dir, 'urls.txt')
        
        with open(urls_file_path, 'r') as file:
            urls = file.read().splitlines()
        urls_str = '*&^'.join(urls)
        
        client = Restack()
        workflow_id = f"{int(time.time() * 1000)}-dataset_extraction_workflow"
        
        # Schedule the dataset generation workflow
        runId = await client.schedule_workflow(
            workflow_name="dataset_extraction_workflow",
            workflow_id=workflow_id,
            input={"prompt": urls_str}
        )
        print("Scheduled dataset_extraction_workflow", runId)
        
        # Get the workflow result
        result = await client.get_workflow_result(
            workflow_id=workflow_id,
            run_id=runId
        )
        
        # Check if the result indicates success
        if "Successfully processed" in str(result):
            return {
                "status": "success",
                "message": "Data files were successfully created",
                "result": result
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Data extraction failed or returned unexpected result"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to generate dataset: {str(e)}"
        )

@app.post("/api/data/create")
async def create_qna_dataset():
    # try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_dir = os.path.join(current_dir, 'processed_data2')
    
    # Check if directory exists
    if not os.path.exists(processed_data_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Directory not found: {processed_data_dir}"
        )
    
    # Get list of txt files
    files = [f for f in os.listdir(processed_data_dir) if f.endswith('.txt')]
    if not files:
        raise HTTPException(
            status_code=404,
            detail="No .txt files found in the directory"
        )
    
    # Initialize global list to store Q&A pairs
    global_qna_list = []
    for filename in files:
        file_path = os.path.join(processed_data_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            log.error(f"Error reading file {filename}", error=str(e))
            continue
        
        log.info(f"Processing file", filename=filename)
            
        # Split content into sections
        sections = content.split('---|---')
        
        topic = sections[0].strip().replace('TOPIC: ', '')
        # log.info(f"Processing topic", topic=topic, section_count=len(sections)-1)
        
        #Process each section
        for section_index, section in enumerate(sections[1:], 1):
            if section.strip():  # Skip empty sections
                log.info(f"Processing section", 
                        filename=filename,
                        section_number=f"{section_index}/{len(sections)-1}")
                
                # Generate Q&A for this section using workflow.step
                client = Restack()
                workflow_id = f"{int(time.time() * 1000)}-create_qna_dataset_workflow"

                # Start workflow run with prepared data
                runId = await client.schedule_workflow(
                    workflow_name="create_qna_dataset_workflow",
                    workflow_id=workflow_id,
                    input={
                        "prompt": "SJSU Engineering Department",
                        "topic": topic,
                        "filename": filename,  
                        "content": content
                    }
                )
                
                # log.info(f"Processing section", 
                #         filename=filename,
                #         section_number=f"{section_index}/{len(sections)-1}")
                
                # Get the workflow result
                qna_data = await client.get_workflow_result(
                    workflow_id=workflow_id,
                    run_id=runId
                )
                
                try:
                    # Parse the JSON response and add to global list
                    # qna_data = json.loads(qna_json)
                    log.info(f"qna_data type: {type(qna_data)}")
                    if isinstance(qna_data, list):
                        global_qna_list.extend(qna_data)
                        # log.info("Successfully processed section", 
                        #         filename=filename,
                        #         section_number=section_index,
                        #         qna_pairs_added=len(qna_data))
                    log.info(f"global_qna_list: {global_qna_list}")
                except json.JSONDecodeError as e:
                    log.error(f"Error parsing JSON", 
                            filename=filename,
                            section_number=section_index,
                            error=str(e))
                    continue
    
    # Create Restack client
    # client = Restack()
    # workflow_id = f"{int(time.time() * 1000)}-create_qna_dataset_workflow"

    # # Start workflow run with prepared data
    # runId = await client.schedule_workflow(
    #     workflow_name="create_qna_dataset_workflow",
    #     workflow_id=workflow_id,
    #     input={
    #         "prompt": "SJSU Engineering Department",  # or whatever prompt you want to use
    #         "files": files,  # List of filenames
    #         "contents": content_of_files  # List of file contents
    #     }
    # )
    
    # log.info("Scheduled create_qna_dataset_workflow", 
    #         run_id=runId, 
    #         file_count=len(files))
    
    # # Get the workflow result
    # result = await client.get_workflow_result(
    #     workflow_id=workflow_id,
    #     run_id=runId
    # )
    
    return {
        "message": "QnA dataset creation completed",
        "files_processed": len(files),
        "result": global_qna_list
    }
        
    # except Exception as e:
    #     log.error("Error in create_qna_dataset endpoint", error=str(e))
    #     raise HTTPException(status_code=500, detail=str(e))

# Remove Flask-specific run code since FastAPI uses uvicorn
def run_app():
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == '__main__':
    run_app()
