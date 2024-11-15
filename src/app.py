from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dataclasses import dataclass
import time
from restack_ai import Restack
import uvicorn
import os

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
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        processed_data_dir = os.path.join(current_dir, 'processed_data2')
        
        # Create Restack client
        client = Restack()
        workflow_id = f"{int(time.time() * 1000)}-create_qna_dataset_workflow"

        # Start workflow run
        runId = await client.schedule_workflow(
            workflow_name="create_qna_dataset_workflow",
            workflow_id=workflow_id,
            input={"prompt": "something", "processed_data_dir": processed_data_dir}
        )
        
        print("Scheduled create_qna_dataset_workflow", runId)
        
        # Get the workflow result
        result = await client.get_workflow_result(
            workflow_id=workflow_id,
            run_id=runId
        )
        # Return response
        return {"message": "QnA dataset creation started", "run_id": runId}
        
    except Exception as e:
        print("Error in create_qna_dataset endpoint", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Remove Flask-specific run code since FastAPI uses uvicorn
def run_app():
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == '__main__':
    run_app()
