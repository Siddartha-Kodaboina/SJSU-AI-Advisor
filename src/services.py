import asyncio
from src.client import client
from src.functions.function import llm_complete, data_extraction, llm_qna_generator, qna_generator
from src.workflows.workflow import llm_complete_workflow, dataset_extraction_workflow, create_qna_dataset_workflow

async def main():
    await client.start_service(
        workflows= [llm_complete_workflow, dataset_extraction_workflow, create_qna_dataset_workflow],
        functions= [llm_complete, data_extraction, qna_generator, llm_qna_generator]
    )

def run_services():
    asyncio.run(main())

if __name__ == "__main__":
    run_services()
