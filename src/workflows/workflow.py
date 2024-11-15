from datetime import timedelta
from restack_ai.workflow import workflow, import_functions, log

with import_functions():
    from src.functions.function import llm_complete, data_extraction, qna_generator, FunctionInputParams

@workflow.defn(name="llm_complete_workflow")
class llm_complete_workflow:
    @workflow.run
    async def run(self, input: dict):
        log.info("llm_complete_workflow started", input=input)
        prompt = input["prompt"]
        result = await workflow.step(llm_complete, FunctionInputParams(prompt=prompt), start_to_close_timeout=timedelta(seconds=120))
        log.info("llm_complete_workflow completed", result=result)
        return result

@workflow.defn(name="dataset_extraction_workflow")
class dataset_extraction_workflow:
    @workflow.run
    async def run(self, input: dict):
        log.info("dataset_extraction_workflow started", input=input)
        # URLS = []
        # urls_str = '*&^'.join(URLS)
        prompt = input["prompt"]
        data_extraction_result = await workflow.step(data_extraction, FunctionInputParams(prompt=prompt), start_to_close_timeout=timedelta(seconds=1200))
        return data_extraction_result

@workflow.defn(name="create_qna_dataset_workflow")
class create_qna_dataset_workflow:
    @workflow.run
    async def run(self, input: dict):
        log.info("create_qna_dataset_workflow started", input=input)
        prompt = input["prompt"]
        qna_result = await workflow.step(qna_generator, FunctionInputParams(prompt=prompt), start_to_close_timeout=timedelta(seconds=1200))
        log.info("create_qna_dataset_workflow completed", result=qna_result)
        return qna_result