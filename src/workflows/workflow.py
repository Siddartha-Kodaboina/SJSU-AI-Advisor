from datetime import timedelta
from restack_ai.workflow import workflow, import_functions, log

with import_functions():
    from src.functions.function import llm_complete, FunctionInputParams

@workflow.defn(name="llm_complete_workflow")
class llm_complete_workflow:
    @workflow.run
    async def run(self, input: dict):
        log.info("llm_complete_workflow started", input=input)
        prompt = input["prompt"]
        result = await workflow.step(llm_complete, FunctionInputParams(prompt=prompt), start_to_close_timeout=timedelta(seconds=120))
        log.info("llm_complete_workflow completed", result=result)
        return result

@workflow.defn(name="dataset_generation")
class dataset_generation:
    @workflow.run
    async def run(self, input: dict):
        log.info("dataset_generation started", input=input)
        URLS = []
        urls_str = '*&^'.join(URLS)
        data_extraction = await workflow.step(data_extraction, FunctionInputParams(prompt=urls_str), start_to_close_timeout=timedelta(seconds=1200))