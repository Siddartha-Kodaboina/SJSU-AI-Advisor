from datetime import timedelta
from restack_ai.workflow import workflow, import_functions, log
import os
import json
with import_functions():
    from src.functions.function import llm_complete, data_extraction, qna_generator, FunctionInputParams, llm_qna_generator, LLMQnAFunctionInputParams

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
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # parent_dir = os.path.dirname(current_dir)
        processed_data_dir = input["processed_data_dir"]
        
        # Global list to store all Q&A pairs
        global_qna_list = []
        
        # Loop through each file in processed_data directory
        for filename in os.listdir(processed_data_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(processed_data_dir, filename)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split content into sections
                sections = content.split('---|---')
                
                # First line of first section contains the topic
                topic = sections[0].strip().replace('TOPIC: ', '')
                log.info(f"topic: {topic}")
                log.info(f"sections: {sections}")
                log.info(f"length of sections: {len(sections)}")
                # Process each section
                for ind, section in enumerate(sections[1:], 1):
                    log.info(f"ind: {ind} processing")# Skip the first section as it's the topic
                    if section.strip():  # Skip empty sections
                        # Generate Q&A for this section
                        qna_json = await workflow.step(llm_qna_generator, LLMQnAFunctionInputParams(topic=prompt, section=section.strip()), start_to_close_timeout=timedelta(seconds=60))
                        # qna_json = await llm_qna_generator(topic, section.strip())
                        log.info(f"-------------filename: {filename}-------------------")
                        log.info(f"qna_data: {qna_json}")
                        log.info(f"-------------filename: {filename}-------------------")
                        try:
                            # Parse the JSON response and add to global list
                            qna_data = json.loads(qna_json)
                            
                            if isinstance(qna_data, list):
                                global_qna_list.extend(qna_data)
                        except json.JSONDecodeError as e:
                            log.error(f"Error parsing JSON for {filename}", error=e)
                            continue
        
        log.info(f"Generated {len(global_qna_list)} Q&A pairs")
        
        log.info("create_qna_dataset_workflow completed", result=global_qna_list)
        return global_qna_list