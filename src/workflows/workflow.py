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
        try:
            log.info("create_qna_dataset_workflow started", input=input)
            
            prompt = input["prompt"]
            topic = input["topic"]
            filename = input["filename"]
            content = input["content"]
            log.info(f"Processing section in create_qna_dataset_workflow") 
            qna_json = await workflow.step(
                llm_qna_generator,
                LLMQnAFunctionInputParams(topic=topic, section=content.strip()),
                start_to_close_timeout=timedelta(seconds=300)
            )
        
        
            # Parse the JSON response and add to global list
            # log.info(f"qna_json before parsing: {qna_json}")
            qna_data = json.loads(qna_json)
            log.info(f"qna_data type: {type(qna_data)}")
            log.info(f"qna_json after parsing: {qna_data}")
            
            # if isinstance(qna_data, list):
            #     pass
                # log.info("Successfully processed section", 
                #         filename=filename,
                #         qna_pairs_added=len(qna_data))
            # log.info("qna_data", qna_data=qna_data)
            return qna_data
        except json.JSONDecodeError as e:
            log.error(f"Error parsing JSON", 
                    filename=filename,
                    error=str(e))
        # Global list to store all Q&A pairs
        # global_qna_list = []
        
        # # Process each file's content
        # for filename, content in zip(files, contents):
        #     log.info(f"Processing file", filename=filename)
            
        #     # Split content into sections
        #     sections = content.split('---|---')
            
        #     # First line of first section contains the topic
        #     topic = sections[0].strip().replace('TOPIC: ', '')
        #     log.info(f"Processing topic", topic=topic, section_count=len(sections)-1)
            
        #     # Process each section
        #     for section_index, section in enumerate(sections[1:], 1):
        #         if section.strip():  # Skip empty sections
        #             log.info(f"Processing section", 
        #                     filename=filename,
        #                     section_number=f"{section_index}/{len(sections)-1}")
                    
        #             # Generate Q&A for this section using workflow.step
        #             qna_json = await workflow.step(
        #                 llm_qna_generator,
        #                 LLMQnAFunctionInputParams(topic=topic, section=section.strip()),
        #                 start_to_close_timeout=timedelta(seconds=300)
        #             )
                    
        #             try:
        #                 # Parse the JSON response and add to global list
        #                 qna_data = json.loads(qna_json)
        #                 if isinstance(qna_data, list):
        #                     global_qna_list.extend(qna_data)
        #                     log.info("Successfully processed section", 
        #                             filename=filename,
        #                             section_number=section_index,
        #                             qna_pairs_added=len(qna_data))
        #             except json.JSONDecodeError as e:
        #                 log.error(f"Error parsing JSON", 
        #                         filename=filename,
        #                         section_number=section_index,
        #                         error=str(e))
        #                 continue
        
        # log.info("create_qna_dataset_workflow completed", 
        #         total_files_processed=len(files),
        #         total_qna_pairs=len(global_qna_list))
        
        # return global_qna_list