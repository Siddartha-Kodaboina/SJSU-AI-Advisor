from llama_index.llms.together import TogetherLLM
from restack_ai.function import function, log, FunctionFailure, log
from llama_index.core.llms import ChatMessage, MessageRole
import os
from dataclasses import dataclass
from dotenv import load_dotenv
from llama_index.readers.web import SimpleWebPageReader
import json

load_dotenv()

@dataclass
class FunctionInputParams:
    prompt: str

@dataclass
class LLMQnAFunctionInputParams:
    topic: str
    section: str


@function.defn(name="llm_complete")
async def llm_complete(input: FunctionInputParams):
    try:
        log.info("llm_complete function started", input=input)
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            log.error("TOGETHER_API_KEY environment variable is not set.")
            raise ValueError("TOGETHER_API_KEY environment variable is required.")
    
        llm = TogetherLLM(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo", api_key=api_key
        )
        messages = [
            ChatMessage(
                # This is a system prompt that is used to set the behavior of the LLM. You can update this llm_complete function to also accept a system prompt as an input parameter.
                role=MessageRole.SYSTEM, content="You are a pirate with a colorful personality"
            ),
            ChatMessage(role=MessageRole.USER, content=input.prompt),
        ]
        resp = llm.chat(messages)
        log.info("llm_complete function completed", response=resp.message.content)
        return resp.message.content
    except Exception as e:
        log.error("llm_complete function failed", error=e)
        raise e

@function.defn(name="data_extraction")
async def data_extraction(input: FunctionInputParams):
    try:
        log.info("data_extraction function started", input=input)
        URLS = input.prompt.split('*&^')
        # Get the parent directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)  # Get parent directory
        data_dir_path = os.path.join(parent_dir, 'data')  # Create data path in parent directory
        log.info(f"data_dir_path: {data_dir_path}")
        # print(data_dir_path)
        # return f"Successfully processed {len(URLS)} URLs"
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir_path, exist_ok=True)
        
        # Load and save documents
        for idx, url in enumerate(URLS, 1):
            if not url.strip():  # Skip empty URLs
                continue
                
            # Load document from URL
            documents = SimpleWebPageReader(html_to_text=True).load_data([url])
            
            # Save to file
            output_path = os.path.join(data_dir_path, f"text{idx}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                for doc in documents:
                    f.write(doc.text)
            
            log.info(f"Saved content from {url} to {output_path}")
        
        return f"Successfully processed {len(URLS)} URLs"
        
    except Exception as e:
        log.error("data_extraction function failed", error=e)
        raise e
    
@function.defn(name="llm_qna_generator")
async def llm_qna_generator(input: LLMQnAFunctionInputParams):
    try:
        topic = input.topic
        section_content = input.section
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            log.error("TOGETHER_API_KEY environment variable is not set.")
            raise ValueError("TOGETHER_API_KEY environment variable is required.")
        
        llm = TogetherLLM(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo", 
            api_key=api_key
        )
        
        prompt = f"""As a SJSU Computer Enginering department advisor, review this documentation section and generate comprehensive specific Questions and Answers pairs that students might ask with curated answers providing all the details and links, instead of jsut mentioning where to find more information.
        
        Topic: {topic}
        Content: {section_content}
        
        Generate a JSON response containing questions and answers covering all aspects (what, when, where, why, how, who, time, deadlines, status, payments, subjects, subject code, admissions, enrollment restrictions, can AI students take SE course wiseversa etc.).
        Format the response as a list of maps with 'question' and 'answer' keys.
        Only provide the JSON response, nothing else.
        There is a limit to the number questions per each section, Ensure questions are unique and non-repetitive and cover all the topics of the sections.
        """
        prompt += """
        Example Output Format and Structure of Json File(Only provide a list of json objects with two keys question and answer):
        <json>
        [
            {"question": "What is the deadline for applying to SJSU?", "answer": "The deadline for applying to SJSU is 17th of October."},
            {"question": "What is the deadline for applying to SJSU?", "answer": "The deadline for applying to SJSU is 17th of October."},
            {"question": "What is the deadline for applying to SJSU?", "answer": "The deadline for applying to SJSU is 17th of October."},
            {"question": "What is the deadline for applying to SJSU?", "answer": "The deadline for applying to SJSU is 17th of October."}
        ]
        </json>
        """
        log.info(f"prompt: {prompt}")
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="You are a SJSU department advisor who reviews documentation and prepares helpful Q&A for students."
            ),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]
        log.info(f"messages: {messages}")
        resp = llm.chat(messages)
        content = resp.message.content
        # content = '''[
        #     {"question": "What is the deadline for applying to SJSU?", "answer": "The deadline for applying to SJSU is 17th of October."},
        #     {"question": "What is the deadline for applying to SJSU?", "answer": "The deadline for applying to SJSU is 17th of October."},
        #     {"question": "What is the deadline for applying to SJSU?", "answer": "The deadline for applying to SJSU is 17th of October."},
        #     {"question": "What is the deadline for applying to SJSU?", "answer": "The deadline for applying to SJSU is 17th of October."}
        # ]'''
        
        # Parse the JSON string into a Python object before returning
        try:
            # qna_data = json.loads(content)
            # if not isinstance(qna_data, list):
            #     raise ValueError("Expected JSON array response")
            return content  # Return the parsed list of Q&A pairs
        except json.JSONDecodeError as e:
            log.error("Failed to parse LLM response as JSON", error=str(e))
            raise ValueError(f"Invalid JSON response from LLM: {str(e)}")
            
    except Exception as e:
        log.error("llm_qna_generator function failed", error=str(e))
        raise e
    
@function.defn(name="qna_generator")
async def qna_generator(input: FunctionInputParams):
    try:
        pass
        return []
        # log.info("qna_generator function started", input=input)
        
        # # Get the parent directory path
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # parent_dir = os.path.dirname(current_dir)
        # processed_data_dir = os.path.join(parent_dir, 'processed_data2')
        
        # # Global list to store all Q&A pairs
        # global_qna_list = []
        
        # # Loop through each file in processed_data directory
        # for filename in os.listdir(processed_data_dir):
        #     if filename.endswith('.txt'):
        #         file_path = os.path.join(processed_data_dir, filename)
                
        #         with open(file_path, 'r', encoding='utf-8') as f:
        #             content = f.read()
                
        #         # Split content into sections
        #         sections = content.split('---|---')
                
        #         # First line of first section contains the topic
        #         topic = sections[0].strip().replace('TOPIC: ', '')
        #         log.info(f"topic: {topic}")
        #         log.info(f"sections: {sections}")
        #         log.info(f"length of sections: {len(sections)}")
        #         # Process each section
        #         for ind, section in enumerate(sections[1:], 1):
        #             log.info(f"ind: {ind} processing")# Skip the first section as it's the topic
        #             if section.strip():  # Skip empty sections
        #                 # Generate Q&A for this section
        #                 qna_json = await llm_qna_generator(topic, section.strip())
        #                 log.info(f"-------------filename: {filename}-------------------")
        #                 log.info(f"qna_data: {qna_json}")
        #                 log.info(f"-------------filename: {filename}-------------------")
        #                 try:
        #                     # Parse the JSON response and add to global list
        #                     qna_data = json.loads(qna_json)
                            
        #                     if isinstance(qna_data, list):
        #                         global_qna_list.extend(qna_data)
        #                 except json.JSONDecodeError as e:
        #                     log.error(f"Error parsing JSON for {filename}", error=e)
        #                     continue
        
        # log.info(f"Generated {len(global_qna_list)} Q&A pairs")
        # return global_qna_list
        
    except Exception as e:
        log.error("qna_generator function failed", error=e)
        raise e
    
  