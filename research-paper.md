# SJSU Academic Advisor Virtual Assistant: A Fine-tuned LLM Approach for Academic Guidance

## Abstract

This paper presents the development and implementation of an AI-powered virtual assistant designed to address the academic advising needs at San Jose State University's College of Engineering. The system leverages state-of-the-art Language Learning Models (LLMs) and Retrieval-Augmented Generation (RAG) to provide immediate, accurate responses to student queries. By fine-tuning LLAMA 3 on department-specific data, we created a specialized model capable of handling complex academic inquiries while maintaining response times under one second. Our implementation demonstrates significant improvement in query response times compared to the traditional 48-hour waiting period, while maintaining high accuracy in academic guidance.

## Introduction

Academic advising plays a crucial role in student success within higher education institutions. At San Jose State University's College of Engineering, the current advisory system faces significant challenges, including:

- Extended response times exceeding 48 hours for general queries
- High advisor workload during peak periods
- Complex and diverse student inquiries requiring specialized knowledge
- Limited availability of immediate assistance for time-sensitive questions

This research presents an innovative solution utilizing advanced natural language processing techniques to create a virtual assistant specifically tailored to the needs of engineering students. Our approach combines the power of Large Language Models with domain-specific knowledge to provide accurate, instantaneous academic guidance.

## Data Preparation

### Data Collection Strategy

The data preparation phase involved a systematic approach to gathering and processing academic information from multiple SJSU sources. Figure 1 [Insert Data Preparation Pipeline Diagram] illustrates the complete data preparation workflow.

1. **Source Identification and URL Collection**
   - Comprehensive collection of 15 authoritative SJSU web resources
   - Focus on engineering department FAQs, program requirements, and academic policies
   - Systematic categorization of URLs based on content type

2. **Content Extraction**
   - Implementation of LlamaIndex's Simple web page reader
   - Automated extraction of structured and unstructured content
   - Preservation of hierarchical information structure

3. **Text Processing and Segmentation**
   - Manual content review and section division
   - Removal of irrelevant content and formatting artifacts
   - Organization of content into logical segments for Q&A generation

4. **Q&A Dataset Generation**
   - Development of 260 question-answer pairs
   - Implementation of automated workflow using Restack
   - Quality assurance through manual review process

### Dataset Characteristics
- Total volume: >10,000 words
- Question-answer pairs: 260
- Content categories: FAQs, program requirements, policies, procedures
- Data format: Structured JSON with conversation pairs

## Model Initialization

The model initialization phase involved careful configuration of various parameters to optimize performance. Figure 2 [Insert Fine-tuning Pipeline Diagram] shows the complete fine-tuning architecture.

### Parameter Configuration Analysis

1. **PEFT Model Parameters**
```python
r = 16                  # Rank for LoRA adaptation
lora_alpha = 16         # LoRA scaling factor
lora_dropout = 0        # Optimized for inference
bias = "none"           # Memory optimization setting
```

2. **Target Modules**
- Attention components: q_proj, k_proj, v_proj, o_proj
- FFN components: gate_proj, up_proj, down_proj

3. **Memory Optimization**
- Implementation of gradient checkpointing
- Utilization of "unsloth" optimization for 30% VRAM reduction

## Model Specifications

### Pre-training Statistics
- GPU: Tesla T4
- Maximum memory: 14.748 GB
- Initial memory reservation: 3.74 GB

### Post-training Metrics
- Training duration: 336.92 seconds (5.62 minutes)
- Peak memory usage: 3.74 GB
- Memory utilization: 25.359% of maximum

## Model Training (Fine-tuning)

### Training Configuration

The fine-tuning process utilized the following optimized parameters:

```python
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
warmup_steps = 5
num_train_epochs = 10
learning_rate = 2e-4
```

### Training Process

1. **Dataset Preparation**
   - Standardization using unsloth.chat_templates
   - Implementation of ShareGPT format
   - Batch processing optimization

2. **Training Execution**
   - Mixed precision training (BF16/FP16)
   - Linear learning rate scheduling
   - 8-bit Adam optimizer implementation

## Evaluation

### Performance Metrics
- Response generation time: <1 second
- Memory efficiency: 25.359% utilization
- Training efficiency: 5.62 minutes total duration

### Quality Assessment
- Response accuracy evaluation
- Context relevance analysis
- Comparison with baseline response times

## Deployment

### Local Deployment Process

1. **Model Export**
   - GGUF file generation
   - Quantization method: F16
   - File size: 2.5GB

2. **Deployment Steps**
   - Ollama integration
   - Local model serving setup
   - Interface implementation

## Conclusion

The SJSU Academic Advisor Virtual Assistant demonstrates the successful application of fine-tuned LLMs in academic advisory services. The system achieves significant improvements in response times while maintaining high accuracy in academic guidance. Future work includes expanding the knowledge base and implementing additional specialized features for specific departments.

[Note: Insert figures at appropriate locations marked in the text. Each figure should include detailed captions explaining the architecture and workflow components.]
