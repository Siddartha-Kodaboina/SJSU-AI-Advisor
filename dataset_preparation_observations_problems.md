# Dataset preparation observations/problems

## Problem with generating partial URLs

### Solution

As a SJSU Computer Enginering department advisor, review this documentation section and generate comprehensive specific Questions and Answers pairs that students might ask with curated answers providing all the details and **links(correct working links not partial links (should start with https or HTTP))**, instead of jsut mentioning where to find more information.
        
        Topic: {topic}
        Content: {attached the file}
        
        Generate a JSON response containing questions and answers covering all aspects (what, when, where, why, how, who, time, deadlines, status, payments, subjects, subject code, admissions, enrollment restrictions, links, can AI students take SE course wiseversa etc.).
        Format the response as a list of maps with 'question' and 'answer' keys.
        Only provide the JSON response, nothing else.
        Generate as many qns and answers as possible (in most of the cases atleast 15), if possible generate question about each line if necessary, Ensure questions are unique and non-repetitive and cover all the topics of the sections and cover all the questions that might be asked by students in real world. don't imagine additional context, generate questions from the attached context file.
        IN the file first line is the topic of the file, and MAke the data generated answers are real to the data in attached file
        """

        Example Output Format and Structure of Json File(Only provide a list of json objects with two keys question and answer):
        <json>
        [
            {"question": "What is the deadline for applying to SJSU?", "answer": "The deadline for applying to SJSU is 17th of October."},
            {"question": "What is the deadline for applying to SJSU?", "answer": "The deadline for applying to SJSU is 17th of October."},
            {"question": "What is the deadline for applying to SJSU?", "answer": "The deadline for applying to SJSU is 17th of October."},
            {"question": "What is the deadline for applying to SJSU?", "answer": "The deadline for applying to SJSU is 17th of October."}
        ]
        </json>

### Change in results:

#### Previus Results:
```
{
        "question": "Who should I contact for visa or residency-related queries as an international student?",
        "answer": "For visa or residency-related questions, you should contact the ISSS Office. Visit their website at /isss."
}
```

#### Results with the fix:

```
{
        "question": "Who should I contact for visa or residency-related queries as an international student?",
        "answer": "For visa or residency-related questions, you should contact the ISSS Office. Visit their website at https://www.sjsu.edu/isss."
}
```

## RAG ANswer Generation Prompt

```
prompt = f"""Context: {context}

Student Query: {user_query}

please provide a response based on the given context and to a Student Query as if it is texted by the SJSU Computer Engineering Advisor :"""
```
