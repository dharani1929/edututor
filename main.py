from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from langchain_ibm import WatsonxLLM
from langchain.prompts import PromptTemplate
import pinecone
import os
from dotenv import load_dotenv
import uuid
import json

load_dotenv()

app = FastAPI()

# Watsonx Configuration
watsonx_llm = WatsonxLLM(
    model_id=os.getenv("WATSONX_MODEL_ID"),
    url=os.getenv("WATSONX_ENDPOINT"),
    apikey=os.getenv("WATSONX_API_KEY"),
    project_id=os.getenv("WATSONX_PROJECT_ID")
)

# Pinecone Configuration
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="gcp-starter"
)
index = pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))

# Data Models
class QuizRequest(BaseModel):
    topic: str
    difficulty: str
    user_id: str

class Question(BaseModel):
    id: str
    question: str
    options: list[str]
    correct_answer: str

class QuizSubmission(BaseModel):
    user_id: str
    quiz_id: str
    responses: dict[str, str]  # {question_id: answer}

# Quiz Generation
quiz_prompt = PromptTemplate(
    input_variables=["topic", "difficulty"],
    template="Generate 5 {difficulty} multiple-choice questions about {topic}. Format as JSON: [{'id':'uuid','question':'text','options':['a','b','c','d'],'correct_answer':'a'}]"
)

@app.post("/generate-quiz")
async def generate_quiz(request: QuizRequest):
    try:
        prompt = quiz_prompt.format(topic=request.topic, difficulty=request.difficulty)
        response = watsonx_llm(prompt)
        
        # Clean and parse response
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        questions = json.loads(response[json_start:json_end])
        
        # Add UUIDs and validate
        for q in questions:
            q['id'] = str(uuid.uuid4())
        
        # Store in Pinecone with answers
        quiz_id = str(uuid.uuid4())
        metadata = {
            "user_id": request.user_id,
            "topic": request.topic,
            "difficulty": request.difficulty,
            "questions": questions
        }
        index.upsert([(quiz_id, [0]*768, metadata)])
        
        # Return without answers
        client_questions = [{**q, "correct_answer": None} for q in questions]
        return {"quiz_id": quiz_id, "questions": client_questions}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/submit-quiz")
async def submit_quiz(submission: QuizSubmission):
    try:
        # Retrieve quiz from Pinecone
        quiz_data = index.fetch(ids=[submission.quiz_id])
        if not quiz_data['vectors']:
            raise HTTPException(status_code=404, detail="Quiz not found")
            
        metadata = quiz_data['vectors'][submission.quiz_id]['metadata']
        questions = metadata['questions']
        
        # Calculate score
        score = 0
        for q in questions:
            if submission.responses.get(q['id']) == q['correct_answer']:
                score += 1
        
        # Update metadata
        metadata['score'] = score
        metadata['completed'] = True
        index.update(id=submission.quiz_id, set_metadata=metadata)
        
        return {"score": score, "total": len(questions)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)