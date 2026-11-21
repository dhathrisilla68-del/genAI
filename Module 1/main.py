
import os
import json
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header, Depends 
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from dotenv import load_dotenv
import jwt
from schema import TextRequest


# Load environment variables
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in environment variables.")

model = os.getenv("CHAT_MODEL")
if not model:
    raise ValueError("CHAT_MODEL is not set in environment variables.")

# Initialize FastAPI
app = FastAPI(
    title="Module 1 - API Interaction of AI Tutor Project",
    description="AI Tutor API using Gemini for text, image, audio, video and notes summarization.",
    version="1.0.0"
)

# Initialize Gemini client
ai_client = genai.Client(api_key=gemini_api_key)

def get_current_user(token: str = Header(...)):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])["username"]
    except Exception:
        raise HTTPException(status_code=401, detail={"message": "Invalid token or expired token"})

# 1. Generate text explanation
@app.post("/generate_text")
async def generate_explanation(request: TextRequest, username: str = Depends(get_current_user)):

    prompt = f"""
    You are an AI tutor creating study material for a student.

    Topic: "{request.topic}"
    Subject: "{request.course or 'Not specified'}"
    Student level: "{request.student_level}"

    Explain the topic clearly and accurately using simple language.

    Output a valid JSON response containing:
    1. "topic": string
    2. "lesson_content": string
    3. "key_terms": list of objects with "term" and "definition"
    4. "example": string
    5. "practice_questions": list of 3 questions
    """

    try:
        response = ai_client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json"
            )
        )

        return JSONResponse(
            status_code=200,
            content=json.loads(response.text)
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# 2. Analyze academic image
@app.post("/analyze_image")
async def analyze_image(
    file: UploadFile = File(...),
    question: str = Form(...),
    username: str = Depends(get_current_user)
):

    if file.content_type not in ["image/jpeg", "image/png"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Only JPEG and PNG images are allowed."}
        )

    try:
        image_bytes = await file.read()

        image_input = types.Part.from_bytes(
            data=image_bytes,
            mime_type=file.content_type
        )

        prompt = f"""
        You are an AI tutor analyzing an academic image.

        Question: "{question}"

        Explain the image clearly and accurately.

        Output valid JSON containing:
        1. "image_analysis": detailed explanation
        2. "key_points": list of important points
        3. "related_concepts": list of related concepts
        """

        response = ai_client.models.generate_content(
            model=model,
            contents=[prompt, image_input],
            config=types.GenerateContentConfig(
                temperature=0.7,
                response_mime_type="application/json"
            )
        )

        return JSONResponse(
            status_code=200,
            content=json.loads(response.text)
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# 3. Evaluate audio answer
@app.post("/evaluate_audio_answer")
async def evaluate_audio_answer(
    file: UploadFile = File(...),
    question: str = Form(...),
    username: str = Depends(get_current_user)
):

    if file.content_type not in ["audio/mpeg", "audio/wav"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Only MP3 and WAV audio files are allowed."}
        )

    try:
        audio_bytes = await file.read()

        audio_input = types.Part.from_bytes(
            data=audio_bytes,
            mime_type=file.content_type
        )

        prompt = f"""
        You are an AI tutor evaluating a student's spoken answer.

        Question: "{question}"

        Listen to the student's audio answer and evaluate its content.

        Output valid JSON containing:
        1. "evaluation": detailed evaluation
        2. "score": integer score out of 10
        3. "feedback": suggestions for improvement
        """

        response = ai_client.models.generate_content(
            model=model,
            contents=[prompt, audio_input],
            config=types.GenerateContentConfig(
                temperature=0.7,
                response_mime_type="application/json"
            )
        )

        return JSONResponse(
            status_code=200,
            content=json.loads(response.text)
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# 4. Analyze academic video and generate MCQs
@app.post("/analyze_video")
async def analyze_video(
    file: UploadFile = File(...),
    question: str = Form(...),
    username: str = Depends(get_current_user)
):

    if file.content_type not in ["video/mp4", "video/avi"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Only MP4 and AVI video files are allowed."}
        )

    try:
        video_bytes = await file.read()

        video_input = types.Part.from_bytes(
            data=video_bytes,
            mime_type=file.content_type
        )

        prompt = f"""
        You are an AI tutor analyzing an academic video.

        Question: "{question}"

        Explain the video clearly.

        Generate 3 multiple-choice questions based on the video.
        Each question must have 4 options and one correct answer.

        Output valid JSON containing:
        1. "video_analysis": detailed video explanation
        2. "mcq_questions": list of objects containing
           "question", "options" and "correct_answer"
        """

        response = ai_client.models.generate_content(
            model=model,
            contents=[prompt, video_input],
            config=types.GenerateContentConfig(
                temperature=0.7,
                response_mime_type="application/json"
            )
        )

        return JSONResponse(
            status_code=200,
            content=json.loads(response.text)
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# 5. Summarize uploaded study notes
@app.post("/summarize_notes")
async def summarize_notes(
    file: UploadFile = File(...),
    student_level: str = Form("beginner"),
    username: str = Depends(get_current_user)
):

    filename = (file.filename or "").lower()

    if not (filename.endswith(".pdf") or filename.endswith(".txt")):
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid file. Please upload a PDF or TXT file."
            }
        )

    try:
        notes_bytes = await file.read()

        if not notes_bytes:
            return JSONResponse(
                status_code=400,
                content={"error": "The uploaded file is empty."}
            )

        if filename.endswith(".pdf"):
            notes_input = types.Part.from_bytes(
                data=notes_bytes,
                mime_type="application/pdf"
            )
        else:
            notes_input = notes_bytes.decode("utf-8-sig")

        prompt = f"""
        You are an intelligent AI study assistant.

        Read the uploaded study notes carefully.

        Student level: "{student_level}"

        Create a clear and useful summary using simple English.

        Include:
        1. "title": title of the notes
        2. "summary": complete summary in simple language
        3. "key_points": list of the most important points
        4. "important_terms": list of objects containing
           "term" and "definition"
        5. "conclusion": short conclusion of the notes
        6. "revision_questions": list of 5 questions for revision

        Make sure the summary covers the important information
        from the uploaded notes without inventing facts.

        Return a valid JSON response.
        """

        response = ai_client.models.generate_content(
            model=model,
            contents=[prompt, notes_input],
            config=types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json"
            )
        )

        return JSONResponse(
            status_code=200,
            content=json.loads(response.text)
        )

    except UnicodeDecodeError:
        return JSONResponse(
            status_code=400,
            content={"error": "Unable to read the TXT file. Please use UTF-8 encoding."}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
