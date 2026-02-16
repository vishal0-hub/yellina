import base64
import io
import json
import re
import tempfile

import google.generativeai as genai
from gtts import gTTS

GEMINI_API_KEY = "AIzaSyCiTwD84lw1iW-JYe5kvAFP_-5sURiH9OA"

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash")


def generate_interview_questions(resume_text):
    """
    Generate 5 interview questions based on resume content.
    Returns a list of 5 question strings.
    """
    prompt = f"""You are an expert technical interviewer. Based on the following resume, \
generate exactly 5 interview questions. The questions should be a mix of:
- 2 technical questions specific to the skills/technologies mentioned in the resume
- 2 behavioral/situational questions related to the candidate's experience
- 1 question about a specific project or achievement mentioned in the resume

Rules:
- Questions must be directly relevant to what is mentioned in the resume
- Each question should be clear, concise, and open-ended
- Questions should progressively increase in difficulty
- Do NOT ask generic questions unrelated to the resume content

Return ONLY a valid JSON array of exactly 5 strings. No markdown, no code blocks, \
no explanation. Example format:
["Question 1?", "Question 2?", "Question 3?", "Question 4?", "Question 5?"]

Resume:
{resume_text[:4000]}"""

    response = model.generate_content(prompt)
    content = response.text.strip()

    # Extract JSON array from response
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if match:
        questions = json.loads(match.group(0))
        if isinstance(questions, list) and len(questions) == 5:
            return questions

    # Fallback if parsing fails
    return [
        "Tell me about your most significant professional achievement.",
        "Describe a challenging technical problem you solved recently.",
        "How do you approach learning new technologies or skills?",
        "Tell me about a time you had to work under pressure.",
        "Where do you see your career heading in the next few years?",
    ]


def analyze_answer(question, answer, resume_text):
    """
    Analyze a candidate's answer and provide feedback + rating.
    Returns dict with 'feedback' (str) and 'rating' (int 1-10).
    """
    prompt = f"""You are an expert interview coach analyzing a candidate's response.

Question asked: "{question}"

Candidate's answer: "{answer}"

Context from their resume (for reference):
{resume_text[:2000]}

Analyze the answer and provide:
1. Constructive feedback (2-4 sentences) covering:
   - Relevance to the question
   - Depth and specificity of the response
   - Communication clarity
   - What could be improved

2. A rating from 1 to 10 where:
   - 1-3: Poor (vague, off-topic, or no substance)
   - 4-5: Below Average (partially relevant but lacks depth)
   - 6-7: Good (relevant and reasonably detailed)
   - 8-9: Very Good (specific, well-structured, demonstrates expertise)
   - 10: Excellent (exceptional depth, concrete examples, perfect delivery)

Return ONLY a valid JSON object with exactly two keys. No markdown, no code blocks.
Example: {{"feedback": "Your answer was...", "rating": 7}}"""

    response = model.generate_content(prompt)
    content = response.text.strip()

    # Extract JSON object from response
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        result = json.loads(match.group(0))
        if "feedback" in result and "rating" in result:
            result["rating"] = max(1, min(10, int(result["rating"])))
            return result

    # Fallback
    return {
        "feedback": "Thank you for your answer. Try to provide more specific examples from your experience.",
        "rating": 5,
    }


def generate_summary(qa_pairs, resume_text):
    """
    Generate an overall interview performance summary.
    qa_pairs: list of dicts with 'question', 'answer', 'feedback', 'rating' keys.
    Returns a summary string.
    """
    qa_text = ""
    ratings = []
    for i, qa in enumerate(qa_pairs):
        qa_text += f"\nQ{i + 1}: {qa['question']}\n"
        qa_text += f"Answer: {qa['answer']}\n"
        qa_text += f"Feedback: {qa['feedback']}\n"
        qa_text += f"Rating: {qa['rating']}/10\n"
        ratings.append(qa["rating"])

    avg_rating = sum(ratings) / len(ratings) if ratings else 0

    prompt = f"""You are an expert interview coach. Analyze the complete interview session below \
and provide a comprehensive performance summary.

Resume context:
{resume_text[:2000]}

Interview session:
{qa_text}

Average rating: {avg_rating:.1f}/10

Provide a structured summary with these exact sections:

OVERALL PERFORMANCE:
(2-3 sentences summarizing the candidate's overall interview performance and the average score)

STRENGTHS:
(3-4 bullet points identifying what the candidate did well, referencing specific answers)

AREAS FOR IMPROVEMENT:
(3-4 bullet points identifying weaknesses, referencing specific answers)

RECOMMENDATIONS:
(3-4 actionable suggestions for how the candidate can improve for future interviews)

FINAL SCORE: X/10

Keep the tone professional, constructive, and encouraging. Be specific -- reference actual \
answers and questions rather than making generic statements."""

    response = model.generate_content(prompt)
    return response.text.strip()


def speech_to_text(audio_base64):
    """
    Transcribe base64-encoded audio to text using Gemini multimodal.
    Returns the transcribed text string.
    """
    audio_bytes = base64.b64decode(audio_base64)

    response = model.generate_content(
        [
            "Transcribe this audio accurately. Return ONLY the transcribed text, "
            "nothing else. No labels, no quotes, no explanation.",
            {"mime_type": "audio/mp3", "data": audio_bytes},
        ]
    )
    return response.text.strip()


def text_to_speech(text, lang="en"):
    """
    Convert text to speech audio using gTTS.
    Returns base64-encoded MP3 audio string.
    """
    tts = gTTS(text=text, lang=lang)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return base64.b64encode(audio_buffer.getvalue()).decode()
