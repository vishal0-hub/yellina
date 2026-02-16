"""
Unit tests for interview_ai.py functions
"""
import json
import unittest
from unittest.mock import Mock, patch, MagicMock
from Agent_app.interview_ai import (
    generate_interview_questions,
    analyze_answer,
    generate_summary,
    speech_to_text,
    text_to_speech,
)


class TestInterviewAI(unittest.TestCase):
    """Test suite for interview AI functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.sample_resume = """
        John Doe
        Senior Software Engineer

        Skills: Python, Django, React, AWS, Docker, Kubernetes

        Experience:
        - Led development of microservices architecture at TechCorp (2020-2023)
        - Built RESTful APIs serving 1M+ requests daily
        - Mentored team of 5 junior developers

        Education:
        BS Computer Science, Stanford University, 2018

        Projects:
        - E-commerce platform with real-time inventory management
        - ML-powered recommendation system increasing sales by 25%
        """

        self.sample_question = "Tell me about your experience with microservices architecture."
        self.sample_answer = "I led the migration of our monolithic application to microservices at TechCorp, breaking it into 12 independent services using Docker and Kubernetes."

    @patch('Agent_app.interview_ai.model')
    def test_generate_interview_questions_success(self, mock_model):
        """Test successful question generation"""
        # Mock Gemini response
        mock_response = Mock()
        mock_response.text = json.dumps([
            "Tell me about your experience with microservices architecture at TechCorp.",
            "How did you handle the challenges of serving 1M+ requests daily?",
            "Describe a situation where you mentored a junior developer.",
            "What was your approach to team leadership?",
            "Can you explain the ML-powered recommendation system you built?"
        ])
        mock_model.generate_content.return_value = mock_response

        questions = generate_interview_questions(self.sample_resume)

        self.assertEqual(len(questions), 5)
        self.assertIsInstance(questions, list)
        self.assertTrue(all(isinstance(q, str) for q in questions))
        mock_model.generate_content.assert_called_once()

    @patch('Agent_app.interview_ai.model')
    def test_generate_interview_questions_fallback(self, mock_model):
        """Test fallback questions when generation fails"""
        # Mock failed Gemini response
        mock_response = Mock()
        mock_response.text = "Invalid response"
        mock_model.generate_content.return_value = mock_response

        questions = generate_interview_questions(self.sample_resume)

        self.assertEqual(len(questions), 5)
        self.assertIn("Tell me about your most significant professional achievement.", questions)

    @patch('Agent_app.interview_ai.model')
    def test_generate_interview_questions_with_exception(self, mock_model):
        """Test handling of API exceptions"""
        mock_model.generate_content.side_effect = Exception("API Error")

        questions = generate_interview_questions(self.sample_resume)

        # Should return fallback questions
        self.assertEqual(len(questions), 5)
        self.assertIsInstance(questions, list)

    @patch('Agent_app.interview_ai.model')
    def test_analyze_answer_success(self, mock_model):
        """Test successful answer analysis"""
        mock_response = Mock()
        mock_response.text = json.dumps({
            "feedback": "Great answer! You demonstrated specific experience with the technology.",
            "rating": 8
        })
        mock_model.generate_content.return_value = mock_response

        result = analyze_answer(self.sample_question, self.sample_answer, self.sample_resume)

        self.assertIn("feedback", result)
        self.assertIn("rating", result)
        self.assertIsInstance(result["rating"], int)
        self.assertTrue(1 <= result["rating"] <= 10)

    @patch('Agent_app.interview_ai.model')
    def test_analyze_answer_rating_bounds(self, mock_model):
        """Test rating is bounded between 1 and 10"""
        mock_response = Mock()
        mock_response.text = json.dumps({
            "feedback": "Test feedback",
            "rating": 15  # Invalid rating
        })
        mock_model.generate_content.return_value = mock_response

        result = analyze_answer(self.sample_question, self.sample_answer, self.sample_resume)

        self.assertEqual(result["rating"], 10)  # Should be capped at 10

    @patch('Agent_app.interview_ai.model')
    def test_analyze_answer_fallback(self, mock_model):
        """Test fallback response when analysis fails"""
        mock_response = Mock()
        mock_response.text = "Invalid JSON"
        mock_model.generate_content.return_value = mock_response

        result = analyze_answer(self.sample_question, self.sample_answer, self.sample_resume)

        self.assertIn("feedback", result)
        self.assertIn("rating", result)
        self.assertEqual(result["rating"], 5)  # Default rating

    @patch('Agent_app.interview_ai.model')
    def test_generate_summary(self, mock_model):
        """Test summary generation"""
        qa_pairs = [
            {
                "question": "Question 1",
                "answer": "Answer 1",
                "feedback": "Good answer",
                "rating": 7
            },
            {
                "question": "Question 2",
                "answer": "Answer 2",
                "feedback": "Excellent answer",
                "rating": 9
            },
            {
                "question": "Question 3",
                "answer": "Answer 3",
                "feedback": "Needs improvement",
                "rating": 5
            }
        ]

        mock_response = Mock()
        mock_response.text = """
        OVERALL PERFORMANCE:
        The candidate showed strong technical knowledge with an average score of 7.0/10.

        STRENGTHS:
        - Excellent communication skills
        - Deep technical expertise

        AREAS FOR IMPROVEMENT:
        - Could provide more specific examples

        RECOMMENDATIONS:
        - Practice STAR method for behavioral questions

        FINAL SCORE: 7/10
        """
        mock_model.generate_content.return_value = mock_response

        summary = generate_summary(qa_pairs, self.sample_resume)

        self.assertIsInstance(summary, str)
        self.assertIn("OVERALL PERFORMANCE", summary)
        self.assertIn("STRENGTHS", summary)
        self.assertIn("RECOMMENDATIONS", summary)

    @patch('Agent_app.interview_ai.model')
    def test_speech_to_text(self, mock_model):
        """Test speech to text conversion"""
        mock_response = Mock()
        mock_response.text = "This is the transcribed text from audio."
        mock_model.generate_content.return_value = mock_response

        # Mock base64 audio data
        audio_base64 = "SGVsbG8gV29ybGQ="  # "Hello World" in base64

        result = speech_to_text(audio_base64)

        self.assertEqual(result, "This is the transcribed text from audio.")
        mock_model.generate_content.assert_called_once()

    @patch('Agent_app.interview_ai.gTTS')
    def test_text_to_speech(self, mock_gtts):
        """Test text to speech conversion"""
        # Mock gTTS instance
        mock_tts_instance = Mock()
        mock_gtts.return_value = mock_tts_instance

        text = "Hello, this is a test."
        result = text_to_speech(text)

        # Check that gTTS was called with correct parameters
        mock_gtts.assert_called_once_with(text=text, lang="en")
        mock_tts_instance.write_to_fp.assert_called_once()

        # Result should be base64 string
        self.assertIsInstance(result, str)

    @patch('Agent_app.interview_ai.gTTS')
    def test_text_to_speech_different_language(self, mock_gtts):
        """Test text to speech with different language"""
        mock_tts_instance = Mock()
        mock_gtts.return_value = mock_tts_instance

        text = "Ciao, questo è un test."
        result = text_to_speech(text, lang="it")

        mock_gtts.assert_called_once_with(text=text, lang="it")


class TestInterviewAIIntegration(unittest.TestCase):
    """Integration tests for interview AI functions"""

    def setUp(self):
        self.sample_resume = """
        Jane Smith - Full Stack Developer
        5 years experience in web development
        Skills: JavaScript, Python, React, Node.js
        """

    @patch('Agent_app.interview_ai.model')
    def test_full_interview_flow(self, mock_model):
        """Test complete interview flow from questions to summary"""
        # Mock question generation
        questions_response = Mock()
        questions_response.text = json.dumps([
            "Q1?", "Q2?", "Q3?", "Q4?", "Q5?"
        ])

        # Mock answer analysis
        analysis_response = Mock()
        analysis_response.text = json.dumps({
            "feedback": "Good answer",
            "rating": 7
        })

        # Mock summary generation
        summary_response = Mock()
        summary_response.text = "Interview Summary"

        mock_model.generate_content.side_effect = [
            questions_response,
            analysis_response,
            summary_response
        ]

        # Generate questions
        questions = generate_interview_questions(self.sample_resume)
        self.assertEqual(len(questions), 5)

        # Analyze an answer
        analysis = analyze_answer(questions[0], "Test answer", self.sample_resume)
        self.assertIn("feedback", analysis)
        self.assertIn("rating", analysis)

        # Generate summary
        qa_data = [{
            "question": questions[0],
            "answer": "Test answer",
            "feedback": analysis["feedback"],
            "rating": analysis["rating"]
        }]
        summary = generate_summary(qa_data, self.sample_resume)
        self.assertIsInstance(summary, str)


if __name__ == '__main__':
    unittest.main()