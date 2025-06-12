import unittest

from src.api.gemini import ask_gemini
from src.api.prometheus import absolute_grading, relative_grading, absolute_grading_no_reference, relative_grading_no_reference


class TestAPIs(unittest.TestCase):
    def test_ask_gemini(self):
        response = ask_gemini("What is the capital of France?")
        assert isinstance(response, str)
        assert "Paris" in response or len(response) > 0

    @unittest.skip("Requires GPU and heavy model")
    def test_absolute_grading(self):
        instruction = "Explain how a black hole forms."
        response = "A black hole forms when a star collapses under its own gravity after a supernova."
        reference_answer = "Black holes are formed from the remnants of massive stars that undergo gravitational collapse after a supernova explosion."
        score_rubric = "Rate the response from 1 to 5 based on completeness, correctness, and clarity."
        absolute_grading(instruction, response, reference_answer, score_rubric)

    @unittest.skip(reason="Requires GPU and heavy model")
    def test_relative_grading(self):
        instruction = "Explain how gravity works."
        response_a = "Gravity pulls objects toward each other due to mass."
        response_b = "Gravity is a magnetic force that comes from the Earth’s core."
        reference_answer = "Gravity is the attractive force between two masses, described by Newton's Law of Universal Gravitation."
        score_rubric = "Choose the better response based on accuracy and scientific correctness."
        relative_grading(instruction, response_a, response_b, reference_answer, score_rubric)

    @unittest.skip(reason="Requires GPU and heavy model")
    def test_absolute_grading_no_reference(self):
        instruction = "Summarize the plot of Hamlet."
        response = "Hamlet is a play about a prince who seeks revenge on his uncle who murdered his father."
        score_rubric = "Rate the summary from 1 to 5 for clarity and key idea coverage."
        absolute_grading_no_reference(instruction, response, score_rubric)

    @unittest.skip(reason="Requires GPU and heavy model")
    def test_relative_grading_no_reference(self):
        instruction = "Translate 'Good morning' to French."
        response_a = "Bonjour"
        response_b = "Bonne matinée"
        score_rubric = "Pick the translation that is more natural and correct in French."
        relative_grading_no_reference(instruction, response_a, response_b, score_rubric)