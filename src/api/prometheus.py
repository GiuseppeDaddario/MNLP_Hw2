from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT
from prometheus_eval.prompts import RELATIVE_PROMPT

#Should be used with GPU

def absolute_grading(instruction,response,reference_answer,score_rubric):
    """
    Performs absolute evaluation of a model-generated response against a reference answer.

    Uses a grading rubric to assign a score (typically 1–5) and generate qualitative feedback.
    """
    model = VLLM(model="prometheus-eval/prometheus-7b-v2.0")
    judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)

    # Example usage
    # instruction = "Explain how a black hole forms."
    # response = "A black hole forms when a star collapses under its own gravity after a supernova."
    # reference_answer = "Black holes are formed from the remnants of massive stars that undergo gravitational collapse after a supernova explosion."
    # score_rubric = "Rate the response from 1 to 5 based on completeness, correctness, and clarity."

    score, feedback = judge.evaluate_absolute(
        instruction,
        response,
        reference_answer,
        score_rubric
    )

    print("Score:", score)
    print("Feedback:", feedback)


def relative_grading(instruction, response_a, response_b, reference_answer, score_rubric):
    """
    Performs relative evaluation between two model-generated responses.

    Compares both responses based on a given rubric and optionally a reference answer,
    returning the preferred response and qualitative feedback.
    """
    model = VLLM(model="prometheus-eval/prometheus-7b-v2.0")
    judge = PrometheusEval(model=model, relative_grade_template=RELATIVE_PROMPT)

    # Example usage
    # instruction = "Explain how gravity works."
    # response_a = "Gravity pulls objects toward each other due to mass."
    # response_b = "Gravity is a magnetic force that comes from the Earth’s core."
    # reference_answer = "Gravity is the attractive force between two masses, described by Newton's Law of Universal Gravitation."
    # score_rubric = "Choose the better response based on accuracy and scientific correctness."

    winner, feedback = judge.evaluate_relative(
        instruction,
        response_a,
        response_b,
        reference_answer,
        score_rubric
    )

    print("Winner: Response", "A" if winner == 1 else "B")
    print("Feedback:", feedback)

def absolute_grading_no_reference(instruction, response, score_rubric):
    """
    Performs absolute evaluation of a response without a reference answer.

    The evaluation relies only on the rubric and instruction to determine the quality
    of the generated response.
    """
    model = VLLM(model="prometheus-eval/prometheus-7b-v2.0")
    judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)

    # Example usage
    # instruction = "Summarize the plot of Hamlet."
    # response = "Hamlet is a play about a prince who seeks revenge on his uncle who murdered his father."
    # score_rubric = "Rate the summary from 1 to 5 for clarity and key idea coverage."

    score, feedback = judge.evaluate_absolute(
        instruction,
        response,
        reference_answer=None,  # 👈 Nessun riferimento
        score_rubric=score_rubric
    )

    print("Score:", score)
    print("Feedback:", feedback)


def relative_grading_no_reference(instruction, response_a, response_b, score_rubric):
    """
    Performs relative evaluation between two responses without a reference answer.

    Chooses the better response based on the provided rubric, focusing on clarity, correctness,
    or other specified criteria.
    """
    model = VLLM(model="prometheus-eval/prometheus-7b-v2.0")
    judge = PrometheusEval(model=model, relative_grade_template=RELATIVE_PROMPT)

    # Example usage
    # instruction = "Translate 'Good morning' to French."
    # response_a = "Bonjour"
    # response_b = "Bonne matinée"
    # score_rubric = "Pick the translation that is more natural and correct in French."

    winner, feedback = judge.evaluate_relative(
        instruction,
        response_a,
        response_b,
        reference_answer=None,  # 👈 Nessun riferimento
        score_rubric=score_rubric
    )

    print("Winner: Response", "A" if winner == 1 else "B")
    print("Feedback:", feedback)
