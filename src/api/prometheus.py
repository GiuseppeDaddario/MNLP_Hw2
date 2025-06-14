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

    score, feedback = judge.evaluate_absolute(
        instruction,
        response,
        reference_answer,
        score_rubric
    )

    return{
        "score": score,
        "feedback": feedback
    }


def relative_grading(instruction, response_a, response_b, reference_answer, score_rubric):
    """
    Performs relative evaluation between two model-generated responses.

    Compares both responses based on a given rubric and optionally a reference answer,
    returning the preferred response and qualitative feedback.
    """
    model = VLLM(model="prometheus-eval/prometheus-7b-v2.0")
    judge = PrometheusEval(model=model, relative_grade_template=RELATIVE_PROMPT)

    winner, feedback = judge.evaluate_relative(
        instruction,
        response_a,
        response_b,
        reference_answer,
        score_rubric
    )

    return{
        "winner": "A" if winner == 1 else "B",
        "feedback": feedback
    }

def absolute_grading_no_reference(instruction, response, score_rubric):
    """
    Performs absolute evaluation of a response without a reference answer.

    The evaluation relies only on the rubric and instruction to determine the quality
    of the generated response.
    """
    model = VLLM(model="prometheus-eval/prometheus-7b-v2.0")
    judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)

    score, feedback = judge.evaluate_absolute(
        instruction,
        response,
        reference_answer=None,
        score_rubric=score_rubric
    )

    return {
        "score": score,
        "feedback": feedback
    }


def relative_grading_no_reference(instruction, response_a, response_b, score_rubric):
    """
    Performs relative evaluation between two responses without a reference answer.

    Chooses the better response based on the provided rubric, focusing on clarity, correctness,
    or other specified criteria.
    """
    model = VLLM(model="prometheus-eval/prometheus-7b-v2.0")
    judge = PrometheusEval(model=model, relative_grade_template=RELATIVE_PROMPT)

    winner, feedback = judge.evaluate_relative(
        instruction,
        response_a,
        response_b,
        reference_answer=None,
        score_rubric=score_rubric
    )

    return {
        "winner": "A" if winner == 1 else "B",
        "feedback": feedback
    }
