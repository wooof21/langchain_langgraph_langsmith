
"""
LangSmith Evaluation Datasets — Production Approach
Persistent, versioned test suites for LLM applications
"""

from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from langsmith.schemas import Run, Example
from dotenv import load_dotenv
import re
from config.chat_models import ChatModelFactory

load_dotenv()

client = Client()
llm = ChatModelFactory.openai_6_luna()


# ============================================================
# Step 1: Create an evaluation dataset
# ============================================================


def create_eval_dataset():
    """Create a dataset with test cases in LangSmith."""

    dataset_name = "qa-eval-dataset"

    # Delete if exists (for demo purposes — don't do this in production)
    existing = list(client.list_datasets(dataset_name=dataset_name))
    if existing:
        client.delete_dataset(dataset_id=existing[0].id)

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Q&A evaluation dataset for testing our chain",
    )

    # Add test examples — inputs and expected outputs
    examples = [
        {
            "inputs": {"question": "What is Python?"},
            "outputs": {
                "answer": "Python is a high-level programming language known for its readability and versatility."
            },
        },
        {"inputs": {"question": "What is 15 * 4?"}, "outputs": {"answer": "60"}},
        {
            "inputs": {"question": "What does HTML stand for?"},
            "outputs": {"answer": "HyperText Markup Language"},
        },
        {
            "inputs": {"question": "Name one benefit of exercise."},
            "outputs": {
                "answer": "Exercise improves cardiovascular health and reduces the risk of chronic diseases."
            },
        },
        {
            "inputs": {"question": "What is the capital of Japan?"},
            "outputs": {"answer": "Tokyo"},
        },
    ]

    for ex in examples:
        client.create_example(
            inputs=ex["inputs"], outputs=ex["outputs"], dataset_id=dataset.id
        )

    print(f"Created dataset '{dataset_name}' with {len(examples)} examples")
    return dataset_name


# ============================================================
# Step 2: Define the chain to evaluate
# ============================================================

prompt = ChatPromptTemplate.from_template("Answer this question concisely: {question}")
qa_chain = prompt | llm


@traceable(name="qa_target")
def qa_target(inputs: dict) -> dict:
    """
    Target function for LangSmith evaluation.
    Must accept a dict (inputs) and return a dict (outputs).
    """
    response = qa_chain.invoke({"question": inputs["question"]})
    return {"answer": response.content}


# ============================================================
# Step 3: Define evaluators
# ============================================================

# Evaluator: checks correctness against reference using LLM-as-judge
eval_llm = ChatModelFactory.openai_6_luna()

'''
run, exmaple: objects that LangSmith passes into the evaluator when it evaluates the dataset.

    * run -> what the application actually did. -> actual execution/result
        - inputs: {"questin": "some question"}
        - return: {"answer": "some answer"}
        - LangSmith records a Run contains both: 
            * run.inputs -> {"questin": "some question"}
            * run.outputs -> {"answer": "some answer"}
    * example -> the dataset example / expected test case. -> expected dataset example
        - dataset contains: {"inputs": {"question": "some question"}, "outputs":{"answer": "some answer"}}
        - evaluator receives an `Example` contains both:
            * example.inputs: {"questin": "some question"}
            * example.outputs: {"answer": "some answer"}
'''
def correctness(run: Run, example: Example) -> dict:
    """LLM-as-judge evaluator for correctness against reference answer."""
    prediction = run.outputs.get("answer", "")
    reference = example.outputs.get("answer", "")
    question = example.inputs.get("question", "")

    grade_prompt = ChatPromptTemplate.from_template(
        "You are a grader. Given a question, a submission, and a reference answer, "
        "determine if the submission is correct, accurate, and factual compared to "
        "the reference answer.\n\n"
        "Question: {question}\n"
        "Submission: {submission}\n"
        "Reference: {reference}\n\n"
        "Respond with ONLY 'Y' if correct or 'N' if incorrect."
    )
    result = eval_llm.invoke(
        grade_prompt.format(
            question=question, submission=prediction, reference=reference
        )
    )
    score = 1.0 if result.content.strip().upper() == "Y" else 0.0
    return {"key": "correctness", "score": score}


def helpfulness(run: Run, example: Example) -> dict:
    """LLM-as-judge evaluator for helpfulness (no reference needed)."""
    prediction = run.outputs.get("answer", "")
    question = example.inputs.get("question", "")

    grade_prompt = ChatPromptTemplate.from_template(
        "You are a grader. Given a question and a response, "
        "determine if the response is helpful, clear, and easy to understand.\n\n"
        "Question: {question}\n"
        "Response: {response}\n\n"
        "Respond with ONLY 'Y' if helpful or 'N' if not helpful."
    )
    result = eval_llm.invoke(
        grade_prompt.format(question=question, response=prediction)
    )
    score = 1.0 if result.content.strip().upper() == "Y" else 0.0
    return {"key": "helpfulness", "score": score}


# Custom evaluator: simple keyword check
def contains_answer(run: Run, example: Example) -> dict:
    """
    Custom evaluator — checks if the response contains
    key terms from the expected answer.
    """
    prediction = run.outputs.get("answer", "").lower()
    reference = example.outputs.get("answer", "").lower()

    # Extract key words from reference (words > 3 chars)
    key_words = [
        word 
        for word in re.findall(r"\b\w+\b", reference)
        if len(word) > 3
        ]

    # Check if at least 50% of key words appear in prediction
    if not key_words:
        return {"key": "contains_answer", "score": 0.0}

    matches = sum(1 for word in key_words if word in prediction)
    score = matches / len(key_words)

    return {"key": "contains_answer", "score": score}


# ============================================================
# Step 4: Run the evaluation
# ============================================================


def run_evaluation(dataset_name: str):
    """Run evaluation against the dataset."""

    print(f"\nRunning evaluation against '{dataset_name}'...\n")

    results = evaluate(
        qa_target,
        data=dataset_name,
        evaluators=[correctness, helpfulness, contains_answer],
        experiment_prefix="qa-chain-v1",  # Tags this run for comparison
        max_concurrency=2,
    )

    # Print summary
    print("\nEvaluation Results:")
    print("-" * 50)

    for result in results:
        question = result["run"].inputs.get("question", "N/A")
        answer = result["run"].outputs.get("answer", "N/A")

        print(f"\nQ: {question}")
        print(f"A: {answer[:80]}...")

        for eval_result in result["evaluation_results"]["results"]:
            print(f"  {eval_result.key}: {eval_result.score}")

    return results


# ============================================================
# Step 5: Compare experiments (after model or prompt change)
# ============================================================


def run_comparison(dataset_name: str):
    """
    Run a second experiment with a different config,
    then compare in LangSmith dashboard.
    """

    # New prompt — more detailed instructions
    detailed_prompt = ChatPromptTemplate.from_template(
        "Answer this question accurately and concisely. "
        "If it's a factual question, be precise. "
        "If it's a math question, show just the answer.\n\n"
        "Question: {question}"
    )
    v2_chain = detailed_prompt | llm

    @traceable(name="qa_target_v2")
    def qa_target_v2(inputs: dict) -> dict:
        response = v2_chain.invoke({"question": inputs["question"]})
        return {"answer": response.content}

    print("\nRunning v2 experiment for comparison...\n")

    results = evaluate(
        qa_target_v2,
        data=dataset_name,
        evaluators=[correctness, helpfulness, contains_answer],
        experiment_prefix="qa-chain-v2",  # Different prefix for comparison
        max_concurrency=2,
    )

    print("\nDone! Compare v1 vs v2 in LangSmith dashboard:")
    print("  → From LangSmith project → Datasets → qa-eval-dataset")
    print("  → Click 'Compare Experiments' to see v1 vs v2 side by side")

    return results




# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LangSmith Evaluation Datasets Demo")
    print("=" * 60)

    # Step 1: Create dataset
    dataset_name = create_eval_dataset()

    # Step 2: Run first evaluation (v1)
    print("\n" + "=" * 60)
    print("Experiment 1: Basic prompt (v1)")
    print("=" * 60)
    run_evaluation(dataset_name)

    # Step 3: Run second evaluation (v2) for comparison
    print("\n" + "=" * 60)
    print("Experiment 2: Detailed prompt (v2)")
    print("=" * 60)
    run_comparison(dataset_name)

    print("\n" + "=" * 60)
    print("All experiments logged to LangSmith!")
    print("=" * 60)