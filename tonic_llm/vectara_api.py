import json
import logging
import os

import openai
import requests
# importing necessary functions from dotenv library
from dotenv import load_dotenv
from tonic_validate import Benchmark
from tonic_validate import ValidateScorer
import pandas as pd

# loading variables from .env file
load_dotenv()

TONIC_VALIDATE_API_KEY = os.getenv("TONIC_VALIDATE_API_KEY")
API_KEY = os.getenv("API_KEY")
CUSTOMER_ID = 43437896
CORPUS_ID = 3
SERVING_ENDPOINT = "api.vectara.io"
openai.api_key = os.getenv("OPENAI_API_KEY")


def _get_query_json(customer_id: int, corpus_id: int, query_value: str, num_results: int):
    """Returns a query JSON."""
    query = {
        "query": [
            {
                "query": query_value,
                "num_results": num_results,
                "corpus_key": [{"customer_id": customer_id, "corpus_id": corpus_id}],
            },
        ],
    }
    return json.dumps(query)


def _get_rag_response(top_response):
    # Access the top response from Vectrator's response
    # top_response = response['responseSet'][0]['response'][0]

    # Create the response dictionary in the desired format
    response = {
        "llm_answer": top_response['text'],  # Use 'text' for the answer
        "llm_context_list": [top_response['metadata'][i]['value'] for i in range(len(top_response['metadata'])) if
                             top_response['metadata'][i]['name'] == 'title'],  # Extract title from metadata
        "score": top_response['score'],
        "resultOffset": top_response['resultOffset'],
        "resultLength": top_response['resultLength'],
    }

    return response


def get_query(query: str, num_results: int = 1):
    """Queries the data.

    Args:
        customer_id: Unique customer ID in vectara platform.
        corpus_id: ID of the corpus to which data needs to be indexed.
        query_address: Address of the querying server. e.g., api.vectara.io
        api_key: A valid API key with query access on the corpus.

    Returns:
        (response, True) in case of success and returns (error, False) in case of failure.
    """
    post_headers = {
        "customer-id": f"{CUSTOMER_ID}",
        "x-api-key": API_KEY
    }

    response = requests.post(
        f"https://{SERVING_ENDPOINT}/v1/query",
        data=_get_query_json(CUSTOMER_ID, CORPUS_ID, query, num_results),
        verify=True,
        headers=post_headers)

    if response.status_code != 200:
        logging.error("Query failed with code %d, reason %s, text %s",
                      response.status_code,
                      response.reason,
                      response.text)
        return response

    # refactoring the code to handle the response
    first_response = response.json()['responseSet'][0]['response'][0]
    response = _get_rag_response(first_response)

    # message = response.json()
    # if (message["status"] and
    #         any(status["code"] != "OK" for status in message["status"])):
    #     logging.error("Query failed with status: %s", message["status"])
    #     return message["status"]
    #
    # for response_set in message["responseSet"]:
    #     for status in response_set["status"]:
    #         if status["code"] != "OK":
    #             return status

    return response


def make_scores_df(response_scores):
    scores_df = {
        "question": [],
        "reference_answer": [],
        "llm_answer": [],
        "retrieved_context": []
    }
    for score_name in response_scores.overall_scores:
        scores_df[score_name] = []
    for data in response_scores.run_data:
        scores_df["question"].append(data.reference_question)
        scores_df["reference_answer"].append(data.reference_answer)
        scores_df["llm_answer"].append(data.llm_answer)
        scores_df["retrieved_context"].append(data.llm_context)
        for score_name, score in data.scores.items():
            scores_df[score_name].append(score)
    return pd.DataFrame(scores_df)


if __name__ == "__main__":
    import openai

    query = "How to get hire at gitlab?"

    response = get_query(query)

    qa_pairs = []
    with open('./data/QA_gitlab.json', 'r') as qa_file:
        qa_pairs = json.load(qa_file)

    question_list = [qa_pair['question'] for qa_pair in qa_pairs]
    answer_list = [qa_pair['answer'] for qa_pair in qa_pairs]

    benchmark = Benchmark(questions=question_list, answers=answer_list)

    scorer = ValidateScorer(model_evaluator="gpt-3.5-turbo")
    # scorer = ValidateScorer()
    response_scores = scorer.score(benchmark, get_query, scoring_parallelism=2, callback_parallelism=2)
    print(response_scores)



    scores_df = make_scores_df(response_scores)
    # save dataframe to csv
    scores_df.to_csv("scores_df.csv")

    # Upload the run to the Tonic Validate API
    from tonic_validate import ValidateApi

    validate_api = ValidateApi()
    validate_api.upload_run("project-id", response_scores)

    # BLEU (Bilingual Evaluation Understudy) is an algorithm for evaluating the quality of text which has been
    # machine-translated from one natural language to another.


    # from tonic_validate import ValidateScorer, BLEUScorer
    #
    # # Initialize the BLEU scorer
    # bleu_scorer = BLEUScorer()
