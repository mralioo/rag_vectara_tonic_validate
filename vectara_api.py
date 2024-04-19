import datetime
import json
import logging
import os

import openai
import pandas as pd
import requests
from dotenv import load_dotenv
from tonic_validate import Benchmark
from tonic_validate import ValidateApi
from tonic_validate import ValidateScorer
from tonic_validate.metrics import AnswerSimilarityMetric, AnswerConsistencyMetric, AnswerConsistencyBinaryMetric, \
    AugmentationAccuracyMetric, AugmentationPrecisionMetric, RetrievalPrecisionMetric

# loading variables from .env file
load_dotenv()


def _get_rag_best_response(response):
    best_response = response[0]
    for res in response:
        if res['score'] > best_response['score']:
            best_response = res
    return best_response


def _get_rag_response(top_response):
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


def get_response(prompt, params):
    # Default parameters
    if params is None:
        params = {}

    num_results = params.get("num_results", 1)
    sentences_before = params.get("sentences_before", 0)
    sentences_after = params.get("sentences_after", 0)
    lexicalInterpolationConfig = params.get("lexicalInterpolationConfig", {"lambda": 1})
    diversityBias = params.get("diversityBias", 0.0)
    semantics = params.get("semantics", 0)


    payload = {
        "query": [
            {
                "query": prompt,
                "queryContext": "",
                "start": 0,
                "numResults": num_results,
                "contextConfig": {
                    "charsBefore": 0,
                    "charsAfter": 0,
                    "sentencesBefore": sentences_before,
                    "sentencesAfter": sentences_after,
                    "startTag": "%START_SNIPPET%",
                    "endTag": "%END_SNIPPET%",
                },
                "corpusKey": [
                    {
                        "customerId": str(CUSTOMER_ID),
                        "corpusId": CORPUS_ID,
                        "semantics": semantics,
                        "lexicalInterpolationConfig": lexicalInterpolationConfig,
                        "dim": [],
                    }
                ],

                "rerankingConfig": {
                    "rerankerId": 272725718,
                    "maxConfig": {
                        "diversityBias": diversityBias,
                    }
                },

                "summary": [
                    {
                        "debug": False,
                        "chat": {"store": True, "conversationId": ""},
                        "maxSummarizedResults": 3,
                        "responseLang": "en",
                        "summarizerPromptName": "vectara-summary-ext-v1.2.0",
                        "factualConsistencyScore": True,
                    }
                ],
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-api-key": VECTARA_API_KEY,
        "customer-id": str(CUSTOMER_ID),
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        logging.error("Query failed with code %d, reason %s, text %s",
                      response.status_code,
                      response.reason,
                      response.text)
        return response

    # take the highest score response, the score is in the range of 0-1 and the higher indicate a grater propability
    # of being factual, while lower score indicate a greater probability of hallucination
    best_response = response.json()['responseSet'][0]['response'][0]
    result = _get_rag_response(best_response)

    return result


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

    # Load the API keys from the environment variables
    TONIC_VALIDATE_API_KEY = os.getenv("TONIC_VALIDATE_API_KEY")

    VECTARA_API_KEY = os.getenv("VECTARA_API_KEY")
    CUSTOMER_ID = int(os.getenv("CUSTOMER_ID"))
    CORPUS_ID = 12
    SERVING_ENDPOINT = "api.vectara.io"

    openai.api_key = os.getenv("OPENAI_API_KEY")

    url = "https://api.vectara.io/v1/query"

    # specify the version of the tested qa file
    version = "v3"
    qa_file = f"validation_qa_files/qa_gitlab_{version}.json"
    save_to_csv = False

    # RAG parameters
    params = {
        "num_results": 10,
        "sentences_before": 3,
        "sentences_after": 3,
        "semantics": 0,  # 0: "none", 1: "semantic", 2: "syntactic", 3: "both"
        "lexicalInterpolationConfig": {"lambda": 1}, #
        "diversityBias": 0.8
    }

    # test
    prompt_test = "how to get hired at Gitlab?"

    response = get_response(prompt_test, params=params)

    # Load the QA pairs from the json file for benchmarking
    qa_pairs = []
    with open(qa_file, 'r', encoding='utf-8') as qa_file:
        qa_pairs = json.load(qa_file)

    question_list = [qa_pair['question'] for qa_pair in qa_pairs]
    answer_list = [qa_pair['answer'] for qa_pair in qa_pairs]

    # References. if not exists, set None
    reference_articles = [qa_pair.get('reference_article', None) for qa_pair in qa_pairs]
    reference_answers = [qa_pair.get('reference_answer', None) for qa_pair in qa_pairs]

    # Create a benchmark object for Tonic Validate
    benchmark = Benchmark(questions=question_list, answers=answer_list)

    # Run the benchmark against the openai model and get the scores
    metrics = [
        AnswerSimilarityMetric(),
        # This metric is used to compare the similarity between the reference answer and the model answer
        AnswerConsistencyMetric(), # This metric is used to compare the consistency between the model answers
        # AnswerConsistencyBinaryMetric(), # This metric is used to compare the consistency between the model answers
        # AugmentationAccuracyMetric(), # This metric is used to compare the accuracy of the model answers
        # AugmentationPrecisionMetric(), # This metric is used to compare the precision of the model answers
        # RetrievalPrecisionMetric(), # This metric is used to compare the precision of the model answers

    ]
    scorer = ValidateScorer(metrics=metrics,
                            model_evaluator="gpt-3.5-turbo")

    response_scores = scorer.score(benchmark,
                                   lambda prompt: get_response(prompt, params=params),
                                   scoring_parallelism=2,
                                   callback_parallelism=2)
    print(response_scores)

    # save dataframe to csv

    if save_to_csv:
        results_file = f"scores_{version}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        scores_df = make_scores_df(response_scores)
        if not os.path.exists("./tonic_results"):
            os.makedirs("./tonic_results")
        scores_df.to_csv(f"./tonic_results/{results_file}", index=False)

    # Upload the run to the Tonic Validate API

    validate_api = ValidateApi(TONIC_VALIDATE_API_KEY)

    # include the model and version in the metadata and params

    run_metadata = {
                       "model": "gpt-3.5-turbo",
                       "version": version,
                   } | params

    # add your project id from the tonic validate project here
    project_id = "469df571-c755-4d7b-b74b-b999f5511f1e"

    validate_api.upload_run(project_id=project_id,
                            run=response_scores,
                            run_metadata=run_metadata)

    print(scores_df)
