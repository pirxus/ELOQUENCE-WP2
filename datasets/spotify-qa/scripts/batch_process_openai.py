
"""
Source: https://cookbook.openai.com/examples/batch_processing

NOTE we assume we don't hit context window limits, we don't check that
TODO should handle openai.APIConnectionError...
"""

import argparse
import logging
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import openai
import tiktoken
from tqdm import tqdm


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s [%(filename)s:%(lineno)d]",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


API_KEY = os.environ["OPENAI_PERSONAL_API_KEY"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset (local file)")
    parser.add_argument("--ids_keep", type=str, default=None, help="Subset of IDs to keep")
    parser.add_argument("--model_name", type=str, required=True, help="OpenAI model name")
    parser.add_argument("--system_prompt", type=str, required=True, help=".prompt file with system prompt")
    parser.add_argument("--outdir", type=str, required=True, help="Output dir to store results")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for API requests")
    parser.add_argument("--text_field", type=str, default="text", help="Field name for document texts")
    parser.add_argument("--id_field", type=str, default="docid", help="Field name for document IDs")
    parser.add_argument("--drop_duplicates", action="store_true", help="Drop duplicate documents (by ID)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for OpenAI API")
    parser.add_argument("--json_format", action="store_true", help="Output in JSON format")
    # parser.add_argument("--n_batches", type=int, default=1, help="Number of batches to split the dataset into")
    # parser.add_argument('--rate-limit', type=int, default=3000, help="Rate limit for OpenAI API (per minute)")
    args = parser.parse_args()

    logger.info(f"Loading dataset: {args.dataset}")
    sep = "\t" if args.dataset.endswith(".tsv") else ","
    df = pd.read_csv(args.dataset, sep=sep)

    logger.info("Casting ID field to string")
    df[args.id_field] = df[args.id_field].astype(str)

    if args.drop_duplicates:
        logger.info("Dropping duplicate documents by ID")
        df = df.drop_duplicates(subset=args.id_field).reset_index(drop=True)

    if args.ids_keep:
        logger.info(f"Filtering dataset by IDs in {args.ids_keep}")
        ids_keep = read_ids(args.ids_keep)
        df = keep_ids_in_df(df, ids_keep, args.id_field)

    # if outdir exists, read ids and remove them from dataset:
    last_batch = 0
    if Path(args.outdir).is_dir():
        df, last_batch = remove_done_ids(df, args.id_field, args.outdir)
    if len(df) == 0:
        logger.info("No documents left to process. Exiting.")
        return

    logger.info(f"Starting from batch {last_batch+1}")
    logger.info(f"{len(df)=}. Starting from batch {last_batch+1}")

    print("First dataset example:")
    print(df[args.text_field].iloc[0][:200] + " [...]")
    print("Last dataset example:")
    print(df[args.text_field].iloc[-1][:200] + " [...]")
    print()

    logger.info(f"Reading prompt from {args.system_prompt}")
    with open(args.system_prompt, "r") as f:
        system_prompt = f.read().strip()

    n_samples = len(df)
    pbar = tqdm(total=n_samples, desc="Done", unit="docs")
    # batch_size = n_samples // n_batches
    batch_size = args.batch_size
    n_batches = (n_samples + batch_size - 1) // batch_size

    for i, start_idx in enumerate(range(0, n_samples, batch_size), start=last_batch+1):
        end_idx = min(start_idx + batch_size, n_samples)
        df_batch = df.iloc[start_idx:end_idx]

        # Change format: dict with keys ids, and values texts
        input_batch = {
            id_: text for id_, text in zip(df_batch[args.id_field], df_batch[args.text_field])
        }

        # Count tokens:
        total_tokens = sum(num_tokens_from_string(text, args.model_name) for text in input_batch.values())
        print(f"\n--- {len(input_batch)} docs // {total_tokens} tokens")

        logger.info("Initializing OpenAI client...")
        batch_client = OpenAIBatchClient(API_KEY)

        logger.info(f"Creating tasks for batch {i}/{n_batches}...")
        batch_client.create_tasks(
            input_batch, system_prompt, args.model_name, temperature=args.temperature, json_format=args.json_format
        )

        logger.info(f"Launching batch job {i}/{n_batches}...")
        batch_client.launch_job()

        logger.info(f"Polling job status until completion for batch {i}/{n_batches}...")
        poll_status(batch_client)

        logger.info(f"Saving results for batch {i}/{n_batches}...")
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        out_file = f"{args.outdir}/batch_{i:05d}.jsonl"  # Unique output file for each batch
        batch_client.save_results(out_file)

        pbar.update(len(input_batch))

    pbar.close()

    logger.info("DONE!")


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 5,
    errors: tuple = (openai.RateLimitError, openai.APIConnectionError, openai.APIError),
):
    """Retry a function with exponential backoff.
    Source: https://platform.openai.com/docs/guides/rate-limits/error-mitigation
    """
    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay
        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Retry on specific errors
            except errors:
                # Increment retries
                num_retries += 1
                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
                # Sleep for the delay
                time.sleep(delay)
            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e
    return wrapper


class OpenAIBatchClient:
    """Wrapper for OpenAI API batch processing.
    """
    def __init__(self, api_key: str, org_id: Optional[str] = None):
        self.client = openai.OpenAI(api_key=api_key, organization=org_id)

    def create_tasks(
            self, inputs: dict, system_prompt: str, model_name: str = "gpt-4o-mini-2024-07-18",
            temperature: float = 0.2, json_format: bool = False
        ):
        """inputs: dict with keys ids, and values texts
        """
        # params:
        params = {
            "model": model_name,
            "temperature": temperature,
            "logprobs": True,
            "top_logprobs": 1,
            "max_tokens": 4096,
        }
        # TODO should make a class with config params to make this more flexible!!!
        if json_format:
            params["response_format"] = {"type": "json_object"}
        # Run:
        tasks = []
        for id_, text in inputs.items():
            task = {
                "custom_id": id_,  # NOTE using "custom_id" is compulsory!
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": text
                        }
                    ],
                }
            }
            # Add params to body:
            task["body"].update(params)
            tasks.append(task)
        self.tasks = tasks

    @retry_with_exponential_backoff
    def launch_job(self) -> None:
        # Save to tmp file:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file_name = f"/tmp/openai_tasks_{now}.jsonl"
        with open(batch_file_name, 'w') as file:
            for obj in self.tasks:
                file.write(json.dumps(obj) + '\n')
        self.batch_file_name = batch_file_name
        # Upload the file
        batch_file_obj = self.client.files.create(
            file=open(batch_file_name, "rb"),
            purpose="batch",
        )

        # create the batch job
        batch_job = self.client.batches.create(
            input_file_id=batch_file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",  # 24h is the only available
        )
        self.batch_file_obj = batch_file_obj
        self.batch_job = batch_job
        self.batch_job_id = batch_job.id

    @retry_with_exponential_backoff
    def check_status(self) -> tuple:
        self.batch_job = self.client.batches.retrieve(self.batch_job_id)
        status = self.batch_job.status
        request_counts = self.batch_job.request_counts
        return status, request_counts

    @retry_with_exponential_backoff
    def save_results(self, out_file: str) -> None:
        self.batch_job = self.client.batches.retrieve(self.batch_job_id)
        result_file_id = self.batch_job.output_file_id
        result = self.client.files.content(result_file_id).content
        with open(out_file, 'wb') as file:
            file.write(result)


def poll_status(batch_client, sleep_time=60) -> None:
    """See: https://help.openai.com/en/articles/9197833-batch-api-faq
    """
    status, request_counts = batch_client.check_status()

    while status in ["in_progress", "validating", "finalizing"]:
        time.sleep(sleep_time)
        status, request_counts = batch_client.check_status()
        print(f"{status=}")
        print(request_counts)

    if status == "completed":
        logger.info("Job completed successfully")

    elif status == "failed":
        errors = batch_client.batch_job.errors
        if errors:
            error_message = errors.data[0].message
            logger.error(f"Job failed with error: {error_message}")
        else:
            logger.error("Job failed")
        raise Exception("Job failed")

    else:
        logger.info(f"Job ended with unexpected status: {status}")

    print(request_counts)
    print("Job:\n", batch_client.batch_job)


def read_ids(file: str) -> List[str]:
    with open(file) as f:
        ids = [str(line.strip()) for line in f]
    return ids


def keep_ids_in_df(df: pd.DataFrame, ids: List, id_col: str = "id") -> pd.DataFrame:
    """Filter DataFrame by IDs, keeping the order in the list."""
    # Filter data by IDs:
    df_filtered = df[df[id_col].isin(ids)].copy()
    # Reorder according to IDs list:
    id_to_pos = {id_: pos for pos, id_ in enumerate(ids)}
    df_filtered['sort_pos'] = df_filtered[id_col].map(id_to_pos)
    df_filtered = df_filtered.sort_values('sort_pos').drop('sort_pos', axis=1)
    df_filtered = df_filtered.reset_index(drop=True).copy()
    return df_filtered


def remove_done_ids(df: pd.DataFrame, id_field: str, results_dir: str) -> tuple[pd.DataFrame, int]:
    """Remove already done documents from the dataset and return
    last_batch number
    """
    results_files = sorted(Path(results_dir).glob("batch_*.jsonl"))
    last_batch = 0
    if results_files:
        done_ids = set()
        # grab all ids from all files of each json:
        for file in results_files:
            with open(file, "r") as f:
                for line in f:
                    id_ = json.loads(line)["custom_id"]
                    done_ids.add(id_)
        last_file = results_files[-1]
        last_batch = int(last_file.stem.split("_")[-1])
        # remove encoded_ids from dataset:
        if done_ids:
            logger.info(f"Removing {len(done_ids)} already done documents")
            df = remove_ids_in_df(df, done_ids, id_field)
        else:
            logger.info(f"No encoded IDs found in {results_dir}")
    return df, last_batch


def remove_ids_in_df(df: pd.DataFrame, ids: set, id_col: str = "id") -> pd.DataFrame:
    """Filter DataFrame by IDs, keeping the order in the list."""
    df_filtered = df[~df[id_col].isin(ids)].copy()
    return df_filtered


def num_tokens_from_string(string: str, model_name: str) -> int:
    '''Returns the number of tokens in a text string.'''
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# def truncate_text_tokens(text, model_name: str):
#     """Truncate a string to have `max_tokens` according to the given encoding.
#     Source: https://cookbook.openai.com/examples/embedding_long_inputs
#     """
#     if model_name in ["text-embedding-3-large", "text-embedding-3-small"]:
#         encoding_name = 'cl100k_base'
#         max_tokens = 8191
#     else:
#         raise ValueError(f"Model {model_name} not supported")
#     encoding = tiktoken.get_encoding(encoding_name)
#     return encoding.encode(text)[:max_tokens]


if __name__ == "__main__":
    main()
