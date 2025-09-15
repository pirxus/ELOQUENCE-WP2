"""Parse results of OpenAI batch.
"""

import csv
import json
import logging
import re
import string
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import pandas as pd
import pylcs
from tqdm import tqdm


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s [%(filename)s:%(lineno)d]",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser()
    parser.add_argument("--responses_dir", type=str, required=True, help="Input directory containing responses JSONL files")
    parser.add_argument("--output_file", type=str, required=True, help="Output TSV file path")
    parser.add_argument("--id_field", type=str, default="docid", help="Field where to store the custom_id")
    # parser.add_argument("--text_field", type=str, default="text", help="Field where original text is stored")
    args = parser.parse_args()

    # main
    responses_dir = args.responses_dir
    output_file = args.output_file
    id_field = args.id_field
    # text_field = args.text_field

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    jsonl_files = sorted(list(Path(responses_dir).glob("*.jsonl")))
    # jsonl_files = sorted(list(Path(responses_dir).rglob("*.jsonl")))
    n_files = len(jsonl_files)

    # TODO make this more maintainable:
    header = (id_field, "info_seeking", "info_seeking_logprob", "self_contained", "self_contained_logprob", "response")
    
    with open(output_file, "w", encoding="utf-8") as out_f:
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(header)
        for i, jsonl_file in enumerate(jsonl_files, 1):
            logger.info(f"Processing file {i}/{n_files} ({jsonl_file})...")
            num_lines = sum(1 for _ in open(jsonl_file, "r", encoding="utf-8"))
            with open(jsonl_file, "r", encoding="utf-8") as in_f:
                pbar = tqdm(total=num_lines, desc=f"File {i}/{n_files}", unit="line")
                for line in in_f:
                    data = json.loads(line)
                    id_ = data.get("custom_id", "")
                    response = data["response"]["body"]["choices"][0]["message"]["content"]
                    logprobs = data["response"]["body"]["choices"][0]["logprobs"]["content"]
                    # if id_ == "...":
                        # print("here")
                    # try:
                    # except Exception as e:
                    #     logger.error(f"Error processing {id_}: {e}")
                    #     writer.writerow((id_, "", "", "", ""))
                    #     # logger.error(f"Response: {response}")
                    #     # TODO maybe use different values for error and empty response
                    
                    if not response:
                        writer.writerow((id_, "", "", "", "", ""))
                    
                    else:
                        verdicts = parse_response(response, logprobs)
                        writer.writerow((id_, *verdicts, response))
                    pbar.update(1)
                pbar.close()

    logger.info(f"DONE!")


def parse_response(response: str, logprobs: List) -> tuple:
    """Extract verdicts with logprobs from format:

    Information-seeking: [Short reasoning]. Verdict: [yes/no].
    Self-contained: [Short reasoning]. Verdict: [yes/no].
    """
    verdicts_dict = extract_verdicts_with_bytes(response)
    logprobs_dict = get_verdict_logprobs(logprobs, verdicts_dict)
    # dict with property: [verdict, logprob]
    res = {k: [v["verdict"], logprobs_dict[k]] for k, v in verdicts_dict.items() if v is not None}
    res_list = res["information_seeking"] + res["self_contained"]
    return tuple(res_list)


def extract_verdicts_with_bytes(text: str) -> dict:
    """TODO maybe improve by using lowercase??
    """
    properties = [
        "Information-seeking",
        "Self-contained"
    ]
    results = {}
    for prop in properties:
        pattern = rf"{prop}:.*?Verdict:\s*(yes|no)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            verdict = match.group(1).lower()
            verdict_start = match.start(1)
            verdict_end = match.end(1)
            byte_start = len(text[:verdict_start].encode('utf-8'))
            byte_end = len(text[:verdict_end].encode('utf-8'))
            bytes_idx = list(range(byte_start, byte_end))
            key = prop.lower().replace('-', '_')
            results[key] = {
                'verdict': verdict,
                'bytes_idx': bytes_idx
            }
        else:
            key = prop.lower().replace('-', '_')
            results[key] = None
    return results


# Iterate over the logprobs and save the ones corresponding to the verdicts
def get_verdict_logprobs(logprobs: List, parsed_verdicts: dict) -> dict:
    """TODO could be improved by analyzing saved logprobs of top5 tokens? 
    """
    byte_idx = 0
    res_dict = {}
    for token_dict in logprobs:
        token_bytes = token_dict["bytes"]
        n_bytes = len(token_bytes)
        byte_idx_end = byte_idx + n_bytes
        token_bytes_idx = list(range(byte_idx, byte_idx_end))
        # print(token_bytes_idx)
        for key, verdict_dict in parsed_verdicts.items():
            if verdict_dict is not None:
                if set(token_bytes_idx).intersection(verdict_dict["bytes_idx"]):
                    # print(token_dict['token'], token_dict['logprob'])
                    res_dict[key] = token_dict['logprob']
        byte_idx = byte_idx_end
    return res_dict



# def clean_text(text: str) -> str:
#     """Clean text before matching.
#     """
#     text = text.lower()
#     text = text.translate(str.maketrans("", "", string.punctuation))
#     # remove multiple whitespaces:
#     text = " ".join(text.split()) # TODO is this efficient?
#     text = text.strip()
#     return text




if __name__ == '__main__':
    main()
