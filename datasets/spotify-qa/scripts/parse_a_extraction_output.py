"""Parse results of OpenAI batch.
"""

import csv
import json
import logging
import string
import re
import unicodedata
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
    parser.add_argument("--input_file", type=str, required=True, help="File containing original input data")
    parser.add_argument("--responses_dir", type=str, required=True, help="Input directory containing responses JSONL files")
    parser.add_argument("--output_file", type=str, required=True, help="Output TSV file path")
    parser.add_argument("--id_field", type=str, default="docid", help="Field where to store the custom_id")
    parser.add_argument("--text_field", type=str, default="text", help="Field where original text is stored")
    args = parser.parse_args()

    # main
    input_file = args.input_file
    responses_dir = args.responses_dir
    output_file = args.output_file
    id_field = args.id_field
    text_field = args.text_field

    logger.info(f"Reading original input data from {input_file}...")
    sep = "\t" if args.input_file.endswith(".tsv") else ","
    df = pd.read_csv(args.input_file, sep=sep)    

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    jsonl_files = sorted(list(Path(responses_dir).rglob("*.jsonl")))
    n_files = len(jsonl_files)

    if n_files == 0:
        logger.error(f"No JSONL files found in {responses_dir}")
        return

    header = (id_field, "answer", "a_verbatim", "match_rate_a", "a_span")
    
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
                    # if id_ == "...":
                        # print("here")
                    # try:
                    #     response = json.loads(response)
                    # except Exception as e:
                    #     logger.error(f"Error processing {id_}: {e}")
                    #     logger.error(f"Response: {response}")
                    #     writer.writerow((id_, "", "", "", "", ""))
                    #     # TODO maybe use different values for error and empty response
                    
                    if not response:
                        writer.writerow((id_, "", "", "", ""))
                    
                    else:
                        answer = parse_response(response)
                        if answer is not None and answer.lower() != "n/a":
                            span, verbatim, match_rate = find_matching_text_in_input(df, id_, answer, text_field, id_field)
                            writer.writerow((id_, answer, verbatim, match_rate, span))
                        else:
                            writer.writerow((id_, answer, "", "", ""))
                    pbar.update(1)
                pbar.close()

    logger.info(f"DONE!")


def parse_response(response: str) -> tuple:
    """Extract question from response with format:
    property: [...]
    """
    properties = [
        "Answer"
    ]
    results = {}
    for prop in properties:
        pattern = rf"{prop}:(.*?)\n" if prop != "Answer" else rf"{prop}:(.*)"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            results[prop.lower()] = match.group(1).strip()
        else:
            results[prop.lower()] = None
    return results["answer"]


def find_matching_text_in_input(
        df_input: pd.DataFrame, id_: str, candidate: str, text_field: str = "text",
        id_field: str = "docid"
    ) -> tuple:
    """Find matching text in the input data.
    Returns: 
        - start and end indices of the candidate text in the original text
        - the verbatim text (i.e. the text that was matched)
        - the match rate
    """
    original_input = df_input[df_input[id_field] == id_][text_field].values[0]
    if len(original_input) == 0:
        raise ValueError(f"Could not find matching input data for {id_}")
    # Clean both texts for matching but preserve original for verbatim extraction
    clean_original = clean_text(original_input)
    clean_candidate = clean_text(candidate)
    # Get character indices in the cleaned text
    t_span_in_clean = get_span(clean_original, clean_candidate)
    # Map clean text indices back to original text indices
    t_span = map_clean_indices_to_original(original_input, clean_original, t_span_in_clean)
    # Extract verbatim text from the original input using the mapped indices
    t_verbatim = original_input[t_span[0]:t_span[1]+1]
    # Calculate match rate using the cleaned lengths for consistency
    match_rate = len(clean_original[t_span_in_clean[0]:t_span_in_clean[1]+1]) / len(clean_candidate)
    return t_span, t_verbatim, match_rate


def clean_text(text: str) -> str:
    # Normalize text to decompose characters (e.g., en-dash to regular dash or decomposed form)
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    # Remove punctuation, including Unicode punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Remove extra whitespaces and strip the text
    text = " ".join(text.split())
    return text.strip()


def find_matching_indices(text: str, source: str) -> List[int]:
    """Find matching text in the source.
    Returns indices of the matching text in the source.
    """
    res = pylcs.lcs_string_idx(text, source)
    return [i for i in res if i != -1]


def get_span(text: str, question: str) -> tuple:
    """Get question span in the text as start and end indices.
    """
    q_indices = find_matching_indices(question, text)
    return (q_indices[0], q_indices[-1])


def map_clean_indices_to_original(original: str, cleaned: str, clean_span: tuple) -> tuple:
    """Map indices from cleaned text back to original text.
    
    Args:
        original: Original text with punctuation and capitalization
        cleaned: Cleaned text without punctuation, lowercase
        clean_span: (start, end) indices in the cleaned text
        
    Returns:
        (start, end) indices in the original text
    """
    # Create mapping from cleaned text positions to original text positions
    clean_to_orig_map = []
    clean_pos = 0
    
    for orig_pos, char in enumerate(original):
        # Skip characters that would be removed in cleaning (punctuation)
        if char.lower() in string.ascii_lowercase + string.digits + ' ':
            if clean_pos < len(cleaned) and (
                char.lower() == cleaned[clean_pos] or 
                (char.isspace() and cleaned[clean_pos].isspace())
            ):
                clean_to_orig_map.append(orig_pos)
                clean_pos += 1
                
    # If we couldn't build a complete mapping, fall back to approximate matching
    if clean_pos < len(cleaned):
        # Find the start in the original approximately
        start_region = original.lower().find(cleaned[clean_span[0]:clean_span[0]+10])
        if start_region == -1:
            # Very fallback case
            return (0, len(original)-1)
        
        # Approximate the end based on relative position
        end_region = start_region + (clean_span[1] - clean_span[0])
        if end_region >= len(original):
            end_region = len(original) - 1
            
        return (start_region, end_region)
    
    # Map the cleaned indices to original indices
    start_orig = clean_to_orig_map[clean_span[0]] if clean_span[0] < len(clean_to_orig_map) else clean_to_orig_map[-1]
    
    # Handle edge case where end index might be out of bounds
    if clean_span[1] < len(clean_to_orig_map):
        end_orig = clean_to_orig_map[clean_span[1]]
    else:
        # Use the last available mapping and extend to the end of a word
        end_orig = clean_to_orig_map[-1]
        # Try to extend to word boundaryT
        for i in range(end_orig + 1, len(original)):
            if original[i].isspace() or original[i] in string.punctuation:
                break
            end_orig = i
            
    return (start_orig, end_orig)


if __name__ == '__main__':
    main()
