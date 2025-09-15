
import os
import string
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

import fire
import pandas as pd
import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)
logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s [%(filename)s:%(lineno)d]",
        level=logging.INFO
    )


@dataclass
class Config:
    out_dir: str
    pipeline_results_file: str
    original_data_file: str
    doc_size: int = 100 # words
    doc_overlap: int = 50 # words


class PodcastIRDatasetProcessor:
    def __init__(self, config: Config):
        self.config = config

    def run(self):
        logger.info(f"Starting...")
        self.read_data()
        self.process_data()
        self.save_data()
        logger.info(f"Done!")

    def read_data(self):
        logger.info(f"Reading original podcast transcriptions...")
        df_input = pd.read_csv(self.config.original_data_file, sep="\t", names=["podcast_id", "text"]) 
        logger.info(f"Reading pipeline QA results...")
        df_results = pd.read_csv(self.config.pipeline_results_file, sep="\t")
        self.df_input = df_input
        self.df_results = df_results
    
    def process_data(self):
        podcasts = dict(zip(self.df_input["podcast_id"], self.df_input["text"]))
        qa_pairs = (
            self.df_results
            [["q_verbatim", "a_verbatim", "podcast_id"]]
            .rename(columns={"podcast_id": "podcast_id", "q_verbatim": "question", "a_verbatim": "answer"})
            .reset_index(drop=True)
            .to_dict(orient="records")
        )
        logger.info(f"Processing {len(podcasts)} podcasts and {len(qa_pairs)} QA pairs...")
        ir_dataset = self.process_podcasts(podcasts, qa_pairs)
        self.df_docs = pd.DataFrame(ir_dataset["documents"])
        self.df_queries = pd.DataFrame(ir_dataset["queries"])
        self.df_qrels = pd.DataFrame(ir_dataset["qrels"])
        self.df_question_spans = pd.DataFrame(ir_dataset["question_spans"])
        self.df_answer_spans = pd.DataFrame(ir_dataset["answer_spans"])

    def save_data(self):
        os.makedirs(self.config.out_dir, exist_ok=True)
        
        logger.info(f"Saving docs.tsv")
        df_tmp = self.df_docs.rename(
            columns={
                "processed_text": "text",
                "start_idx": "start_word_idx",
                "end_idx": "end_word_idx",
                "hidden_indices": "masked_word_indices",
            })[
                ["docid", "text", "podcast_id", "start_word_idx", "end_word_idx", "original_text", "masked_word_indices"]
            ]
        # set original_text to empty string if masked_word_indices is empty, to save space:
        mask_empty = df_tmp["masked_word_indices"].apply(lambda x: len(x) == 0)
        df_tmp.loc[mask_empty, "original_text"] = ""
        print(df_tmp.shape)
        df_tmp.to_csv(f"{self.config.out_dir}/docs.tsv", sep="\t", index=False)
        # df_tmp.head(2)
        ## docs.tsv:
        # docid: Unique document identifier (format: pod_id_suffix#segment_number)
        # text: Processed text with questions replaced by [QUESTION_REMOVED] tokens
        # podcast_id: Original podcast identifier
        # start_idx: Start word index in the original transcript
        # end_idx: End word index in the original transcript
        # original_text: Original text of the segment
        # masked_indices: List of word indices that were hidden in the processed text (questions)

        logger.info(f"Saving queries.tsv")
        df_tmp = self.df_queries.copy()
        df_tmp = df_tmp.merge(self.df_question_spans, on=["qid", "text"], how="left")
        df_tmp = df_tmp.drop(columns=["text"]).rename(
            columns={"span_text": "text"}
        )[["qid", "text", "start_word_index", "end_word_index", "podcast_id"]]
        df_tmp.to_csv(f"{self.config.out_dir}/queries.tsv", sep="\t", index=False)
        print(df_tmp.shape)
        # df_tmp.head(2)
        ## queries.tsv:
        # qid: Unique query identifier
        # text: Question text
        # start_word_index: Start word index in the original transcript
        # end_word_index: End word index in the original transcript
        # podcast_id: Original podcast identifier

        logger.info(f"Saving qrels.tsv")
        df_tmp = self.df_qrels.copy()
        df_tmp = self.df_qrels.rename(columns={"relevance": "rel"})[["qid", "docid", "rel", "answer_id"]]
        df_tmp["rel"] = df_tmp["rel"].astype(np.float32)
        print(df_tmp.shape)
        df_tmp.to_csv(f"{self.config.out_dir}/qrels.tsv", sep="\t", index=False)
        ## qrels.tsv:
        # qid: Query identifier
        # docid: Document identifier
        # rel: Relevance grade (fraction of words in the answer contained in the segment)
        # answer_id: Answer identifier
        # df_tmp.head(2)

        logger.info(f"Saving answers.tsv")
        df_tmp = self.df_answer_spans.copy()
        df_tmp = df_tmp.rename(
            columns={
                "span_text": "text",
                "start_word_index": "start_word_idx",
                "end_word_index": "end_word_idx",
        })[["answer_id", "text", "podcast_id", "start_word_idx", "end_word_idx", ]]
        print(df_tmp.shape)
        df_tmp.to_csv(f"{self.config.out_dir}/answers.tsv", sep="\t", index=False)
        ## answers.tsv:
        # answer_id: Unique answer identifier
        # text: Answer text
        # podcast_id: Original podcast identifier
        # start_word_idx: Start word index in the original transcript
        # end_word_idx: End word index in the original transcript
        
        missing_qids = set(self.df_queries["qid"]) - set(self.df_qrels["qid"])
        missing_questions = self.df_queries[self.df_queries["qid"].isin(missing_qids)]["text"].tolist()
        logger.info(f"Missing questions in qrels: {len(missing_questions)}")

    def normalize_text(self, text: str) -> str:
        """Normalize text for matching purposes.
        """
        # Remove punctuation, convert to lowercase, 
        # remove multiple whitespace?
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator).lower()
        # text = " ".join(text.split()) # TODO is this efficient?
        # text = text.strip()
        return text
    
    def find_question_spans(self, podcast_id: str, transcript: str, questions: List[str], question_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Find spans in the transcript that contain the questions from QA pairs.
        
        Args:
            podcast_id: Identifier for the podcast
            transcript: The full podcast transcript
            questions: List of questions from QA pairs
            question_ids: List of qids corresponding to questions
            
        Returns:
            List of dictionaries with question spans and metadata
        """
        # Normalize the transcript for matching
        normalized_transcript = self.normalize_text(transcript)
        
        # Store original transcript for returning original text
        transcript_words = transcript.split()
        normalized_transcript_words = normalized_transcript.split()
        
        question_spans = []
        
        for idx, (question, qid) in enumerate(zip(questions, question_ids)):
            normalized_question = self.normalize_text(question)
            normalized_question_words = normalized_question.split()
            question_length = len(normalized_question_words)
            
            # Sliding window search through transcript
            for i in range(len(normalized_transcript_words) - question_length + 1):
                window = ' '.join(normalized_transcript_words[i:i+question_length])
                
                # If we found a match
                if normalized_question == window:
                    # Get original text with original case and punctuation
                    original_span_text = ' '.join(transcript_words[i:i+question_length])
                    # Store the span information
                    question_spans.append({
                        'qid': qid,  # Use the actual qid from QA pair
                        'text': question,
                        'span_text': original_span_text,
                        'start_word_index': i,
                        'end_word_index': i + question_length,
                        'podcast_id': podcast_id
                    })
        
        return question_spans
    
    def find_answer_spans(self, podcast_id: str, transcript: str, answers: List[str], answer_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Find spans in the transcript that contain the answers from QA pairs.
        Similar to question span finding but for answers.
        """
        normalized_transcript = self.normalize_text(transcript)
        
        transcript_words = transcript.split()
        normalized_transcript_words = normalized_transcript.split()
        
        answer_spans = []
        
        for idx, (answer, answer_id) in enumerate(zip(answers, answer_ids)):
            normalized_answer = self.normalize_text(answer)
            normalized_answer_words = normalized_answer.split()
            answer_length = len(normalized_answer_words)
            
            for i in range(len(normalized_transcript_words) - answer_length + 1):
                window = ' '.join(normalized_transcript_words[i:i+answer_length])
                
                if normalized_answer == window:
                    original_span_text = ' '.join(transcript_words[i:i+answer_length])
                    
                    answer_spans.append({
                        'answer_id': answer_id,  # Use the actual answer ID from QA pair
                        'answer_text': answer,
                        'span_text': original_span_text,
                        'start_word_index': i,
                        'end_word_index': i + answer_length,
                        'podcast_id': podcast_id
                    })
        
        return answer_spans
    
    def segment_transcript(self, podcast_id: str, transcript: str, question_spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Segment the transcript into overlapping chunks and apply information hiding.
        
        Args:
            podcast_id: Identifier for the podcast
            transcript: The full podcast transcript
            question_spans: Question spans to hide (can be empty for podcasts without QA pairs)
            
        Returns:
            List of segmented documents with questions removed
        """
        # Split transcript into words
        words = transcript.split()
        total_words = len(words)
        
        # Create a set of word indices to hide (questions)
        words_to_hide = set()
        for span in question_spans:
            for i in range(span['start_word_index'], span['end_word_index']):
                words_to_hide.add(i)
        
        # Create segments with overlap
        segments = []
        for start_idx in range(0, total_words, self.config.doc_size - self.config.doc_overlap):
            end_idx = min(start_idx + self.config.doc_size, total_words)
            # segment_id = f"{podcast_id}#{len(segments)+1:03d}"
            podcast_id_ = podcast_id.split("/")[-1] # last string after should be ID
            docid = f"{podcast_id_}#{len(segments)}"
            
            # Apply information hiding by removing question words
            segment_words = []
            hidden_indices = []
            
            for i in range(start_idx, end_idx):
                if i in words_to_hide:
                    # Replace with [QUESTION_REMOVED] as a marker
                    segment_words.append("[QUESTION_REMOVED]")
                    hidden_indices.append(i - start_idx)  # Relative to segment start
                else:
                    segment_words.append(words[i])
            
            # Store segment information
            segments.append({
                'docid': docid,
                'podcast_id': podcast_id,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'original_text': ' '.join(words[start_idx:end_idx]),
                'processed_text': ' '.join(segment_words),
                'hidden_indices': hidden_indices,
                'word_count': end_idx - start_idx
            })
            
            # If we've reached the end of the transcript, break
            if end_idx == total_words:
                break
        
        return segments
    
    def create_qrels(self, segments: List[Dict[str, Any]], answer_spans: List[Dict[str, Any]], qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create query-document relevance judgments (qrels).
        
        Args:
            segments: List of transcript segments
            answer_spans: List of answer spans
            qa_pairs: List of QA pairs
            
        Returns:
            List of relevance judgments
        """
        qrels = []
        
        # Create a mapping from answer_id to qid
        answer2query = {}
        for qa in qa_pairs:
            answer2query[qa['answer_id']] = qa['qid']

        # For each segment, check if it contains any answer spans
        for segment in segments:
            segment_start = segment['start_idx']
            segment_end = segment['end_idx']
            
            # print(answer_spans)
            for answer_span in answer_spans:
                answer_start = answer_span['start_word_index']
                answer_end = answer_span['end_word_index']
                
                # Check if there's overlap between the segment and answer span
                if (answer_start < segment_end and answer_end > segment_start and 
                    answer_span['podcast_id'] == segment['podcast_id']):
                    
                    # Calculate overlap percentage
                    overlap_start = max(segment_start, answer_start)
                    overlap_end = min(segment_end, answer_end)
                    overlap_length = overlap_end - overlap_start
                    answer_length = answer_end - answer_start
                    
                    # # Calculate relevance grade based on how much of the answer is contained
                    # if overlap_length == answer_length:
                    #     relevance = 3  # Complete answer
                    # elif overlap_length > answer_length * 0.5:
                    #     relevance = 2  # Major part of the answer
                    # else:
                    #     relevance = 1  # Minor part of the answer

                    # Float relevance grade: fraction of answer contained in segment
                    relevance = overlap_length / answer_length
                    
                    qid = answer2query.get(answer_span['answer_id'])
                    if qid:
                        qrels.append({
                            'qid': qid,
                            'docid': segment['docid'],
                            'relevance': relevance,
                            'answer_id': answer_span['answer_id']
                        })
        
        return qrels
    
    def process_podcasts(self, podcasts: Dict[str, str], qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process all podcasts to create the IR dataset.
        
        Args:
            podcasts: Dictionary mapping podcast IDs to transcripts
            qa_pairs: List of QA pairs with question, answer, and podcast_id fields
            
        Returns:
            Dictionary with queries, documents, and qrels
        """
        # First, assign consistent IDs to QA pairs
        for i, qa in enumerate(qa_pairs):
            qa['qid'] = str(i)
            qa['answer_id'] = i
        
        # Group QA pairs by podcast
        podcast_qa = {}
        for qa in qa_pairs:
            podcast_id = qa['podcast_id']
            
            if podcast_id not in podcast_qa:
                podcast_qa[podcast_id] = {'questions': [], 'answers': [], 'qa_pairs': [], 'question_ids': []}
            
            podcast_qa[podcast_id]['questions'].append(qa['question'])
            podcast_qa[podcast_id]['answers'].append(qa['answer'])
            podcast_qa[podcast_id]['qa_pairs'].append(qa)
            podcast_qa[podcast_id]['question_ids'].append(qa['qid'])
        
        # Process all podcasts (including those without QA pairs)
        all_segments = []
        all_question_spans = []
        all_answer_spans = []
        
        for podcast_id, transcript in tqdm(podcasts.items(), unit='podcast'):
            # Find question and answer spans if this podcast has QA pairs
            question_spans = []
            if podcast_id in podcast_qa:
                questions = podcast_qa[podcast_id]['questions']
                answers = podcast_qa[podcast_id]['answers']
                answer_ids = [qa['answer_id'] for qa in podcast_qa[podcast_id]['qa_pairs']]
                question_ids = podcast_qa[podcast_id]['question_ids']
                
                question_spans = self.find_question_spans(podcast_id, transcript, questions, question_ids)
                answer_spans = self.find_answer_spans(podcast_id, transcript, answers, answer_ids)
                all_question_spans.extend(question_spans)
                all_answer_spans.extend(answer_spans)
            
            # Segment ALL podcasts (with or without QA pairs)
            segments = self.segment_transcript(podcast_id, transcript, question_spans)
            all_segments.extend(segments)
        
        # Create qrels
        all_qrels = self.create_qrels(all_segments, all_answer_spans, qa_pairs)
        
        # Format queries
        queries = [{'qid': qa['qid'], 'text': qa['question']} for qa in qa_pairs]
        
        # Return the complete dataset
        return {
            'queries': queries,
            'documents': all_segments,
            'qrels': all_qrels,
            'question_spans': all_question_spans,
            'answer_spans': all_answer_spans
        }


def main(**kwargs):
    config = Config(**kwargs)
    builder = PodcastIRDatasetProcessor(config)
    return builder.run()


if __name__ == "__main__":
    fire.Fire(main)
