
I used a conda env:

```bash	
conda create -n spotify-qa python=3.12 -y &&
conda activate spotify-qa &&
pip install -r requirements.txt
```

Set the env var with your OPENAI API key: `export OPENAI_PERSONAL_API_KEY="your-api-key"`.

Follow instruction in `run_v0.1.sh`.


----

Final output of pipeline has 4 tsv files with headers:

docs.tsv:

* docid: Unique document identifier (format: pod_id_suffix#segment_number)
* text: Processed text with questions replaced by [QUESTION_REMOVED] tokens
* podcast_id: Original podcast identifier
* start_idx: Start word index in the original transcript
* end_idx: End word index in the original transcript
* original_text: Original text of the segment -- it is not empty ONLY IF the segment contains a question
* masked_indices: List of word indices that were hidden in the processed text (questions)

queries.tsv:

* qid: Unique query identifier
* text: Question text
* start_word_index: Start word index in the original transcript
* end_word_index: End word index in the original transcript
* podcast_id: Original podcast identifier

qrels.tsv:

* qid: Query identifier
* docid: Document identifier
* rel: Relevance grade (fraction of words in the answer contained in the segment)
* answer_id: Answer identifier

answers.tsv:

* answer_id: Unique answer identifier
* text: Answer text
* podcast_id: Original podcast identifier
* start_word_idx: Start word index in the original transcript
* end_word_idx: End word index in the original transcript
