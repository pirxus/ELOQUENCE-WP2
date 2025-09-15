
### SETUP ############################################################
mkdir -p output/logs
mkdir -p data
# And copy transcriptions to data/spotify.text2


### RUN ##############################################################

### 1. notebooks/create_sample_ids.ipynb: 
#   run cells to create a subset of candidates from which we will sample some ids from

### 2. LLM Q extraction:
CANDIDATES="spotify_candidates_v0.1"
INITIAL_IDS="initial_ids_v0.1"
Q_EXTRACT_PROMPT="q_extraction_v0.1"
MODEL="gpt-4o-mini-2024-07-18"
TIME=`date +"%Y-%m-%d_%H-%M-%S"`
LOG_FILE="output/logs/q_extraction_${TIME}.log"
nohup python -u scripts/batch_process_openai.py \
    --dataset "output/data/${CANDIDATES}.tsv" \
    --ids_keep "output/lists/${INITIAL_IDS}.lst" \
    --model_name "$MODEL" \
    --system_prompt "prompts/${Q_EXTRACT_PROMPT}.prompt" \
    --outdir "output/q_extraction/${MODEL}/${Q_EXTRACT_PROMPT}/${INITIAL_IDS}" \
    --batch_size 20 \
    --temperature 0.7 \
    --text_field "text" \
    --id_field "podcast_id" \
    --drop_duplicates \
> "$LOG_FILE" 2>&1 &
echo "tail -f $LOG_FILE"
# NOTE openai batch mode is limited to 1M/2M queued tokens.

### 3. Parse Q extraction results:
CANDIDATES="spotify_candidates_v0.1"
INITIAL_IDS="initial_ids_v0.1"
Q_EXTRACT_PROMPT="q_extraction_v0.1"
MODEL="gpt-4o-mini-2024-07-18"
python -u scripts/parse_q_extraction_output.py \
    --input_file "output/data/${CANDIDATES}.tsv" \
    --responses_dir "output/q_extraction/${MODEL}/${Q_EXTRACT_PROMPT}/${INITIAL_IDS}" \
    --output_file "output/q_extraction/parsed/${MODEL}/${Q_EXTRACT_PROMPT}/${INITIAL_IDS}.tsv" \
    --id_field "podcast_id" \
    --text_field "text"

### 4. notebooks/q_hallucination_filter.ipynb:
#   run to keep subset of ids without "hallucinated questions"

### 5. LLM Question judgement:
CANDIDATES="spotify_candidates_v0.1"
INITIAL_IDS="initial_ids_v0.1"
Q_EXTRACT_PROMPT="q_extraction_v0.1"
Q_JUDGE_PROMPT="q_judge_v0.1"
MODEL="gpt-4o-mini-2024-07-18"
TIME=`date +"%Y-%m-%d_%H-%M-%S"`
LOG_FILE="output/logs/question_judge_${TIME}.log"
nohup python -u scripts/batch_process_openai.py \
    --dataset "output/q_extraction/parsed/${MODEL}/${Q_EXTRACT_PROMPT}/${INITIAL_IDS}.tsv" \
    --ids_keep "output/lists/q_extraction/${MODEL}/${Q_EXTRACT_PROMPT}/${INITIAL_IDS}_match_rate_90.lst" \
    --model_name "$MODEL" \
    --system_prompt "prompts/${Q_JUDGE_PROMPT}.prompt" \
    --outdir "output/q_quality/${MODEL}/${Q_JUDGE_PROMPT}/${INITIAL_IDS}" \
    --batch_size 100 \
    --temperature 0.3 \
    --text_field "question" \
    --id_field "podcast_id" \
    --drop_duplicates \
> "$LOG_FILE" 2>&1 &
echo "tail -f $LOG_FILE"

### 6. Parse question judgement results:
INITIAL_IDS="initial_ids_v0.1"
MODEL="gpt-4o-mini-2024-07-18"
Q_JUDGE_PROMPT="q_judge_v0.1"
python -u scripts/parse_quality_output.py \
    --responses_dir "output/q_quality/${MODEL}/${Q_JUDGE_PROMPT}/${INITIAL_IDS}" \
    --output_file "output/q_quality/parsed/${MODEL}/${Q_JUDGE_PROMPT}/${INITIAL_IDS}.tsv" \
    --id_field "podcast_id"

### 7. notebooks/q_quality_filter.ipynb:
#   run to keep subset of ids with "good quality" questions and make input for answer extraction

### 8. Answer extraction:
INITIAL_IDS="initial_ids_v0.1"
MODEL="gpt-4o-mini-2024-07-18"
Q_JUDGE_PROMPT="q_judge_v0.1"
Q_EXTRACT_PROMPT="q_extraction_v0.1"
A_EXTRACT_PROMPT="a_extraction_v0.1"
TIME=`date +"%Y-%m-%d_%H-%M-%S"`
LOG_FILE="output/logs/answer_extract_${TIME}.log"
nohup python -u scripts/batch_process_openai.py \
    --dataset "output/a_extraction/inputs/${MODEL}.${Q_EXTRACT_PROMPT}.${Q_JUDGE_PROMPT}/${INITIAL_IDS}.tsv" \
    --model_name "$MODEL" \
    --system_prompt "prompts/${A_EXTRACT_PROMPT}.prompt" \
    --outdir "output/a_extraction/${MODEL}/${Q_JUDGE_PROMPT}/${A_EXTRACT_PROMPT}/${INITIAL_IDS}" \
    --batch_size 20 \
    --temperature 0.3 \
    --id_field "podcast_id" \
    --text_field "a_extraction_input" \
    --drop_duplicates \
> "$LOG_FILE" 2>&1 &
echo "tail -f $LOG_FILE"

### 9. Parse answer extraction results:
CANDIDATES="spotify_candidates_v0.1"
INITIAL_IDS="initial_ids_v0.1"
Q_JUDGE_PROMPT="q_judge_v0.1"
Q_EXTRACT_PROMPT="q_extraction_v0.1"
A_EXTRACT_PROMPT="a_extraction_v0.1"
MODEL="gpt-4o-mini-2024-07-18"
python -u scripts/parse_a_extraction_output.py \
    --input_file "output/data/${CANDIDATES}.tsv" \
    --responses_dir "output/a_extraction/${MODEL}/${Q_JUDGE_PROMPT}/${A_EXTRACT_PROMPT}/${INITIAL_IDS}" \
    --output_file "output/a_extraction/parsed/${MODEL}/${Q_JUDGE_PROMPT}/${A_EXTRACT_PROMPT}/${INITIAL_IDS}.tsv" \
    --id_field "podcast_id" \
    --text_field "text"

### 10. notebooks/gather_results.ipynb:
#   Gather all results, keep good samples, and save in a single file (see filter in the notebook)


### 11. Make IR dataset:
python scripts/build_ir_dataset.py \
    --original_data_file "data/spotify.text2" \
    --pipeline_results_file "output/pipeline_results_v0.1.tsv" \
    --doc_size 100 \
    --doc_overlap 50 \
    --out_dir "output/text_ir_dataset_v0.1"

