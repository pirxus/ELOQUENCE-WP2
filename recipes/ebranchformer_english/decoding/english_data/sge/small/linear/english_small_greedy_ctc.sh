#!/usr/bin/bash
#$ -N ASR
#$ -q all.q@@gpu
#$ -l ram_free=16G,mem_free=16G
#$ -l scratch=2
#$ -l gpu=1,gpu_ram=20G

# Job should finish in 24 hours
ulimit -t 14400

# Enable opening multiple files
ulimit -n 4096

# Enable to save bigger checkpoints
ulimit -f unlimited

# Enable more threads per process by increasing virtual memory (https://stackoverflow.com/questions/344203/maximum-number-of-threads-per-process-in-linux)
ulimit -v unlimited

# Initialize environment
unset PYTHONPATH
unset PYTHONHOME

source /mnt/matylda5/ipoloka/miniconda3/bin/activate /mnt/matylda5/ipoloka/envs/hugginface_asr


SRC_DIR="/mnt/matylda5/ipoloka/projects/huggingface_asr"
EXPERIMENT="english_model_small_greedy_ctc"
EXPERIMENT_PATH="${SRC_DIR}/experiments/${EXPERIMENT}"
DATASET_DIR="/mnt/scratch/tmp/ipoloka/full_dataset"
HF_HOME="/mnt/scratch/tmp/ipoloka/hf_cache"
PROJECT="intermediate_mixing_v2"
PORT=9040

export WANDB_PROJECT="${PROJECT}"
export WANDB_RUN_ID="${EXPERIMENT}"
export HF_HOME=$HF_HOME
export PYTHONPATH="${PYTHONPATH}:${SRC_DIR}/src"
export $(/mnt/matylda4/kesiraju/bin/gpus 1) || exit 1
export http_proxy=socks5://localhost:$PORT
export https_proxy=socks5://localhost:$PORT
export PATH="/mnt/matylda5/ipoloka/utils/SCTK/bin:$PATH"

ssh -N -D $PORT pcspeech4 &
SSH_PID=$!

# Redirect stdout and stderr to a single file
exec &> "/mnt/matylda5/ipoloka/projects/huggingface_asr/experiments/${EXPERIMENT}.log"


cd $SRC_DIR

args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_train_batch_size="256"
  --per_device_eval_batch_size="16"
  --do_evaluate
  --learning_rate="5e-2"
  --logging_steps="1"
  --save_strategy="epoch"
  --early_stopping_patience="6"
  --evaluation_strategy="epoch"
  --report_to="wandb"
  --optim="adamw_torch"
  --dataloader_num_workers="6"
  --metric_for_best_model="eval_wer"
  --remove_unused_columns="False"
  --save_total_limit="1"
  --num_train_epochs="200"
  --greater_is_better="False"
  --group_by_length="False"
  --fp16
  --gradient_accumulation_steps="1"
  --load_best_model_at_end
  --max_steps="1"


  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="0.2"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="6"
  --dataset_name="${DATASET_DIR}"
  --writer_batch_size="50"
  --train_split="validation"
  --validation_split="validationXX"
  --validation_slice_seed="42"
  --cut_validation_from_train
  --validation_slice="1"
  --test_splits wsj_test fisher_swbd_dev voxpopuli_test tedlium3_test librispeech_test.clean librispeech_test.other commonvoice_en_test fleurs_test


  # Preprocessing related arguments
  --data_preprocessing_config="${SRC_DIR}/configs/default_data_preprocessing2d_pure.json"

  # Model related arguments
  --tokenizer_name="Lakoc/english_corpus_uni5000_normalized"
  --feature_extractor_name="Lakoc/log_80mel_extractor_16k"
  --from_pretrained="/mnt/matylda5/ipoloka/projects/huggingface_asr/experiments/english_mixing_linear_lr5_e4_small/checkpoint-234"
  --expect_2d_input
  --decoder_pos_emb_fixed

  # Generation related arguments
  --num_beams="1"
  --max_length="512"
  --predict_with_generate
  --decoding_ctc_weight="0.3"
)

python src/trainers/train_enc_dec_asr.py "${args[@]}"

kill $SSH_PID
