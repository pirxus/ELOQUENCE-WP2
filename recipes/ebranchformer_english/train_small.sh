#!/usr/bin/bash
#SBATCH --job-name TED
#SBATCH --account OPEN-28-57
#SBATCH --partition qgpu
#SBATCH --gpus 8
#SBATCH --nodes 1
#SBATCH --time 2-00:00:00
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/huggingface_asr/outputs/ebranchformer_english_small_v2.out

EXPERIMENT="ebranchformer_english_small_v2"
PROJECT="regularizations_english_corpus"
WORK_DIR="/mnt/proj1/open-28-58/lakoc/huggingface_asr"
ENV_DIR="/mnt/proj1/open-28-58/lakoc/LoCo-ASR"
RECIPE_DIR="${WORK_DIR}/recipes/ebranchformer_english"

export HF_HOME="/scratch/project/open-28-57/lakoc/huggingface_cache"
export HF_DATASETS_IN_MEMORY_MAX_SIZE=100000000000
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src"
export OMP_NUM_THREADS=64

conda deactivate
source activate loco_asr

EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"

cd $WORK_DIR

args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_train_batch_size="64"
  --per_device_eval_batch_size="64"
  --dataloader_num_workers="24"
  --num_train_epochs="50"
  --group_by_length="True"
  --bf16
  --do_train
  --do_evaluate
  --joint_decoding_during_training
  --load_best_model_at_end

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="2e-3"
  --warmup_steps="20000"
  --early_stopping_patience="5"
  --weight_decay="1e-6"
  --max_grad_norm="5.0"
  --lsm_factor="0.1"
  --gradient_accumulation_steps="1"

  # Logging, saving and evaluation related arguments
  --report_to="wandb"
  --logging_steps="10"
  --save_strategy="epoch"
  --evaluation_strategy="epoch"
  --wandb_predictions_to_save=600
  --greater_is_better="False"
  --save_total_limit="5"
  --track_ctc_loss

  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="0.2"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="64"
  --datasets_creation_config="${RECIPE_DIR}/datasets.json"
  --writer_batch_size="1000"
  --test_splits wsj_test fisher_swbd_test voxpopuli_test tedlium3_test librispeech_test.clean librispeech_test.other commonvoice_en_test fleurs_test
  --validation_slice=10000

  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing.json"

  # Model related arguments
  --from_encoder_decoder_config
  --tokenizer_name="Lakoc/english_corpus_uni5000"
  --feature_extractor_name="Lakoc/log_80mel_extractor_16k"
  --base_encoder_model="Lakoc/fisher_ebranchformer_enc_12_layers_fixed"
  --base_decoder_model="Lakoc/gpt2_256h_8l_add_head6_03"
  --ctc_weight="0.3"
  --decoder_pos_emb_fixed
  --expect_2d_input

  # Generation related arguments
  --num_beams="4"
  --max_length="512"
  --predict_with_generate
  --decoding_ctc_weight="0.3"
  --eval_beam_factor="10"
)

torchrun --standalone --nnodes=1 --nproc-per-node=8 src/trainers/train_enc_dec_asr.py "${args[@]}"
