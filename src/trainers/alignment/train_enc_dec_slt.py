"""Main training script for training attention based encoder decoder SLT models."""
import sys

from transformers import (
    AutoFeatureExtractor,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    WhisperForConditionalGeneration,
)
from transformers.utils import logging

from decoding.config import GenerationConfigCustom
from utilities.callbacks import init_callbacks
from utilities.collators import SpeechCollatorWithPadding
from utilities.data_utils import get_dataset
from utilities.eval_utils import compute_metrics
from utilities.general_utils import do_evaluate, do_generate
from utilities.model_utils import instantiate_aed_model
from utilities.training_arguments import (
    DataTrainingArguments,
    GeneralTrainingArguments,
    GenerationArguments,
    ModelArguments,
)
from utilities.training_utils import AdditionalLossTrackerTrainer
from models.ctc_encoder_plus_autoregressive_decoder import JointCTCAttentionEncoderDecoder
from models.auto_wrappers import CustomAutoModelForCTC, CustomModelForCausalLM

if __name__ == "__main__":
    logging.set_verbosity_debug()
    logger = logging.get_logger("transformers")
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GeneralTrainingArguments, GenerationArguments))

    model_args, data_args, training_args, gen_args = parser.parse_args_into_dataclasses()

    # 1. Collect, preprocess dataset and extract evaluation dataset
    dataset, training_eval_dataset = get_dataset(
        datasets_creation_config_path=data_args.datasets_creation_config,
        dataset_name=data_args.dataset_name,
        dataset_config=data_args.dataset_config,
        data_dir=data_args.data_dir,
        preprocessing_num_workers=data_args.preprocessing_num_workers,
        writer_batch_size=data_args.writer_batch_size,
        sampling_rate=data_args.sampling_rate,
        max_input_len=data_args.max_duration_in_seconds,
        min_input_len=data_args.min_duration_in_seconds,
        len_column=training_args.length_column_name,
        text_column=data_args.text_column_name,
        audio_column=data_args.audio_column_name,
        train_split=data_args.train_split,
        validation_split=data_args.validation_split,
        text_transformations=data_args.text_transformations,
        split_long_segments_to_chunks=data_args.split_long_segments_to_chunks,
        validation_slice_str=data_args.validation_slice,
        cut_validation_from_train=data_args.cut_validation_from_train,
        seed=data_args.validation_slice_seed,
        reshuffle_at_start=data_args.reshuffle_at_start,
        dataset_shard_size=data_args.dataset_shard_size,
        dump_prepared_dataset=data_args.dump_prepared_dataset,
        load_pure_dataset_only=data_args.load_pure_dataset_only,
    )

    logger.info(f"Dataset processed successfully.{dataset}")

    if training_args.preprocess_dataset_only:
        logger.info("Finished preprocessing dataset.")
        sys.exit(0)

    # 2. Create feature extractor and tokenizer
    feature_extractor = AutoFeatureExtractor.from_pretrained(training_args.feature_extractor_name)
    tokenizer = AutoTokenizer.from_pretrained(training_args.tokenizer_name)

    # 3. Instantiate model

    encoder = CustomAutoModelForCTC.from_pretrained(training_args.base_encoder_model)


    model = JointCTCAttentionEncoderDecoder.from_pretrained(training_args.from_pretrained)
    encoder = model.encoder
    decoder = CustomModelForCausalLM.from_config(model.config.decoder)
    model = JointCTCAttentionEncoderDecoder(encoder=encoder, decoder=decoder, config=model.config)


    #model = instantiate_aed_model(model_args, tokenizer, feature_extractor)

    # 4. Update generation config
    gen_config = GenerationConfigCustom(
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
        length_penalty=gen_args.length_penalty,
        early_stopping=gen_args.early_stopping,
        eos_token_id=tokenizer.eos_token_id,
        max_length=gen_args.max_length,
        num_beams=gen_args.num_beams,
        ctc_weight=gen_args.decoding_ctc_weight,
        ctc_margin=gen_args.ctc_margin,
        lm_weight=gen_args.lm_weight,
        lm_model=AutoModelForCausalLM.from_pretrained(gen_args.lm_model) if gen_args.lm_model else None,
        space_token_id=gen_args.space_token_id,
        apply_eos_space_trick=gen_args.apply_eos_space_trick,
        eos_space_trick_weight=gen_args.eos_space_trick_weight,
    )
    logger.info(f"Model updating generation config:\n")
    training_args.generation_max_length = gen_args.max_length
    training_args.generation_num_beams = gen_args.num_beams

    if isinstance(model, WhisperForConditionalGeneration):
        model.generation_config.num_beams = gen_args.num_beams
    else:
        model.generation_config = gen_config

    # 5. Initialize callbacks
    callbacks = init_callbacks(data_args, training_args, dataset, feature_extractor)

    # 6. Initialize data collator
    data_collator = SpeechCollatorWithPadding(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        padding=True,
        sampling_rate=data_args.sampling_rate,
        audio_path=data_args.audio_column_name,
        text_path=data_args.text_column_name,
        model_input_name=model.main_input_name,
        mask_unks=training_args.mask_unks,
        pad_to_multiple_of=data_args.pad_to_multiples_of,
    )

    # 7. Initialize trainer
    trainer_class = AdditionalLossTrackerTrainer if training_args.track_ctc_loss else Seq2SeqTrainer
    trainer = trainer_class(
        args=training_args,
        model=model,
        callbacks=callbacks,
        train_dataset=dataset[data_args.train_split],
        eval_dataset=training_eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(tokenizer, pred, gen_args.wandb_predictions_to_save),
    )

    # 8. Train model
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.restart_from or None)

    # 9. Evaluation
    if training_args.do_evaluate:
        do_evaluate(
            trainer=trainer,
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            gen_args=gen_args,
            training_args=training_args,
            data_args=data_args,
        )
    # 10. N-best generation
    if training_args.do_generate:
        do_generate(
            trainer=trainer,
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            gen_args=gen_args,
            data_args=data_args,
            gen_config=gen_config,
        )
