"""Main training script for the ECED architecture"""
import sys

from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    GenerationConfig,
    HfArgumentParser,
    Seq2SeqTrainer,
    Speech2TextConfig,
    Speech2TextFeatureExtractor,
    Speech2TextForConditionalGeneration,
    TrainerCallback,
    MarianConfig,
    MarianMTModel,
    MarianForCausalLM,
    Blip2QFormerConfig,
    AutoModelForSeq2SeqLM,
    T5Config,
    T5ForConditionalGeneration,
)
from transformers.utils import logging

from utilities.callbacks import init_callbacks
from utilities.collators import SpeechAlignedCollatorWithPadding
from utilities.data_utils import get_dataset
from utilities.eval_utils import compute_metrics_translation
from utilities.model_utils import average_checkpoints as average_checkpoints
from utilities.general_utils import do_evaluate, do_generate
from utilities.training_arguments import (
    DataTrainingArguments,
    GeneralTrainingArguments,
    GenerationArguments,
    ModelArguments,
    ConnectorArguments
)

from models.old_alignment import AlignmentConfig
from models.aligned import SpeechEncoderBridgeMarianEncoderDecoder
from models.t5_plus_marian import T5PlusMarian
from models.ctc_encoder_plus_autoregressive_decoder import JointCTCAttentionEncoderDecoder
from utilities.training_utils import AdditionalLossTrackerTrainer


if __name__ == "__main__":
    logging.set_verbosity_debug()
    logger = logging.get_logger("transformers")
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GeneralTrainingArguments, GenerationArguments, ConnectorArguments))

    model_args, data_args, training_args, gen_args, qformer_args = parser.parse_args_into_dataclasses()

    # 0. prepare the how2 dataset object..
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
    )

    # for lower resource splits of how2..
    if data_args.how2_low_resource_split_file:
        logger.info("Creating a low-resource simulated how2 split.")

        with open(data_args.how2_low_resource_split_file) as f:
            ids = set(map(lambda x: x[:11], f.read().splitlines()))

        dataset['train'] = dataset['train'].filter(lambda ex: ex['audio']['path'][:11] in ids, num_proc=data_args.preprocessing_num_workers)

    logger.info(f"Dataset processed successfully.{dataset}")

    if training_args.preprocess_dataset_only:
        logger.info("Finished preprocessing dataset.")
        sys.exit(0)

    # 2. Create feature extractor and tokenizer
    feature_extractor = AutoFeatureExtractor.from_pretrained(training_args.feature_extractor_name)
    tokenizer_target = AutoTokenizer.from_pretrained(training_args.tokenizer_name)

    if qformer_args.prompt_prefix:
        assert training_args.tokenizer_source_name is not None, "Error: missing source language tokenizer"
        tokenizer_source = AutoTokenizer.from_pretrained(training_args.tokenizer_source_name)
    else:
        tokenizer_source = None

    # 3. Instantiate model
    if 's2t' in model_args.base_encoder_model:
        encoder = Speech2TextForConditionalGeneration.from_pretrained(model_args.base_encoder_model)
        d_model = encoder.config.d_model
    else:
        encoder = JointCTCAttentionEncoderDecoder.from_pretrained(model_args.base_encoder_model)
        d_model = encoder.config.encoder.hidden_size

    try:
        if 'mt0' in model_args.base_decoder_model:
            decoder = AutoModelForSeq2SeqLM.from_pretrained(model_args.base_decoder_model, device_map='auto')
        elif 't5_english_pre' in model_args.base_decoder_model:
            decoder_config = T5Config.from_pretrained('unicamp-dl/translation-en-pt-t5')
            t5 = T5ForConditionalGeneration.from_pretrained('unicamp-dl/translation-en-pt-t5')
            decoder = T5PlusMarian.from_pretrained(model_args.base_decoder_model, decoder_config, t5, tokenizer_target, True)
            decoder.freeze_model()

            # cuz sharedtensors...
            decoder.decoder.lm_head.weight = decoder.decoder.model.decoder.embed_tokens.weight

        else:
            decoder = AutoModelForSeq2SeqLM.from_pretrained(model_args.base_decoder_model)
    except:
        print('exception occured')
        decoder = AutoModelForSeq2SeqLM.from_pretrained(model_args.base_decoder_model, use_safetensors=False)


    if model_args.from_config:
        apmo_config = AlignmentConfig.from_pretrained(model_args.from_config)
    else:

        qformer_config = Blip2QFormerConfig(
                hidden_size=qformer_args.conn_hidden_size,
                num_hidden_layers=qformer_args.conn_layers,
                num_attention_heads=qformer_args.conn_attn_heads,
                intermediate_size=qformer_args.qf_intermediate_size,
                hidden_act='gelu_new',
                cross_attention_frequency=1,
                encoder_hidden_size=d_model
            )

        if qformer_args.qf_config_overrides is not None:
            logger.info(f"Overriding config: {qformer_args.qf_config_overrides}")
            parsed_dict = dict(x.split("=") for x in qformer_args.qf_config_overrides.split(","))
            qformer_config.update(parsed_dict)

        apmo_config = AlignmentConfig(
                encoder_config=encoder.config,
                qformer_config=qformer_config,
                lm_config=decoder.config,
                num_query_tokens=qformer_args.n_queries,
                mm_pooling=qformer_args.qf_mm_pooling,
                mm_loss_weight=qformer_args.qf_mm_loss_weight,
                connector_type=qformer_args.connector_type,
            )

    if model_args.from_pretrained:
        model_path = model_args.from_pretrained
        if model_args.average_checkpoints:
            model_path = average_checkpoints(model_path)

        config = AlignmentConfig.from_pretrained(model_path)
        logger.info(f"Loading model from pretrained checkpoint...")
        
        # replace the loaded decoder with the specified one
        if model_args.replace_aligned_decoder:
            if 't5_asr_pretrain' in model_path:
                print('here')
                config_dummy = T5Config.from_pretrained(model_args.base_decoder_model)
                encoder_dummy = T5ForConditionalGeneration.from_pretrained(model_args.base_decoder_model)
                tokenizer_dummy = AutoTokenizer.from_pretrained('pirxus/how2_en_bpe8000_tc')
                decoder_dummy = T5PlusMarian(config=config_dummy, encoder=encoder_dummy, tokenizer=tokenizer_dummy, freeze_decoder=False)


                model_dummy = SpeechEncoderBridgeMarianEncoderDecoder.from_pretrained(model_path, config, encoder, decoder_dummy)
                logger.info(f"Replacing the decoder...")

                model = SpeechEncoderBridgeMarianEncoderDecoder(encoder=encoder, decoder=decoder, config=apmo_config)
                model.bridge = model_dummy.bridge

            else:
                decoder_dummy = MarianMTModel(config=decoder.config)

                model = SpeechEncoderBridgeMarianEncoderDecoder.from_pretrained(model_path, config, encoder, decoder_dummy)
                logger.info(f"Replacing the decoder...")
                model.decoder = decoder
        else:
            model = SpeechEncoderBridgeMarianEncoderDecoder.from_pretrained(model_path, config, encoder, decoder)

        if 'mt0' in model_args.base_decoder_model:
            for _, param in model.decoder.encoder.named_parameters():
                param.requires_grad = False

            for _, param in model.decoder.decoder.named_parameters():
                param.requires_grad = False
            model.decoder.lm_head.requires_grad = False
        elif 't5_english_pre' in model_args.base_decoder_model:
            pass
        elif 't5' in model_args.base_decoder_model:
            for _, param in model.decoder.encoder.named_parameters():
                param.requires_grad = False

            for _, param in model.decoder.decoder.named_parameters():
                param.requires_grad = False

            if hasattr(model.decoder, 'lm_head'):
                model.decoder.lm_head.requires_grad = False

        else:
            for _, param in model.decoder.model.encoder.named_parameters():
                param.requires_grad = False

            for _, param in model.decoder.model.decoder.named_parameters():
                param.requires_grad = False
            model.decoder.lm_head.requires_grad = False
            model.decoder.final_logits_bias.requires_grad = False

    else:
        model = SpeechEncoderBridgeMarianEncoderDecoder(encoder=encoder, decoder=decoder, config=apmo_config)

    logger.info(f"Finished loading model {model}")

    # 4. Update generation config
    bos = decoder.config.decoder_start_token_id if tokenizer_target.bos_token_id is None else tokenizer_target.bos_token_id
    gen_config = GenerationConfig(
        bos_token_id=bos,
        pad_token_id=tokenizer_target.pad_token_id,
        decoder_start_token_id=bos,
        decoder_end_token_id=tokenizer_target.eos_token_id,
        length_penalty=gen_args.length_penalty,
        early_stopping=gen_args.early_stopping,
        eos_token_id=tokenizer_target.eos_token_id,
        max_length=gen_args.max_length,
        num_beams=gen_args.num_beams,
    )

    logger.info(f"Model updating generation config:\n {str(gen_config)}")
    training_args.generation_max_length = gen_args.max_length
    training_args.generation_num_beams = gen_args.num_beams
    model.generation_config = gen_config
    model.decoder.generation_config = gen_config
    if hasattr(model.decoder, 'decoder'):
        model.decoder.decoder.generation_config = gen_config


    # 5. Initialize callbacks
    callbacks = init_callbacks(data_args, training_args, dataset, feature_extractor)

    # 6. Initialize data collator
    data_collator = SpeechAlignedCollatorWithPadding(
        feature_extractor=feature_extractor,
        tokenizer_target=tokenizer_target,
        tokenizer_source=tokenizer_source,
        padding=True,
        sampling_rate=data_args.sampling_rate,
        audio_path=data_args.audio_column_name,
        text_path=data_args.text_column_name,
        model_input_name=model.main_input_name,
        prompt_prefix=qformer_args.prompt_prefix,
    )

    # 7. Initialize trainer
    trainer_class = AdditionalLossTrackerTrainer if qformer_args.qf_mm_loss_weight > 0 else Seq2SeqTrainer
    trainer = Seq2SeqTrainer
    trainer = trainer_class(
            args=training_args,
            model=model,
            callbacks=callbacks,
            train_dataset=dataset[data_args.train_split],
            eval_dataset=training_eval_dataset,
            data_collator=data_collator,
            compute_metrics=lambda pred: compute_metrics_translation(tokenizer_target, pred, gen_args.wandb_predictions_to_save),
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
            tokenizer=tokenizer_target,
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
            tokenizer=tokenizer_target,
            gen_args=gen_args,
            data_args=data_args,
            gen_config=gen_config,
        )
