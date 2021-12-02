from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer , Seq2SeqTrainingArguments , EncoderDecoderModel


def cofngiure_model(model_args , training_args ,data_args):
    if model_args.run_extraction:
        config = AutoConfig.from_pretrained(
            model_args.config_name
            if model_args.config_name is not None
            else model_args.model_name_or_path,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name
            if model_args.tokenizer_name is not None
            else model_args.model_name_or_path,
            use_fast=True,
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )
    elif model_args.run_generation:
        model_name='klue/bert-base'
        training_args = Seq2SeqTrainingArguments(
            do_train=True, 
            do_eval=True, 
            predict_with_generate=True,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            num_train_epochs=2,
            logging_dir='./logs',
            logging_steps=100,
            output_dir = training_args.output_dir , 
            evaluation_strategy = 'steps' , 
            eval_steps = 300,
            report_to="wandb",
            fp16 = True,
            save_total_limit=2
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)

        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size  if not data_args.run_seq2seq   else model.config.vocab_size


    return model , tokenizer 