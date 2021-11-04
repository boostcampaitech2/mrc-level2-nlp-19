import logging
import os
import sys

from typing import List, Callable, NoReturn, NewType, Any
import dataclasses
import nltk
from datasets import load_metric, load_from_disk, Dataset, DatasetDict

from transformers import BartConfig, BartTokenizerFast, BartForConditionalGeneration
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM

from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    HfArgumentParser,
    set_seed,
)

from tokenizers import Tokenizer
from tokenizers.models import WordPiece

from utils_qa import check_no_error

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)


nltk.download('punkt')
logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments)
    )
    model_args, data_args = parser.parse_args_into_dataclasses()
    print(model_args.model_name_or_path)

    training_args = Seq2SeqTrainingArguments(
        output_dir='./models/train_dataset',
        overwrite_output_dir=True,
        do_train=True, 
        do_eval=True, 
        predict_with_generate=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        save_total_limit=2,
        fp16=True
    )

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)

    
    config = BartConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name is not None
        else model_args.model_name_or_path,
    )
    tokenizer = BartTokenizerFast.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name is not None
        else model_args.model_name_or_path,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    """
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
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )
    """

    print(
        type(training_args),
        type(model_args),
        type(datasets),
        type(tokenizer),
        type(model),
    )

    # do_train mrc model 혹은 do_eval mrc model
    if training_args.do_train or training_args.do_eval:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: Seq2SeqTrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:

    # dataset을 전처리합니다.
    # training과 evaluation에서 사용되는 전처리는 아주 조금 다른 형태를 가집니다.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )
    max_target_length = 128

    # Train preprocessing / 전처리를 진행합니다.
    def pre_processing_function(examples):
        inputs = [f'question: {q}  context: {c} </s>' for q, c in zip(examples[question_column_name], examples[context_column_name])]
        targets = [f'{a["text"][0]} </s>' for a in examples[answer_column_name]]
        model_inputs = tokenizer(
            inputs, 
            max_length=max_seq_length, 
            padding="max_length" if data_args.pad_to_max_length else False, 
            truncation=True
        )

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, 
                max_length=max_target_length, 
                padding="max_length" if data_args.pad_to_max_length else False, 
                truncation=True
            )

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["example_id"] = []
        
        for i in range(len(model_inputs["labels"])):
            model_inputs["example_id"].append(examples["id"][i])
        
        return model_inputs

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]

        # dataset에서 train feature를 생성합니다.
        train_dataset = train_dataset.map(
            pre_processing_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        eval_dataset = datasets["validation"]

        # Validation Feature 생성
        eval_dataset = eval_dataset.map(
            pre_processing_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    label_pad_token_id = tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        # model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
    )

    # Post-processing:
    def post_processing_function(preds, labels):
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
    
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    metric = load_metric("squad")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        if isinstance(preds, tuple):
            preds = preds[0]

        max_val_samples = len(preds)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # decoded_labels은 rouge metric을 위한 것이며, f1/em을 구할 때 사용되지 않음
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 간단한 post-processing
        decoded_preds, decoded_labels = post_processing_function(decoded_preds, decoded_labels)

        print(decoded_preds, decoded_labels, sep='\n')

        formatted_predictions = [{"id": ex["id"], "prediction_text": decoded_preds[i]} for i, ex in enumerate(datasets["validation"].select(range(max_val_samples)))]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"].select(range(max_val_samples))]

        result = metric.compute(predictions=formatted_predictions, references=references)
        print(result)
        return result

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        print(train_result)

        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        print(trainer)

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # State 저장
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_target_length, num_beams=4, metric_key_prefix="eval")

        print(metrics)

        metrics["eval_samples"] = len(eval_dataset)



if __name__ == "__main__":
    main()
