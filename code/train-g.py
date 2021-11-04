import logging
import os
import sys


from typing import List, Callable, NoReturn, NewType, Any
import dataclasses
from datasets import load_metric, load_from_disk, Dataset, DatasetDict
from kobert_tokenizer import KoBERTTokenizer
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer, BartForConditionalGeneration , AutoModelForSeq2SeqLM

import nltk
nltk.download('punkt')
from transformers import (
    Trainer,
    DataCollatorWithPadding,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    BertTokenizer,
    BertConfig, 
    EncoderDecoderConfig, 
    EncoderDecoderModel,
    GPT2Config,
    GPT2Tokenizer,
    PreTrainedTokenizerFast
)

from tokenizers import Tokenizer
from tokenizers.models import WordPiece

from utils_qa import postprocess_qa_predictions, check_no_error
from trainer_qa import QuestionAnsweringTrainer
from retrieval import SparseRetrieval

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)

from collections import Counter

import time
import datetime
import pytz

import wandb

runn = wandb.init(project="MRC_19_1", entity="woongjoon" , tags = ["klue/bert-base * 2", "hf_tokenizer"])

# wandb.init(project="MRC_19_1", 
#            name="bertgpt2",
#            tags=["bert", "gpt2"],
#            group="bert")

logger = logging.getLogger(__name__)

# count = 0
def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # model_args.model_name_or_path = get_pytorch_kobart_model()
    print(model_args)
    
    
    print(model_args.model_name_or_path)
    
    # [참고] argument를 manual하게 수정하고 싶은 경우에 아래와 같은 방식을 사용할 수 있습니다
    # training_args.per_device_train_batch_size = 4
    # print(training_args.per_device_train_batch_size)
    # return
    num_epochs=2
    training_args = Seq2SeqTrainingArguments(
        output_dir=training_args.output_dir, 
        do_train=True, 
        do_eval=True, 
        predict_with_generate=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        num_train_epochs=num_epochs,
        logging_dir='./logs',
        logging_steps=100,
        
        evaluation_strategy = 'steps' , 
        eval_steps = 300,
        # save_strategy='epoch',
        report_to="wandb",
        fp16 = True,
        # generation_num_beams=10,
        save_total_limit=2 # 모델 checkpoint를 최대 몇개 저장할지 설정
    )
    # print(f"model is from {model_args.model_name_or_path}")

    # print(f"model is from {encoder_path, decoder_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)

    
    if data_args.run_seq2seq  :
        model_args.model_name_or_path = 'google/mt5-small'
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            # from_tf=bool(".ckpt" in model_args.model_name_or_path),
            # config=config,
        )
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path
            if model_args.config_name is not None
            else model_args.model_name_or_path,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path
            if model_args.tokenizer_name is not None
            else model_args.model_name_or_path,
            use_fast=True,
        )
        print('seq2seq')
    else:
        model_name='klue/bert-base'
        tokenizer = AutoTokenizer.from_pretrained(model_name)


        model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)
        print('ed')
    print(datasets)

    print("----")
    print(training_args)
    print("----")
    print(data_args)
    
    

    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size  if not data_args.run_seq2seq   else model.config.vocab_size
    training_args.do_train=True
    training_args.do_eval=True
    print(
        type(training_args),
        type(model_args),
        type(datasets),
        type(tokenizer),
        type(model),
    )
    wandb.watch(model, log_freq=100)

    # print('end')
    # exit()
    # do_train mrc model 혹은 do_eval mrc model
    if training_args.do_train or training_args.do_eval:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
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
    # print(column_names)
    # return
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]
    
    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    pad_on_right = tokenizer.padding_side == "right"
    
    
    # 오류가 있는지 확인합니다.
    # last_checkpoint, max_seq_length = check_no_error(
    #     data_args, training_args, datasets, tokenizer
    # )
    print(data_args.pad_to_max_length)
    # data_args.pad_to_max_length=True
    print(tokenizer)
    max_seq_length = data_args.max_seq_length
    max_length = data_args.max_answer_length
    preprocessing_num_workers=12
    # padding=True
    # Train preprocessing / 전처리를 진행합니다.
    def preprocess_function(examples):

        inputs = [f"question: {q}  context: {c} <SEP>" for q, c in zip(examples["question"], examples["context"])]
        targets = [f'{a["text"][0]} <SEP>' for a in examples['answers']]
        # padding=T
        data_args.pad_to_max_length=True
        model_inputs = tokenizer(
            inputs,
            # truncation="only_second" if pad_on_right else "only_first",
            truncation=True,
            max_length=max_seq_length,
            # stride=data_args.doc_stride,
            # return_overflowing_tokens=True,
            return_overflowing_tokens=False,
            # return_offsets_mapping=True,
            return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length" if data_args.pad_to_max_length else False,
            # return_tensors="pt"
        )
        # targets(label)을 위해 tokenizer 설정
        # print("padding: {0}".format(padding))
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                truncation=True,
                # truncation="only_second" if pad_on_right else "only_first",
                max_length=max_length,
                # stride=data_args.doc_stride,
                # return_overflowing_tokens=True,
                return_overflowing_tokens=False,
                # return_offsets_mapping=True,
                return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
                padding="max_length" if data_args.pad_to_max_length else False,
                # return_tensors="pt"
                )

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["example_id"] = []
        model_inputs["decoder_input_ids"]= labels["input_ids"].copy()
        for i in range(len(model_inputs["labels"])):
            model_inputs["example_id"].append(examples["id"][i])
        # print("model_inputs: " + model_inputs.keys())
        # for m in model_inputs.keys():
        #     print(m + ": " + str(type(model_inputs[m])) )
        return model_inputs


    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]


        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=False,
        )




    if training_args.do_eval:
        eval_dataset = datasets["validation"]


        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=False,
        )
    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Post-processing:
    def postprocess_text(preds, labels):
        """
        postprocess는 nltk를 이용합니다.
        Huggingface의 TemplateProcessing을 사용하여
        정규표현식 기반으로 postprocess를 진행할 수 있지만
        해당 미션에서는 nltk를 이용하여 간단한 후처리를 진행합니다
        """
        print("------------------------------")
        print("\n"*5)

        print("postprocess_text")
        print("------------------------------")
        
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        
        print(preds)
        print(labels)
        
        preds = ["\n".join(tokenizer.tokenize(pred)) for pred in preds]
        labels = ["\n".join(tokenizer.tokenize(label)) for label in labels]

        return preds, labels
    metric = load_metric("squad")
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        max_val_samples = 16
        print("------------------------------")
        print("\n"*5)
        print(eval_preds)
        print("compute_metrics")
        print("------------------------------")
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # decoded_labels은 rouge metric을 위한 것이며, f1/em을 구할 때 사용되지 않음
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 간단한 post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        formatted_predictions = [{"id": ex["id"], "prediction_text": decoded_preds[i]} for i, ex in         
                                 enumerate(datasets["validation"].select(range(max_val_samples)))]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"].select(range(max_val_samples))]

        result = metric.compute(predictions=formatted_predictions, references=references)
        print(result)
        return result
    
    num_train_epochs=2
    batch_size=training_args.per_device_train_batch_size
    

    trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            
        )
    # Training
    if training_args.do_train:
        # if last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        # elif os.path.isdir(model_args.model_name_or_path):
        #     checkpoint = model_args.model_name_or_path
        # else:
        #     checkpoint = None
        # checkpoint
        # print("checkpoint:  "+checkpoint)
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

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
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
