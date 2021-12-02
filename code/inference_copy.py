"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""


import logging
import sys
from typing import Callable, List, Dict, NoReturn, Tuple

import numpy as np
from configure import *
from preprocess import *
from datasets import (
    load_metric,
    load_from_disk,
    Sequence,
    Value,
    Features,
    Dataset,
    DatasetDict,
)

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer

from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from utils_qa import postprocess_qa_predictions, check_no_error
from trainer_qa import QuestionAnsweringTrainer
from retrieval import SparseRetrieval, DenseRetrieval

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)


logger = logging.getLogger(__name__)


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.do_train = True

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    model,tokenizer =cofngiure_model(model_args,training_args ,  data_args)

    # True일 경우 : run passage retrieval
    # if data_args.eval_retrieval:
    #     datasets = run_retrieval(
    #         tokenizer,
    #         datasets,
    #         training_args,
    #         data_args,
    #     )


    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "../data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.

    # Spase Passage Retrieval 부분 
    retriever_sparse = SparseRetrieval(
        tokenize_fn=tokenize_fn.tokenize, data_path=data_path, context_path=context_path
        )

    if data_args.sparse_name == "None":
        retriever_sparse.get_sparse_embedding()
        if data_args.use_faiss:
            retriever_sparse.build_faiss(num_clusters=data_args.num_clusters)
            df_sparse = retriever_sparse.retrieve_faiss(
                datasets["validation"], topk=data_args.top_k_retrieval
            )
        else:
            df_sparse = retriever_sparse.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)
        
    elif data_args.sparse_name == "elastic":
        df_sparse = retriever_sparse.elastic_retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # Dense Passage Retrieval 부분
    retriever_dense = DenseRetrieval(
        tokenize_fn=tokenize_fn, datasets=datasets, data_path=data_path, context_path=context_path 
    )
    if data_args.dense_name == "None":
        retriever_dense.get_dense_embedding(inbatch=False)
        retriever_dense.build_faiss(num_clusters=data_args.num_clusters)
        df_dense = retriever_dense.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    elif data_args.dense_name == "in-batch":
        retriever_dense.get_dense_embedding(inbatch=True)
        retriever_dense.build_faiss(num_clusters=data_args.num_clusters)
        df_dense = retriever_dense.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )


    # Sparse Retrieval 결과와 Dense Retrieval 결과를 병합합니다. 
    df = df_sparse
    for idx in range(len(df_sparse)):
        # if idx == 10:
        #     print(df_dense["context"][idx])
        temp = df_sparse["context"][idx] + df_dense["context"][idx]
        df["context"][idx] = " ".join(temp)

    # Dense Retrieval 결과 일부 출력하기        
    # df = df_dense
    # for idx in range(len(df_dense)):
    #     if idx % 1000 == 0:
    #         print(df["context_id"][idx])
    #         print('-----------')
    #         print(df["question"][idx])
    #         print('-----------')
    #         print(df["context"][idx])
    #     df["context"][idx] = " ".join(df_dense["context"][idx])


    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:

    # eval 혹은 prediction에서만 사용함
    column_names = datasets["validation"].column_names

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    # Validation preprocessing / 전처리를 진행합니다.

    eval_dataset = datasets["validation"]
    prepare_validation_features = preprocess_extract_valid(tokenizer,data_args,column_names,max_seq_length)
    # Validation Feature 생성
    eval_dataset = eval_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Post-processing:
    def post_processing_function(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        training_args: TrainingArguments,
    ) -> EvalPrediction:
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]

        if training_args.do_predict:
            return formatted_predictions
        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]

            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction) -> Dict:
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    print("init trainer...")
    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    logger.info("*** Evaluate ***")

    #### eval dataset & eval example - predictions.json 생성됨
    if training_args.do_predict:
        predictions = trainer.predict(
            test_dataset=eval_dataset, test_examples=datasets["validation"]
        )

        # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
        print(
            "No metric can be presented because there is no correct answer given. Job done!"
        )

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
