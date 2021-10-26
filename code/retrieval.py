import os
import json
import time
import faiss
import pickle
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union


from sklearn.feature_extraction.text import TfidfVectorizer

from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from transformers import (
    AutoTokenizer,
    BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
    training_args,
)
# import faiss.contrib.torch_utils

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class SparseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:

        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # Transform by vectorizer
        self.tfidfv = TfidfVectorizer(
            tokenizer=tokenize_fn,
            ngram_range=(1, 2),
            max_features=50000,
        )

        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.

    def get_sparse_embedding(self) -> NoReturn:

        """
        Summary:
            Passage Embedding을 만들고
            TFIDF와 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        pickle_name = f"sparse_embedding.bin"
        tfidfv_name = f"tfidv.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)

        if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.tfidfv = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)
            print(self.p_embedding.shape)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
            print("Embedding pickle saved.")

    def build_faiss(self, num_clusters=64) -> NoReturn:

        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.

        Note:
            Faiss는 Build하는데 시간이 오래 걸리기 때문에,
            매번 새롭게 build하는 것은 비효율적입니다.
            그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
            다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
            제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
        """

        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            p_emb = self.p_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        with timer("transform"):
            query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        with timer("query ex search"):
            result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
            retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
        """

        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(
                    queries, k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vecs = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vecs) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()

class DenseRetrieval:
    def __init__(self,
        args,
        dataset,
        tokenizer,
        p_encoder,
        q_encoder,
        ret_args
    ):
        """
        학습과 추론에 사용될 여러 셋업을 마쳐봅시다.
        """
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.ret_args = ret_args
        self.indexer = None
        self.prepare_in_batch_negative()

    def prepare_in_batch_negative(self,
        dataset=None,
        tokenizer=None
    ):

        if dataset is None:
            dataset = self.dataset

        if tokenizer is None:
            tokenizer = self.tokenizer

        if self.ret_args.retrieve == "dense_train":
            # 1. (Question, Passage) 데이터셋 만들어주기
            q_seqs = tokenizer(
                dataset["question"],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            p_seqs = tokenizer(
                dataset["context"],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            # 2. Tensor dataset
            train_dataset = TensorDataset(
                p_seqs["input_ids"], p_seqs["attention_mask"], p_seqs["token_type_ids"], 
                q_seqs["input_ids"], q_seqs["attention_mask"], q_seqs["token_type_ids"]
            )
            
            self.train_dataloader = DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=self.args.per_device_train_batch_size
            )

        if self.ret_args.retrieve == "dense":
            print('open wiki data...')
            print()
            with open(os.path.join(self.ret_args.data_path, self.ret_args.context_path), "r", encoding="utf-8") as f:
                wiki = json.load(f)
            
            # test를 위해 일단 1000개만 실행
            self.contexts = list(
                dict.fromkeys([v["text"] for v in wiki.values()])
            )[:1000]

            print('tokenizing dataset...')
            start_time = time.time()
            valid_seqs = tokenizer(
                self.contexts,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            print(f'tokenizing done with {time.time()-start_time:.4f}s')
            self.passage_dataset = TensorDataset(
                valid_seqs["input_ids"],
                valid_seqs["attention_mask"],
                valid_seqs["token_type_ids"]
            )
            self.passage_dataloader = DataLoader(
                self.passage_dataset,
                batch_size=self.args.per_device_train_batch_size
            )

    def build_faiss(self, num_clusters=8) -> NoReturn:

        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.
        """

        indexer_name = f"faiss_clusters{num_clusters}_{self.ret_args.faiss}.index"
        indexer_path = os.path.join(self.ret_args.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

            res = faiss.StandardGpuResources()
            self.indexer = faiss.index_cpu_to_gpu(res, 0, self.indexer)

        else:
            print('Faiss indexer initiating...')
            with torch.no_grad():
                self.p_encoder.eval()

                p_embs = []
                for batch in self.passage_dataloader:

                    batch = tuple(t.cuda() for t in batch)
                    p_inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2]
                    }
                    p_emb = p_encoder(**p_inputs).to("cpu").numpy()
                    p_embs.extend(p_emb)

            p_embs = np.array(p_embs)

            print(f'p_embs after stacking : {p_embs.shape}')
            emb_dim = p_embs.shape[-1]

            if self.ret_args.faiss == "L2" :
                print('faiss for L2')
                num_clusters = num_clusters

                res = faiss.StandardGpuResources()
                quantizer = faiss.IndexFlatL2(emb_dim)

                self.indexer = faiss.IndexIVFScalarQuantizer(
                    quantizer,
                    quantizer.d,
                    num_clusters,
                    faiss.METRIC_L2
                )

                self.indexer = faiss.index_cpu_to_gpu(res, 0, self.indexer)
                self.indexer.train(p_embs)
                self.indexer.add(p_embs)
                faiss.write_index(self.indexer, indexer_path)
                print("Faiss Indexer Saved.")

            if self.ret_args.faiss == "IP" :
                print('faiss for IP')
                num_clusters = num_clusters
                res = faiss.StandardGpuResources()
                quantizer = faiss.IndexFlatIP(emb_dim)
                self.indexer = faiss.IndexIVFFlat(
                    quantizer,
                    quantizer.d,
                    num_clusters,
                    faiss.METRIC_INNER_PRODUCT
                )

                self.indexer = faiss.index_cpu_to_gpu(res, 0, self.indexer)
                self.indexer.train(p_embs)
                self.indexer.add(p_embs)
                cpu_index = faiss.index_gpu_to_cpu(self.indexer)
                faiss.write_index(cpu_index, indexer_path)
                print("Faiss Indexer Saved.")

    def train(self, args=None):
        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            {"params": [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon
        )
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total
        )

        # Start training!
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        for _ in train_iterator:

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:
                    if torch.cuda.is_available():
                        batch = tuple(t.cuda() for t in batch)
                    self.p_encoder.train()
                    self.q_encoder.train()

                    p_inputs = {'input_ids': batch[0],
                                'attention_mask': batch[1],
                                'token_type_ids': batch[2]
                                }
                    
                    q_inputs = {'input_ids': batch[3],
                                'attention_mask': batch[4],
                                'token_type_ids': batch[5]}

                    p_outputs = self.p_encoder(**p_inputs) # (batch_size, emb_dim)
                    q_outputs = self.q_encoder(**q_inputs) # (batch_size, emb_dim)

                    
                    # target position : diagonal
                    targets = torch.arange(0, args.per_device_train_batch_size).long()
                    if torch.cuda.is_available():
                        targets = targets.to('cuda')

                    # Calculate similarity score & loss
                    sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))  #(batch_size, batch_size)
                    sim_scores = F.log_softmax(sim_scores, dim=1)
                    loss = F.nll_loss(sim_scores, targets)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()
                    global_step += 1
                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        assert self.passage_dataset is not None, "passage가 load되지 않았습니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")
            doc_scores = doc_scores.squeeze()
            doc_indices = doc_indices.squeeze()
            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc(
                    query_or_dataset["question"], k=topk
                )
                print(doc_indices.shape)
                print(doc_scores.shape)
            
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx][0],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self,
        query,
        k=1,
        args=None,
        p_encoder=None,
        q_encoder=None
    ):
    
        if args is None:
            args = self.args

        if p_encoder is None:
            p_encoder = self.p_encoder

        if q_encoder is None:
            q_encoder = self.q_encoder

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            q_seqs_val = self.tokenizer(
                query,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(args.device)
            q_emb = q_encoder(**q_seqs_val).to("cpu")  # (num_query=1, emb_dim)

            p_embs = []
            for batch in self.passage_dataloader:

                batch = tuple(t.to(args.device) for t in batch)
                p_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2]
                }
                p_emb = p_encoder(**p_inputs).to("cpu").numpy()
                p_embs.extend(p_emb)
        
        p_embs = torch.Tensor(p_embs)
        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True)
        print(f'dot prod_scores shape : {dot_prod_scores.shape}')
        print(f'rank shape : {rank.shape}')
        return dot_prod_scores[:,:k], rank[:,:k]

    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        assert self.passage_dataset is not None, "passage가 load되지 않았습니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")
            print(type(doc_scores))
            print(doc_scores)

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[0][i]:4f}")

            return (doc_scores, [self.dataset[doc_indices[0][i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_faiss(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": " ".join(
                        [self.dataset['context'][pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc_faiss(self,
        query,
        k=1,
        args=None,
        p_encoder=None,
        q_encoder=None
    ):
        if args is None:
            args = self.args

        if p_encoder is None:
            p_encoder = self.p_encoder

        if q_encoder is None:
            q_encoder = self.q_encoder

        with torch.no_grad():
            q_encoder.eval()

            q_seqs_val = self.tokenizer(
                query,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to("cuda")
            q_embs = q_encoder(**q_seqs_val).to("cpu").numpy()  # (num_query=1, emb_dim)
        torch.cuda.empty_cache()
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()

class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()
      
    def forward(
            self,
            input_ids, 
            attention_mask=None,
            token_type_ids=None
        ): 

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs[1]
        return pooled_output

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_name", default="../data/train_dataset", type=str, help="")
    parser.add_argument("--model_name_or_path",default="bert-base-multilingual-cased",type=str,help="")
    parser.add_argument("--data_path", default="../data", type=str, help="")
    parser.add_argument("--context_path", default="wikipedia_documents.json", type=str, help="")
    parser.add_argument("--use_faiss", default=False, type=bool, help="")
    parser.add_argument("--faiss", default="L2", type=str, help="")
    parser.add_argument("--retrieve", default="sparse", type=str, help="")

    args = parser.parse_args()

    # Test

    #dataset
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
    )
    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    if args.retrieve == "sparse":
        retriever = SparseRetrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path=args.data_path,
            context_path=args.context_path,
        )
        retriever.get_sparse_embedding()
        if args.use_faiss:
            retriever.build_faiss()
            # test single query
            with timer("single query by faiss"):
                scores, indices = retriever.retrieve_faiss(query)

            # test bulk
            with timer("bulk query by exhaustive search"):
                df = retriever.retrieve_faiss(full_ds)
                df["correct"] = df["original_context"] == df["context"]

                print("correct retrieval result by faiss", df["correct"].sum() / len(df))

        else:
            with timer("bulk query by exhaustive search"):
                df = retriever.retrieve(full_ds)
                df["correct"] = df["original_context"] == df["context"]
                print(
                    "correct retrieval result by exhaustive search",
                    df["correct"].sum() / len(df),
                )

            with timer("single query by exhaustive search"):
                scores, indices = retriever.retrieve(query)

    if args.retrieve == "dense":
        print(f'Bert Encoder load from pretrained...{args.model_name_or_path}')
        print()
        p_encoder = BertEncoder.from_pretrained(args.model_name_or_path).cuda()
        q_encoder = BertEncoder.from_pretrained(args.model_name_or_path).cuda()
        print()
        print(f'Bert Encoder load from saved...')
        print()
        model_dict = torch.load("./dense_encoder/encoder.pth")
        p_encoder.load_state_dict(model_dict['p_encoder'])
        q_encoder.load_state_dict(model_dict['q_encoder'])

        print(f'Initialize training args...')
        print()
        training_args = TrainingArguments(
            output_dir="dense_retireval",
            evaluation_strategy="epoch",
            learning_rate=3e-4,
            per_device_train_batch_size=16,
            # per_device_eval_batch_size=2,
            # num_train_epochs=2,
            weight_decay=0.01
        )

        print('Initialize Denseretrieval...')    
        print()
        print(f'dataset length : {len(full_ds)}')
        retriever = DenseRetrieval(
            training_args,
            full_ds,
            tokenizer,
            p_encoder,
            q_encoder,
            args
        )

        if args.use_faiss:
            retriever.build_faiss()
            # test single query
            with timer("single query by faiss"):
                scores, indices = retriever.retrieve_faiss(query,3)

            # test bulk
            with timer("bulk query by exhaustive search"):
                df = retriever.retrieve_faiss(org_dataset['validation'], 2)
                df["correct"] = df["original_context"] == df["context"]

                print("correct retrieval result by faiss", df["correct"].sum() / len(df))

        else:
            with timer("single query by exhaustive search"):
                scores, indices = retriever.retrieve(query, 3)
                
            with timer("bulk query by exhaustive search"):
                df = retriever.retrieve(org_dataset['validation'], 2)
                df["correct"] = df["original_context"] == df["context"]
                print(
                    "correct retrieval result by exhaustive search",
                    df["correct"].sum() / len(df),
                )

            

    if args.retrieve == "dense_train":

        print(f'Bert Encoder from pretrained...{args.model_name_or_path}')
        print()
        p_encoder = BertEncoder.from_pretrained(args.model_name_or_path).cuda()
        q_encoder = BertEncoder.from_pretrained(args.model_name_or_path).cuda()
        
        print(f'Initialize training args...')
        print()
        training_args = TrainingArguments(
            output_dir="dense_retireval",
            evaluation_strategy="epoch",
            learning_rate=3e-4,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=2,
            weight_decay=0.01
        )

        print('Initialize Denseretrieval...')    
        print()    
        retriever = DenseRetrieval(
            training_args,
            org_dataset['train'],
            tokenizer,
            p_encoder,
            q_encoder,
            args
        )

        print('retriever training...')
        print()
        retriever.train()
        
        if not os.path.exists('./dense_encoder'):
            os.mkdir('./dense_encoder')
        torch.save({
            'p_encoder': p_encoder.state_dict(),
            'q_encoder': q_encoder.state_dict()
            }, './dense_encoder/encoder.pth')
        
        print('encoders saved done!')