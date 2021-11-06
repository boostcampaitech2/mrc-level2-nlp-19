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


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class DenseRetrieval:
    def __init__(
        self,
        args,
        dataset,
        # num_neg,
        tokenizer,
        p_encoder,
        q_encoder,
        ret_args
    ):
        """
        Arguments:
            args (Huggingface Arguments):
                세팅과 학습에 필요한 설정값을 받습니다.

            dataset (datasets.Dataset):
                Huggingface의 Dataset을 받아옵니다.

            num_neg (int):
                In-batch negative 수행시 사용할 negative sample의 수를 받아옵니다.

            tokenizer (Callable):
                Tokenize할 함수를 받아옵니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            p_encoder (torch.nn.Module):
                Passage를 Dense Representation으로 임베딩시킬 모델입니다.

            q_encoder (torhc.nn.Module):
                Query를 Dense Representation으로 임베딩시킬 모델입니다.

        Summary:
            학습과 추론에 필요한 객체들을 받아서 속성으로 저장합니다.
            객체가 instantiate될 때 in-batch negative가 생긴 데이터를 만들도록 함수를 수행합니다.
        """

        self.args = args
        self.dataset = dataset
        # self.num_neg = num_neg
        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.ret_args = ret_args
        self.indexer = None

        self.prepare_in_batch_negative()

    def prepare_in_batch_negative(
        self,
        dataset=None,
        # num_neg=2,
        tokenizer=None,
    ):
        """
        Arguments:
            dataset (datasets.Dataset, default=None):
                Huggingface의 Dataset을 받아오면,
                in-batch negative를 추가해서 Dataloader를 만들어주세요.

            num_neg (int, default=2):
                In-batch negative 수행시 사용할 negative sample의 수를 받아옵니다.
                
            tokenizer (Callable, default=None):
                Tokenize할 함수를 받아옵니다.
                별도로 받아오지 않으면 속성으로 저장된 Tokenizer를 불러올 수 있게 짜주세요.

        Note:
            모든 Arguments는 사실 이 클래스의 속성으로 보관되어 있기 때문에
            별도로 Argument를 직접 받지 않아도 수행할 수 있게 만들어주세요.
        """
        if dataset is None:
            dataset = self.dataset

        if tokenizer is None:
            tokenizer = self.tokenizer

        # with open("../data/wikipedia_documents.json", "r", encoding="utf-8") as f:
        #     wiki = json.load(f)

        # self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))

        # # Negative sampling 을 위한 negative sample 들을 샘플링
        # # 주어진 query/question 에 해당하지 않는 지문들을 뽑아서 훈련데이터로 넣어줍시다.

        # # 1. In-Batch-Negative 만들기
        # # CORPUS를 np.array로 변환해줍니다.
        # corpus = np.array(list(set([example for example in dataset["context"]])))
        # p_with_neg = []

        # for c in dataset["context"]:
        #     while True:
        #         # 0 ~ len(corpus)-1 사이의 랜덤 숫자 size만큼 뽑기
        #         neg_idxs = np.random.randint(len(corpus), size=num_neg)

        #         if not c in corpus[neg_idxs]:
        #             p_neg = corpus[neg_idxs]

        #             p_with_neg.append(c)
        #             p_with_neg.extend(p_neg)
        #             break

        # 처리한 데이터를 torch가 처리할 수 있게 DataLoader로 넘겨주는 작업을 해봅시다.
        # 기본적으로 Huggingface Pretrained 모델이 input_ids, attention_mask, token_type_ids를 받아주니, 이 3가지를 넣어주도록 합시다.
        
        # 2. (Question, Passage) 데이터셋 만들어주기
        if self.ret_args.retrieve == "dense_train":
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
            
            self.contexts = list(
                dict.fromkeys([v["text"] for v in wiki.values()])
            )
            train_corpus = list(set([example["context"] for example in self.dataset["train"]]))
            valid_corpus = list(set([example["context"] for example in self.dataset["validation"]]))
            self.contexts = self.contexts + train_corpus + valid_corpus
            print('tokenizing dataset...')
            start_time = time.time()
            if os.path.exists('passage_tokenizer.pt'):
                valid_seqs = torch.load('passage_tokenizer.pt')
            else:
                valid_seqs = tokenizer(
                    self.contexts,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                torch.save(valid_seqs, 'passage_tokenizer.pt')
                
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

    # train 함수를 정의한 후 p_encoder과 q_encoder를 학습시켜봅시다.
    def train(
        self,
        args=None
    ):
        """
        Summary:
            train을 합니다. 위에 과제에서 이용한 코드를 활용합시다.
            encoder들과 dataloader가 속성으로 저장되어있는 점에 유의해주세요.
        """
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

        # self.p_encoder.zero_grad()
        # self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        for _ in train_iterator:

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for idx,batch in enumerate(tepoch):
                    if torch.cuda.is_available():
                        batch = tuple(t.cuda() for t in batch)
                    self.p_encoder.train()
                    self.q_encoder.train()
            
                    

                    p_inputs = {
                        "input_ids": batch[0].to(args.device),
                        "attention_mask": batch[1].to(args.device),
                        "token_type_ids": batch[2].to(args.device)
                    }
            
                    q_inputs = {
                        "input_ids": batch[3].to(args.device),
                        "attention_mask": batch[4].to(args.device),
                        "token_type_ids": batch[5].to(args.device)
                    }

                    # (batch_size, emb_dim)
                    p_outputs = self.p_encoder(**p_inputs)
                    # (batch_size, emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)

                    # target position : diagonal
                    targets = torch.arange(0, args.per_device_train_batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    if torch.cuda.is_available():
                        targets = targets.to('cuda')
                    
                    # Calculate similarity score & loss
                    # p_outputs = p_outputs.view(batch_size, -1, self.num_neg+1)
                    # q_outputs = q_outputs.view(batch_size, 1, -1)

                    # sim_scores = torch.bmm(q_outputs, p_outputs).squeeze()  #(batch_size, num_neg + 1)
                    # sim_scores = sim_scores.view(batch_size, -1)

                    sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))  #(batch_size, batch_size)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    # tepoch.set_postfix(loss=f"{str(loss.item())}")
                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    if idx%10 == 0:
                        print(f'training loss : {loss:.4f}')
                    # self.p_encoder.zero_grad()
                    # self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs

    def get_relevant_doc(
        self,
        query,
        k=1,
        args=None,
        p_encoder=None,
        q_encoder=None
    ):
        """
        Arguments:
            query (str)
                문자열로 주어진 질문입니다.
            k (int, default=1)
                상위 몇 개의 유사한 passage를 뽑을 것인지 결정합니다.
            args (Huggingface Arguments, default=None)
                Configuration을 필요한 경우 넣어줍니다.
                만약 None이 들어오면 self.args를 쓰도록 짜면 좋을 것 같습니다.

        Summary:
            1. query를 받아서 embedding을 하고
            2. 전체 passage와의 유사도를 구한 후
            3. 상위 k개의 문서 index를 반환합니다.
        """
        if args is None:
            args = self.args

        if p_encoder is None:
            p_encoder = self.p_encoder

        if q_encoder is None:
            q_encoder = self.q_encoder

        # 앞서 학습한 passage encoder, question encoder 을 이용해 dense embedding 생성하기
        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            # question sequense validation 토크나이저 생성
            q_seqs_val = self.tokenizer(
                query,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(args.device)
            # question embedding
            q_emb = q_encoder(**q_seqs_val).to("cpu")  # (num_query=1, emb_dim)

            # Pickle을 저장합니다.
            pickle_name = f"dense_embedding.bin"
            emd_path = os.path.join(self.ret_args.data_path, pickle_name)

            if os.path.isfile(emd_path):
                with open(emd_path, "rb") as file:
                    self.p_embs = pickle.load(file)
                print("Embedding pickle load.")
            else:
                print("Build passage embedding")

            # passage embedding
            p_embs = []
            start_time = time.time()
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
        print(p_embs.shape)
        with open(emd_path, "wb") as file:
            pickle.dump(p_embs, file)
        print(f"Embedding pickle saved. for {time.time() - start_time:.4f}s")
        
        # Dot product를 통해 유사도 구하기
        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True)

        print(f'dot_prod_scores : {dot_prod_scores}')
        return dot_prod_scores[:,:k], rank[:,:k]

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
                cpu_index = faiss.index_gpu_to_cpu(self.indexer)
                faiss.write_index(cpu_index, indexer_path)
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
            doc_scores = doc_scores.squeeze(0)
            doc_indices = doc_indices.squeeze(0)

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:.4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            print(query_or_dataset)
            # Dense한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset["question"], k=topk)
                # doc_scores = doc_scores.squeeze(0)
                # doc_indices = doc_indices.squeeze(0)
                # print(doc_indices.shape)
                # print(doc_scores.shape)
            
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
            !!! retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
        """

        assert self.passage_dataset is not None, "passage가 load되지 않았습니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[0][i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Dense한 Passage를 pd.DataFrame으로 반환합니다.
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_faiss(query_or_dataset["question"], k=topk)
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

    def get_relevant_doc_faiss(
        self,
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
    
    
    def forward(self,
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
    parser.add_argument("--retrieve", default="dense_train", type=str, help="")
    parser.add_argument("--topk", default = 1, type=int, help="")

    args = parser.parse_args()
    # print(args)
    # Namespace(context_path=None, data_path=None, dataset_name=None, model_name_or_path=None, use_faiss=None)
    
    # Test sparse
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
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        weight_decay=0.01
    )

    print('Initialize Denseretrieval...')    
    print()
    print(f'dataset length : {len(full_ds)}')

    # 데이터셋과 모델은 아래와 같이 불러옵니다.
    # flatten_indices() : 원래 테이블의 오른쪽 행을 사용하여 새 화살표 테이블이 만듦
    train_dataset = org_dataset["validation"]
    # .flatten_indices()
    
    # # 메모리가 부족한 경우 일부만 사용하세요 !
    # num_sample = 10
    # sample_idx = np.random.choice(range(len(train_dataset)), num_sample)
    # train_dataset = full_ds[sample_idx]

    retriever = DenseRetrieval(
        training_args,
        org_dataset,
        tokenizer,
        p_encoder,
        q_encoder,
        args
    )


    retriever.train()

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    if args.use_faiss:
        retriever.build_faiss()

        # test single query
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query,args.topk)

        # test bulk
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve_faiss(org_dataset['validation'], args.topk)
            df["correct"] = df["original_context"] == df["context"]

            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    else:
        with timer("single query by exhaustive search"):
            scores, indices = retriever.retrieve(query, args.topk)
        with timer("bulk query by exhaustive search"):

            # # 메모리가 부족한 경우 일부만 사용하세요 !
            # num_sample = 2
            # sample_idx = np.random.choice(range(len(full_ds)), num_sample)
            # train_dataset = full_ds[sample_idx]
            # print(train_dataset)

            df = retriever.retrieve(org_dataset["validation"], args.topk)
            
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
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=2,
            weight_decay=0.01
        )
        print('Initialize Denseretrieval...')    
        print()    
        retriever = DenseRetrieval(
            training_args,
            full_ds,
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
