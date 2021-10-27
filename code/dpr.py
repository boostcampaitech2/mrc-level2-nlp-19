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

from transformers import (
    AutoTokenizer,
    BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
)

from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn.functional as F


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


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


class DenseRetrieval:
    def __init__(
        self,
        args,
        dataset,
        num_neg,
        tokenizer,
        p_encoder,
        q_encoder,
    ) -> NoReturn:
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
        self.num_neg = num_neg

        self.tokenizer = tokenizer
        self.p_encoder = p_encoder.to(args.device)
        self.q_encoder = q_encoder.to(args.device)

        self.p_embedding = None  # get_dense_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.

        self.prepare_in_batch_negative(num_neg=num_neg)

    def prepare_in_batch_negative(
        self,
        dataset=None,
        num_neg=2,
        tokenizer=None
    ):
        print("----------dense.py prepare_in_batch_negative start----------")
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

        # Negative sampling 을 위한 negative sample 들을 샘플링
        # 주어진 query/question 에 해당하지 않는 지문들을 뽑아서 훈련데이터로 넣어줍시다.

        # 1. In-Batch-Negative 만들기
        # CORPUS를 np.array로 변환해줍니다.
        corpus = np.array(list(set([example for example in dataset["context"]])))
        p_with_neg = []

        for c in dataset["context"]:
            while True:
                neg_idxs = np.random.randint(len(corpus), size=num_neg)

                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break

        # 처리한 데이터를 torch가 처리할 수 있게 DataLoader로 넘겨주는 작업을 해봅시다.
        # 기본적으로 Huggingface Pretrained 모델이 input_ids, attention_mask, token_type_ids를 받아주니, 이 3가지를 넣어주도록 합시다.
        
        # 2. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(
            dataset["question"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        p_seqs = tokenizer(
            p_with_neg,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        max_len = p_seqs["input_ids"].size(-1)
        p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_neg+1, max_len)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(-1, num_neg+1, max_len)
        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(-1, num_neg+1, max_len)

        train_dataset = TensorDataset(
            p_seqs["input_ids"], p_seqs["attention_mask"], p_seqs["token_type_ids"], 
            q_seqs["input_ids"], q_seqs["attention_mask"], q_seqs["token_type_ids"]
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.args.per_device_train_batch_size
        )

        valid_seqs = tokenizer(
            dataset["context"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        passage_dataset = TensorDataset(
            valid_seqs["input_ids"],
            valid_seqs["attention_mask"],
            valid_seqs["token_type_ids"]
        )
        '''
        [DataLoader]
        Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.

        The DataLoader supports both map-style and iterable-style datasets with single- or multi-process loading,
        customizing loading order and optional automatic batching (collation) and memory pinning.

        See torch.utils.data documentation page for more details.
        '''
        self.passage_dataloader = DataLoader(
            passage_dataset,
            batch_size=self.args.per_device_train_batch_size
        )

        print("==========dense.py prepare_in_batch_negative end==========")

    # train 함수를 정의한 후 p_encoder과 q_encoder를 학습시켜봅시다.
    def train(
        self,
        args=None
    ):
        print("----------dense.py train start----------")
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

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        # for _ in range(int(args.num_train_epochs)):
        for _ in train_iterator:

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    p_encoder.train()
                    q_encoder.train()
            
                    targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    p_inputs = {
                        "input_ids": batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        "attention_mask": batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        "token_type_ids": batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device)
                    }
            
                    q_inputs = {
                        "input_ids": batch[3].to(args.device),
                        "attention_mask": batch[4].to(args.device),
                        "token_type_ids": batch[5].to(args.device)
                    }

                    # (batch_size*(num_neg+1), emb_dim)
                    p_outputs = self.p_encoder(**p_inputs)
                    # (batch_size*, emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(batch_size, -1, self.num_neg+1)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = torch.bmm(q_outputs, p_outputs).squeeze()  #(batch_size, num_neg + 1)
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f"{str(loss.item())}")

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs
        print("==========dense.py train end==========")

    def get_relevant_doc(
        self,
        query,
        k=1,
        args=None,
        p_encoder=None,
        q_encoder=None
    ):
        print("----------dense.py get_relevant_doc start----------")
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

            # passage embedding
            p_embs = []
            for batch in self.passage_dataloader:

                batch = tuple(t.to(args.device) for t in batch)
                p_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2]
                }
                p_emb = p_encoder(**p_inputs).to("cpu")
                p_embs.append(p_emb)

        # (num_passage, emb_dim)
        p_embs = torch.stack(
            p_embs, dim=0
        ).view(len(self.passage_dataloader.dataset), -1)
        

        # Dot product를 통해 유사도 구하기
        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()

        # print(2222222222222222222222222222222, p_embs, 33333333333333333333, dot_prod_scores, 444444444444444444, rank)

        """
        tensor([[ 0.7563,  0.9276,  0.9635,  ...,  0.5535,  0.4253,  0.9996],
        [ 0.3558,  0.7835,  0.9392,  ...,  0.1651,  0.3144,  0.9997],
        [ 0.7292,  0.9224,  0.9527,  ..., -0.4599,  0.7758,  1.0000],
        ...,
        [ 0.6099,  0.8473,  0.9433,  ...,  0.2389, -0.4142,  0.9938],
        [ 0.7342,  0.9203,  0.9741,  ..., -0.0439,  0.5146,  0.9999],
        [ 0.5005,  0.7715,  0.7126,  ...,  0.5556,  0.0992,  0.9993]])
        """

        """
        tensor([[-9.7220e+01, -1.0050e+02, -1.1506e+02, -1.1577e+02, -7.9811e+01,
         -1.3093e+02, -8.0151e+01, -4.8706e+01, -1.2993e+02, -8.0824e+01],
        [ 1.6180e+01,  3.4977e+01,  1.6717e+01,  2.8833e+01,  3.4405e+01,
          1.3194e+01,  2.6219e+01,  4.3385e+01,  8.6389e+00,  3.1599e+01],
        [ 3.9723e+01,  5.9733e+01,  5.9473e+01,  5.8348e+01,  6.2999e+01,
          5.3824e+01,  5.1889e+01,  5.8070e+01,  3.6547e+01,  4.8369e+01],
        [-6.2038e+00, -1.6204e+00, -1.0324e+01, -1.4898e+01,  1.1068e+01,
         -1.7359e+01,  5.7232e+00,  2.3650e+01, -2.5400e+01,  3.4510e-02],
        [-6.6613e+01, -5.6376e+01, -7.0632e+01, -6.8005e+01, -4.8165e+01,
         -7.9410e+01, -5.0374e+01, -2.0582e+01, -8.6535e+01, -5.4668e+01],
        [-6.6542e+01, -5.3435e+01, -7.2674e+01, -6.0692e+01, -4.5914e+01,
         -7.5407e+01, -5.6701e+01, -3.0053e+01, -8.6252e+01, -4.4623e+01],
        [-2.0133e+01, -1.6770e+01, -2.9859e+01, -2.1076e+01, -8.7047e+00,
         -3.4903e+01, -1.1651e+01,  6.9568e+00, -4.0030e+01, -9.4552e+00],
        [ 1.1997e+01,  2.6791e+01,  1.5369e+01,  1.7148e+01,  3.3559e+01,
          1.2185e+01,  2.7093e+01,  3.4059e+01,  3.9190e-01,  3.0677e+01],
        [-5.8074e+01, -5.0264e+01, -6.1541e+01, -5.9632e+01, -3.5633e+01,
         -6.8517e+01, -4.5823e+01, -1.4994e+01, -7.9000e+01, -4.4646e+01],
        [-7.1516e+01, -6.4831e+01, -8.3239e+01, -7.5139e+01, -5.8612e+01,
         -8.8835e+01, -6.3805e+01, -3.2248e+01, -9.8506e+01, -5.5905e+01]])
        """

        """
        tensor([[7, 4, 6, 9, 0, 1, 2, 3, 8, 5],
        [7, 1, 4, 9, 3, 6, 2, 0, 5, 8],
        [4, 1, 2, 3, 7, 5, 6, 9, 0, 8],
        [7, 4, 6, 9, 1, 0, 2, 3, 5, 8],
        [7, 4, 6, 9, 1, 0, 3, 2, 5, 8],
        [7, 9, 4, 1, 6, 3, 0, 2, 5, 8],
        [7, 4, 9, 6, 1, 0, 3, 2, 5, 8],
        [7, 4, 9, 6, 1, 3, 2, 5, 0, 8],
        [7, 4, 9, 6, 1, 0, 3, 2, 5, 8],
        [7, 9, 4, 6, 1, 0, 3, 2, 5, 8]])
        """

        print("==========dense.py get_relevant_doc end==========")

        return rank[:k]


    def get_dense_embedding(self) -> NoReturn:

        print("----------dense.py get_dense_embedding start----------")

        """
        Summary:
            Passage Embedding을 만들고
            Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        pickle_name = f"dense_embedding.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)

            print(66666666666666666, self.p_embedding.shape)
            
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            print("Embedding pickle saved.")

        print("==========dense.py get_dense_embedding end==========")

    def build_faiss(self, num_clusters=64) -> NoReturn:

        print("----------dense.py build_faiss start----------")

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

        print("==========dense.py build_faiss end==========")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        print("----------dense.py retrieve start----------")

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

        # assert self.p_embedding is not None, "get_dense_embedding() 메소드를 먼저 수행해줘야합니다."

        '''
        print(5555555555555555, query_or_dataset)
        Dataset({
                    features: ['__index_level_0__', 'answers', 'context', 'document_id', 'id', 'question', 'title'],
                    num_rows: 4192
        })
        '''

        # if isinstance(query_or_dataset, str):
        #     doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
        #     print("[Search query]\n", query_or_dataset, "\n")

        #     for i in range(topk):
        #         print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
        #         print(self.contexts[doc_indices[i]])

        #     print("==========dense.py retrieve str end==========")
        #     return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        # elif isinstance(query_or_dataset, Dataset):

        # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
        total = []
        with timer("query exhaustive search"):
            doc_scores, doc_indices = self.get_relevant_doc(
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
                    [self.contexts[pid] for pid in doc_indices[idx]]
                ),
            }
            if "context" in example.keys() and "answers" in example.keys():
                # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                tmp["original_context"] = example["context"]
                tmp["answers"] = example["answers"]
            total.append(tmp)

        cqas = pd.DataFrame(total)
        print("==========dense.py retrieve Dataset end==========")
        return cqas

    # def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

    #     """
    #     Arguments:
    #         query (str):
    #             하나의 Query를 받습니다.
    #         k (Optional[int]): 1
    #             상위 몇 개의 Passage를 반환할지 정합니다.
    #     Note:
    #         vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
    #     """

    #     with timer("transform"):
    #         query_vec = self.tfidfv.transform([query])
    #     assert (
    #         np.sum(query_vec) != 0
    #     ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

    #     with timer("query ex search"):
    #         result = query_vec * self.p_embedding.T
    #     if not isinstance(result, np.ndarray):
    #         result = result.toarray()

    #     sorted_result = np.argsort(result.squeeze())[::-1]
    #     doc_score = result.squeeze()[sorted_result].tolist()[:k]
    #     doc_indices = sorted_result.tolist()[:k]
    #     return doc_score, doc_indices

    # def get_relevant_doc_bulk(
    #     self, queries: List, k: Optional[int] = 1
    # ) -> Tuple[List, List]:
    #     print("----------dense.py get_relevant_doc_bulk start----------")

    #     """
    #     Arguments:
    #         queries (List):
    #             하나의 Query를 받습니다.
    #         k (Optional[int]): 1
    #             상위 몇 개의 Passage를 반환할지 정합니다.
    #     Note:
    #         vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
    #     """

    #     query_vec = self.tfidfv.transform(queries)
    #     assert (
    #         np.sum(query_vec) != 0
    #     ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

    #     result = query_vec * self.p_embedding.T
    #     if not isinstance(result, np.ndarray):
    #         result = result.toarray()
    #     doc_scores = []
    #     doc_indices = []
    #     for i in range(result.shape[0]):
    #         sorted_result = np.argsort(result[i, :])[::-1]
    #         doc_scores.append(result[i, :][sorted_result].tolist()[:k])
    #         doc_indices.append(sorted_result.tolist()[:k])

    #     print("==========dense.py get_relevant_doc_bulk end==========")

    #     return doc_scores, doc_indices

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


if __name__ == "__main__":
    print("----------dense.py main start----------")

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", default="../data/train_dataset", metavar="./data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-multilingual-cased",
        metavar="bert-base-multilingual-cased",
        # default="klue/roberta-large",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", default="../data", metavar="./data", type=str, help="")
    parser.add_argument(
        "--context_path", default="wikipedia_documents.json", metavar="wikipedia_documents", type=str, help=""
    )

    parser.add_argument("--use_faiss", default=False, metavar=False, type=bool, help="")

    args = parser.parse_args()
    # print(11111111111111111111111, args)
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

    train_args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=4e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        weight_decay=0.01
    )

    model_checkpoint = "klue/bert-base"

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    p_encoder = BertEncoder.from_pretrained(model_checkpoint).to(train_args.device)
    q_encoder = BertEncoder.from_pretrained(model_checkpoint).to(train_args.device)

    # 데이터셋과 모델은 아래와 같이 불러옵니다.
    train_dataset = org_dataset["train"].flatten_indices()
    
    # # 메모리가 부족한 경우 일부만 사용하세요 !
    num_sample = 10
    sample_idx = np.random.choice(range(len(train_dataset)), num_sample)
    train_dataset = full_ds[sample_idx]

    retriever = DenseRetrieval(
        args=train_args,
        dataset=train_dataset,
        num_neg=2,
        tokenizer=tokenizer,
        p_encoder=p_encoder,
        q_encoder=q_encoder
    )

    retriever.train()

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    if args.use_faiss:

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

            train_dataset = org_dataset["train"].flatten_indices()
    
            # # 메모리가 부족한 경우 일부만 사용하세요 !
            num_sample = 10
            sample_idx = np.random.choice(range(len(train_dataset)), num_sample)
            train_dataset = full_ds[sample_idx]

            df = retriever.retrieve(train_dataset)
            df["correct"] = df["original_context"] == df["context"]
            print(
                "correct retrieval result by exhaustive search",
                df["correct"].sum() / len(df),
            )

        with timer("single query by exhaustive search"):
            scores, indices = retriever.retrieve(query)
    
    print("==========dense.py main end==========")
