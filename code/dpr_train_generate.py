from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
import torch
from tqdm import tqdm, trange
import torch.nn.functional as F
import pickle
import os
import numpy as np
import json
import wandb



from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
from transformers import BertModel, BertPreTrainedModel, AdamW, TrainingArguments, get_linear_schedule_with_warmup

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)

from datasets import (
    load_metric,
    load_from_disk,
    Sequence,
    Value,
    Features,
    Dataset,
    DatasetDict,
)

class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()
        
    def forward(self, input_ids, 
                attention_mask=None, token_type_ids=None): 

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        
        pooled_output = outputs[1]

        return pooled_output

def make_dataset(retriever, tokenizer, inbatch):
    datasets = load_from_disk('../data/train_dataset')
    if inbatch == False:
        # sparse embedding -> df : 각 question에 대해 topk passage의 결과를 담은 dataframe
        retriever.get_sparse_embedding()

        df_train = retriever.retrieve(datasets["train"], topk=40)
        df_val = retriever.retrieve(datasets["validation"], topk=40)
        # print(len(df))
        
        # negative sampling : context(passage_list;TF-IDF의 값이 높은 passage)에서 정답을 포함하지 않는 passage를 구하여 context값으로 지정
        
        for num, df in enumerate([df_train, df_val]):
            context_lists = []
            if num ==0:
                corpus = datasets["train"]["context"]
            else:
                corpus = datasets["validation"]["context"]

            for idx in range(len(df)):
                context_list = []
                context_list.append(df.loc[idx]["original_context"])
                for context in df.loc[idx]['context']:
                    if not df.loc[idx]['answers']['text'][0] in context:
                        context_list.append(context)
                    if len(context_list) == 5:
                        break
                if len(context_list) < 5:
                    # set number of neagative sample
                    num_neg = 5-len(context_list)
                    corpus = np.array(corpus)
                    while True:
                        neg_idxs = np.random.randint(len(corpus), size=num_neg)
                        if not df.loc[idx]["original_context"] in corpus[neg_idxs]:
                            p_neg = corpus[neg_idxs]
                            context_list.extend(p_neg)
                            break
                    df.loc[idx]['context'] = context_list
                # if idx % 1000 == 0:
                #     print(context_list)
                context_lists.extend(context_list)
            # print(len(context_lists))



            # Training Dataset 준비하기 (question, passage pairs)
            q_seqs = tokenizer(list(df['question']), padding="max_length", truncation=True, return_tensors='pt')
            p_seqs = tokenizer(context_lists, padding="max_length", truncation=True, return_tensors='pt')

            max_len = p_seqs['input_ids'].size(-1)
            p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, 5, max_len)
            p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, 5, max_len)
            p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, 5, max_len)

            # print(q_seqs['input_ids'].size())
            # print(p_seqs['input_ids'].size())
            if num == 0:
                train_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
                                    q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'])
            else:
                valid_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
                                    q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'])

        return train_dataset, valid_dataset

    else:
        # 1. (Question, Passage) 데이터셋 만들어주기
            q_seqs = tokenizer(
                datasets["question"],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            p_seqs = tokenizer(
                datasets["context"],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            # 2. Tensor dataset
            train_dataset = TensorDataset(
                p_seqs["input_ids"], p_seqs["attention_mask"], p_seqs["token_type_ids"], 
                q_seqs["input_ids"], q_seqs["attention_mask"], q_seqs["token_type_ids"]
            )
            
            return train_dataset


def train(args, num_neg, train_dataset, valid_dataset, p_model, q_model):

    # Dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)

    # Optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in p_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in p_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in q_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in q_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Start training!
    global_step = 0

    p_model.zero_grad()
    q_model.zero_grad()
    torch.cuda.empty_cache()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    config={"epochs": args.num_train_epochs, "batch_size": args.per_device_train_batch_size, "learning_rate" : args.learning_rate}
    wandb.init(project="MRCProject", config=config, name="train_encoder_8b_5e")
    for num_epochs in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        print(num_epochs)
        loss_value = 0
        matches = 0
        for step, batch in enumerate(epoch_iterator):
            p_model.train()
            p_model.train()
            
            targets = torch.zeros(args.per_device_train_batch_size).long()
            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)
                targets = targets.cuda()

            p_inputs = {'input_ids': batch[0].view(
                                            args.per_device_train_batch_size*(num_neg+1), -1),
                        'attention_mask': batch[1].view(
                                            args.per_device_train_batch_size*(num_neg+1), -1),
                        'token_type_ids': batch[2].view(
                                            args.per_device_train_batch_size*(num_neg+1), -1)
                        }
            
            q_inputs = {'input_ids': batch[3],
                        'attention_mask': batch[4],
                        'token_type_ids': batch[5]}
            
            p_outputs = p_model(**p_inputs)  #(batch_size*(num_neg+1), emb_dim)
            q_outputs = q_model(**q_inputs)  #(batch_size*, emb_dim)

            # Calculate similarity score & loss
            p_outputs = torch.transpose(p_outputs.view(args.per_device_train_batch_size, num_neg+1, -1), 1, 2)
            q_outputs = q_outputs.view(args.per_device_train_batch_size, 1, -1)

            sim_scores = torch.bmm(q_outputs, p_outputs).squeeze()  #(batch_size, num_neg+1)
            sim_scores = sim_scores.view(args.per_device_train_batch_size, -1)
            sim_scores = F.log_softmax(sim_scores, dim=1)
            preds = torch.argmax(sim_scores, dim=-1)
            
            loss = F.nll_loss(sim_scores, targets)
            loss_value += loss
            matches += (preds == targets).sum()

            loss.backward()
            optimizer.step()
            scheduler.step()
            q_model.zero_grad()
            p_model.zero_grad()
            global_step += 1
            
            torch.cuda.empty_cache()


        # 학습된 모델 저장하기
        MODEL_PATH = "./models/train_dataset"
        torch.save(p_model, os.path.join(MODEL_PATH, f"p_encoder{num_epochs}.pt"))
        torch.save(q_model, os.path.join(MODEL_PATH, f"q_encoder{num_epochs}.pt"))
        print('model_saved')
        train_loss = loss_value / len(epoch_iterator)
        train_acc = matches / len(train_dataset)
        print(
            f"Epoch {num_epochs} || training loss {train_loss:4.4} || training accuracy {train_acc:4.2%}"
        )
        wandb.log({'epoch' : num_epochs, 'training accuracy':  train_acc, 'training loss': train_loss})
        valid_epoch(q_model, p_model, valid_dataset, args.per_device_eval_batch_size, num_neg, num_epochs)

    return p_model, q_model


def valid_epoch(q_model, p_model, valid_dataset, batch_size, num_neg, num_epochs):

    valid_sampler = RandomSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=batch_size)

    with torch.no_grad():
        q_model.eval()
        p_model.eval()

        val_loss_items = []
        val_acc_items = []
        targets = torch.zeros(batch_size).long()
        for step, batch in enumerate(tqdm(valid_dataloader)):
            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)
                targets = targets.cuda()

            p_inputs = {'input_ids': batch[0].view(
                                            batch_size*(num_neg+1), -1),
                        'attention_mask': batch[1].view(
                                            batch_size*(num_neg+1), -1),
                        'token_type_ids': batch[2].view(
                                            batch_size*(num_neg+1), -1)
                        }
        
            q_inputs = {'input_ids': batch[3],
                        'attention_mask': batch[4],
                        'token_type_ids': batch[5]}
            
            p_outputs = p_model(**p_inputs)  #(batch_size*(num_neg+1), emb_dim)
            q_outputs = q_model(**q_inputs)  #(batch_size*, emb_dim)

            # Calculate similarity score & loss
            p_outputs = torch.transpose(p_outputs.view(batch_size, num_neg+1, -1), 1, 2)
            q_outputs = q_outputs.view(batch_size, 1, -1)

            sim_scores = torch.bmm(q_outputs, p_outputs).squeeze()  #(batch_size, num_neg+1)
            sim_scores = sim_scores.view(batch_size, -1)
            sim_scores = F.log_softmax(sim_scores, dim=1)
            preds = torch.argmax(sim_scores, dim=-1)

            loss = F.nll_loss(sim_scores, targets)
            acc = (targets == preds).sum().item()
            val_loss_items.append(loss)
            val_acc_items.append(acc)
            print(preds)
        # print(val_acc_items)
        val_loss = np.sum(val_loss_items) / len(valid_dataloader)
        val_acc = np.sum(val_acc_items) / len(valid_dataset)

    print(
            f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2}"
        )

    wandb.log({'epoch' : num_epochs, 'valid accuracy':  val_acc, 'valid loss': val_loss})


def train_inbatch(self, args, train_dataset, p_encoder, q_encoder):

    train_dataloader = DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=self.args.per_device_train_batch_size
            )

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
    for num_epochs in train_iterator:

        with tqdm(train_dataloader, unit="batch") as tepoch:
            for idx,batch in enumerate(tepoch):
                if torch.cuda.is_available():
                    batch = tuple(t.cuda() for t in batch)
                p_encoder.train()
                q_encoder.train()

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

        MODEL_PATH = "./models/train_dataset"
        torch.save(p_encoder, os.path.join(MODEL_PATH, f"p_encoder_in{num_epochs}.pt"))
        torch.save(q_encoder, os.path.join(MODEL_PATH, f"q_encoder_in{num_epochs}.pt"))

    return p_encoder, q_encoder


def make_dense_embedding(p_encoder, tokenizer, context):

    # Dense Embedding 적용 결과
    print("start_dense_embedding")
    with torch.no_grad():
        p_encoder.eval()

        p_embs = []
        for p in tqdm(context):
            p = tokenizer(p, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
            p_emb = p_encoder(**p).to('cpu').numpy()
            p_embs.append(p_emb)
        p_embs = torch.Tensor(p_embs).squeeze()  # (num_passage, emb_dim)

    return p_embs.to('cpu').numpy()



def run_dpr(context, tokenizer, retriever, inbatch):
    # dense embedding만 새로 생성하고자 하는 경우 활성화하기
    # q_encoder_name = f"q_encoder0.pt"
    # p_encoder_name = f"p_encoder0.pt"
    # q_model_path = os.path.join("./models/train_dataset", q_encoder_name)
    # p_model_path = os.path.join("./models/train_dataset", p_encoder_name)
    # if os.path.isfile(q_model_path):
    #     q_encoder = torch.load(q_model_path)
    #     p_encoder = torch.load(p_model_path)
    # else:
    # negative sampling한 결과
    train_dataset, valid_dataset = make_dataset(retriever, tokenizer, inbatch)
    # load pre-trained model on cuda (if available)
    p_encoder_p = BertEncoder.from_pretrained("klue/bert-base")
    q_encoder_p = BertEncoder.from_pretrained("klue/bert-base")

    # 학습 설정
    if torch.cuda.is_available():
        p_encoder_p.cuda()
        q_encoder_p.cuda()

    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=2,
        weight_decay=0.01
    )

# 학습
    if inbatch == False:
        p_encoder, q_encoder = train(args, 4, train_dataset, valid_dataset, p_encoder_p, q_encoder_p)
    else:
        p_encoder, q_encoder = train_inbatch(args, train_dataset, p_encoder_p, q_encoder_p)

    # dense embedding 결과
    p_embs = make_dense_embedding(p_encoder, tokenizer, context)
    # print(p_embs.shape)
    # dense embedding 결과 저장
    data_path = "../data/"
    if inbatch == False:
        pickle_name = f"dense_embedding.bin"
    else:
        pickle_name = f"dense_embedding_in.bin"

    emd_path = os.path.join(data_path, pickle_name)
    with open(emd_path, "wb") as file:
        pickle.dump(p_embs, file)
    return q_encoder, p_embs


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(
        "klue/bert-base",
        use_fast=True,
    )
    from retrieval import SparseRetrieval
    retriever = SparseRetrieval(tokenize_fn=tokenizer)
    # make_dataset(retriever)
    data_path = "../data/"
    context_path = "wikipedia_documents.json"

    with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
        wiki = json.load(f)

    context = list(
        dict.fromkeys([v["text"] for v in wiki.values()])
    )  # set 은 매번 순서가 바뀌므로

    run_dpr(context, retriever)


    # p_encoder_name = f"p_encoder.pt"
    # p_model_path = os.path.join("./models/train_dataset", p_encoder_name)
    # if os.path.isfile(p_model_path):
    #     p_encoder = torch.load(p_model_path)

    # tokenizer = AutoTokenizer.from_pretrained(
    #         "klue/bert-base",
    #         use_fast=True,
    #     )
    # context_path = "wikipedia_documents.json"
    # with open(os.path.join("../data", context_path), "r", encoding="utf-8") as f:
    #     wiki = json.load(f)

    # contexts = list(
    #     dict.fromkeys([v["text"] for v in wiki.values()])
    # )


    # p_embs = make_dense_embedding(p_encoder, tokenizer, contexts)
    # print(p_embs.shape)

    # pickle_name = f"dense_embedding.bin"
    # emd_path = os.path.join("/opt/ml/data", pickle_name)
    # with open(emd_path, "wb") as file:
    #     pickle.dump(p_embs, file)
    # with open(emd_path, "rb") as file:
    #     p_embedding = pickle.load(file)
    # print(p_embedding.shape)
