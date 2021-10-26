from retrieval import SparseRetrieval
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
import torch
from tqdm import tqdm, trange
import torch.nn.functional as F

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



def train(args, num_neg, dataset, p_model, q_model):
  
  # Dataloader
  train_sampler = RandomSampler(dataset)
  train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)

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

  for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")

    for step, batch in enumerate(epoch_iterator):
      q_encoder.train()
      p_encoder.train()
      
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
      p_outputs = p_outputs.view(args.per_device_train_batch_size, -1, num_neg+1)
      q_outputs = q_outputs.view(args.per_device_train_batch_size, 1, -1)

      sim_scores = torch.bmm(q_outputs, p_outputs).squeeze()  #(batch_size, num_neg+1)
      sim_scores = sim_scores.view(args.per_device_train_batch_size, -1)
      sim_scores = F.log_softmax(sim_scores, dim=1)

      loss = F.nll_loss(sim_scores, targets)
      print(loss)

      loss.backward()
      optimizer.step()
      scheduler.step()
      q_model.zero_grad()
      p_model.zero_grad()
      global_step += 1
      
      torch.cuda.empty_cache()
    
  return p_model, q_model




# parser = HfArgumentParser(
#         (ModelArguments, DataTrainingArguments, TrainingArguments)
#     )
# model_args, data_args, training_args = parser.parse_args_into_dataclasses()


# datasets = load_from_disk(data_args.dataset_name)

# tokenizer = AutoTokenizer.from_pretrained(
#         model_args.tokenizer_name
#         if model_args.tokenizer_name
#         else model_args.model_name_or_path,
#         use_fast=True,
#     )

datasets = load_from_disk("../data/train_dataset")

tokenizer = AutoTokenizer.from_pretrained(
        "klue/bert-base",
        use_fast=True,
    )


# sparse embedding -> df : 각 question에 대해 topk passage의 결과를 담은 dataframe
retriever = SparseRetrieval(
        tokenize_fn=tokenizer, data_path="../data", context_path="wikipedia_documents.json"
    )
retriever.get_sparse_embedding()

df = retriever.retrieve(datasets["train"], topk=5)

# print(df)


# negative sampling : context(passage_list;TF-IDF의 값이 높은 passage)에서 정답을 포함하지 않는 passage를 구하여 context값으로 지정
for idx, example in enumerate(df):
    context_list = []
    for context in example['context']:
        if not example['answer']['text'][0] in context:
            context_list.append(context)
    example['context'] = context_list            
    

# Training Dataset 준비하기 (question, passage pairs)
q_seqs = tokenizer(df['question'], padding="max_length", truncation=True, return_tensors='pt')
p_seqs = tokenizer(df['context'], padding="max_length", truncation=True, return_tensors='pt')


print('[Positive context]')
print(p_seqs[0][0], '\n')
print('[Negative context]')
print(p_seqs[0][1], '\n', p_seqs[0][2])


train_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
                        q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'])





# load pre-trained model on cuda (if available)
p_encoder = BertEncoder.from_pretrained("klue/bert-base")
q_encoder = BertEncoder.from_pretrained("klue/bert-base")

if torch.cuda.is_available():
  p_encoder.cuda()
  q_encoder.cuda()



args = TrainingArguments(
    output_dir="dense_retireval",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.01
)

p_encoder, q_encoder = train(args, 5, train_dataset, p_encoder, q_encoder)

# 모델 저장하기
p_encoder.save_model()
q_encoder.save_model()


