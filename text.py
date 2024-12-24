!pip install datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

datasets = load_dataset("json", data_files="train_pair_1w.json", split="train")
datasets

datasets = dataset.train_test_split(test_size=0.2)
datasets

import torch

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")

def process_function(examples):
  tokenizer_examples = tokenizer(examples["sentence1"], examples["sentence2"], max_length=128, truncation=True)
  tokenizer_examples["labels"] = [int(label) for label in examples["label"]]
  return tokenizer_examples

tokenizer_datasets = datasets.map(process_function, batched=True, remove_columns=datasets["train"].column_names)
tokenizer_datasets

model = AutoModelForSequenceClassification.from_pretrained("hfl/chinese-macbert-base")

!pip install evaluate
import evaluate

acc_metric = evaluate.load("accuracy")
fl_metric = evaluate.load("f1")



def eval_metric(eval_pred):
  predictions, labels = eval_predict
  predictions = predictions.argmax(axis=-1)
  acc = acc_metric.compute(predictions=predictions, references=labels)
  fl = fl_metric.compute(predictions=predictions, references=labels)
  acc.update(fl)
  return acc


train_args = TrainingArguments(
                output_dir="./cross_model",
                per_device_train_batch_size=32,
                per_device_eval_batch_size=32,
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=3,
                learning_rate=2e-5,
                weight_decay=0.01,
                metric_for_best_model="fl",
                load_best_model_at_end=True)

train_args

from transformers import DataCollatorWithPadding
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenizer_datasets["train"],
    eval_dataset=tokenizer_datasets["test"],
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=eval_metric
)

trainer.train()

trainer.evaluate(tokenizer_datasets["test"])

from transfomers import pipeline
model.config.id2label = {0: "不相似", 1: "相似"}
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)