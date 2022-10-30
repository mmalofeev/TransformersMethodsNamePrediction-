from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline, Trainer, TrainingArguments
import torch
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from datasets_generator import get_test_data, get_training_data
from method_extractor import extract_methods_from_all_files
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

model = RobertaForMaskedLM.from_pretrained('huggingface/CodeBERTa-small-v1')
tokenizer = RobertaTokenizer.from_pretrained('huggingface/CodeBERTa-small-v1')

class MethodsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)


def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=512)


def tokenize_labels(batch):
    return tokenizer(batch['label'], padding=True, truncation=True, max_length=512)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


train_dict = get_training_data()
test_dict = get_test_data()
train_dict['label'] = train_dict['label'][:1024]
train_dict['text'] = train_dict['text'][:1024]

# test_dict['label'] = test_dict['label'][:128]
# test_dict['text'] = test_dict['text'][:128]


train_dataset = Dataset.from_dict({'text': train_dict['text'], 'label': train_dict['label']})
test_dataset = Dataset.from_dict({'text': test_dict['text'], 'label': test_dict['label']})

print(train_dataset)

features = train_dataset.features
print(features)

train_not_masked_text = train_dataset['label']
train_masked_text = train_dataset['text']
train_unmasked_inputs = tokenizer(train_not_masked_text, return_tensors='pt', max_length=512, truncation=True,
                                  padding=True)
train_unmasked_inputs['labels'] = train_unmasked_inputs.input_ids.detach().clone()
train_inputs = tokenizer(train_masked_text, return_tensors='pt', max_length=512, truncation=True, padding=True)
train_inputs['labels'] = train_unmasked_inputs.input_ids.detach().clone()


test_not_masked_text = test_dataset['label']
test_masked_text = test_dataset['text']
test_unmasked_inputs = tokenizer(test_not_masked_text, return_tensors='pt', max_length=512, truncation=True,
                                 padding=True)
test_unmasked_inputs['labels'] = test_unmasked_inputs.input_ids.detach().clone()

test_inputs = tokenizer(test_masked_text, return_tensors='pt', max_length=512, truncation=True, padding=True)
test_inputs['labels'] = test_unmasked_inputs.input_ids.detach().clone()

train_dataset = MethodsDataset(train_inputs)
test_dataset = MethodsDataset(test_inputs)


loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.train()
optim = AdamW(model.parameters(), lr=3e-4)
epochs = 3

print("TRAINING")

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
# trainer.train()

# trainer.evaluate()
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

epochs = 1

print("TESTING\n")

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(test_loader, leave=True)
    sum_loss = 0
    count = 0
    for batch in loop:
        count += 1
        # initialize calculated gradients (from prev step)
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        # extract loss
        loss = outputs.loss
        sum_loss += loss.item()
        # calculate loss for every parameter that needs grad update
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
        print("avg_loss = {}", sum_loss / count)
