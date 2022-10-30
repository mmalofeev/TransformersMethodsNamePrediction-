from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline, Trainer, TrainingArguments
import torch
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from datasets_generator import get_test_data, get_training_data
from method_extractor import extract_methods_from_all_files
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

model = RobertaForMaskedLM.from_pretrained('microsoft/codebert-base-mlm')
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base-mlm')

class MethodsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # return {key: val[idx].copy().detach() for key, val in self.encodings.items()}
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


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    label_names=['labels'],
    weight_decay=0.01,
    logging_dir='./logs',
)

train_dict = get_training_data()
test_dict = get_test_data()
# normalized_length_test = len(test_dict['label']) // 16 * 16
# test_dict['label'] = test_dict['label'][:normalized_length_test]
# test_dict['text'] = test_dict['text'][:normalized_length_test]
# normalized_length_train = len(train_dict['label']) // 16 * 16
train_dict['label'] = train_dict['label'][:1024]
train_dict['text'] = train_dict['text'][:1024]


train_dataset = Dataset.from_dict({'text': train_dict['text'], 'label': train_dict['label']})
test_dataset = Dataset.from_dict({'text': test_dict['text'], 'label': test_dict['label']})


features = train_dataset.features

# fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)
# # print(train_dataset['text'][0][:512])
# print("name = ", train_dataset['label'][0])
# print(train_dataset['text'][0])
# print("ZALUPA")
# outputs = fill_mask(train_dataset['text'][0])
# print(outputs)

# train_dataset = train_dataset.map(tokenize)
# train_text = Dataset.from_dict({'text': train_dataset['text'], 'label': train_dataset['text']})
# train_label = Dataset.from_dict({'text': train_dataset['label'], 'label': train_dataset['label']})
# train_dataset = tokenizer(train_label['text'], truncation=True, padding=True, max_length=512, return_tensors="pt")
# train_dataset['label'] = train_dataset['input_ids'].clone()
# train_dataset_text = tokenizer(train_text['text'], truncation=True, padding=True, max_length=512, return_tensors="pt")
# train_dataset['input_ids'] = train_dataset_text['input_ids'].clone()
# test_text = Dataset.from_dict({'text': test_dataset['text']})
# test_label = Dataset.from_dict({'text': test_dataset['label']})
# test_dataset = tokenizer(test_label, truncation=True, padding=True, max_length=512)
# test_dataset['label'] = test_dataset['input_ids'].copy()
# test_dataset_text = tokenizer(test_text, truncation=True, padding=True, max_length=512)
# test_dataset['input_ids'] = test_dataset_text['input_ids']
train_not_masked_text = train_dataset['label']
# print(not_masked_text[0])
train_masked_text = train_dataset['text']
# print(masked_text[0])
train_unmasked_inputs = tokenizer(train_not_masked_text, return_tensors='pt', max_length=512, truncation=True,
                                  padding=True)
train_unmasked_inputs['labels'] = train_unmasked_inputs.input_ids.detach().clone()
train_inputs = tokenizer(train_masked_text, return_tensors='pt', max_length=512, truncation=True, padding=True)
train_inputs['labels'] = train_unmasked_inputs.input_ids.detach().clone()
assert(len(train_inputs['input_ids']) == len(train_inputs['labels']))
for i in range(len(train_inputs['input_ids'])):
    assert(len(train_inputs['input_ids'][i]) == len(train_inputs['labels'][i]))


test_not_masked_text = test_dataset['label']
test_masked_text = test_dataset['text']
assert(len(test_masked_text) == len(test_not_masked_text))
test_unmasked_inputs = tokenizer(test_not_masked_text, return_tensors='pt', max_length=512, truncation=True,
                                 padding=True)
test_unmasked_inputs['labels'] = test_unmasked_inputs.input_ids.detach().clone()

test_inputs = tokenizer(test_masked_text, return_tensors='pt', max_length=512, truncation=True, padding=True)
test_inputs['labels'] = test_unmasked_inputs.input_ids.detach().clone()

assert(len(test_inputs['input_ids']) == len(test_inputs['labels']))
for i in range(len(test_inputs['input_ids'])):
    print(i)
    print(len(test_inputs['input_ids'][i]))
    print(len(test_inputs['labels'][i]))
    assert(len(test_inputs['input_ids'][i]) == len(test_inputs['labels'][i]))
# print(train_inputs.keys())
# train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
# test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
# train_dataset.set_format('torch', columns=['label', 'text', 'input_ids', 'attention_mask'])
# test_dataset.set_format('torch', columns=['label', 'text', 'input_ids', 'attention_mask'])
# train_dataset = train_dataset.rename_column('label', 'labels')
train_dataset = MethodsDataset(train_inputs)
test_dataset = MethodsDataset(test_inputs)

# print(train_inputs.items())


trainer = Trainer(
    model=model,
    args=training_args,
    # compute_metrics=compute_metrics,
    train_dataset=train_inputs,
    eval_dataset=test_inputs
)

loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.train()
optim = AdamW(model.parameters(), lr=3e-4)
epochs = 3

print("TRAINING")

# for epoch in range(epochs):
#     # setup loop with TQDM and dataloader
#     loop = tqdm(loader, leave=True)
#     for batch in loop:
#         # initialize calculated gradients (from prev step)
#         optim.zero_grad()
#         # pull all tensor batches required for training
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#         # process
#         outputs = model(input_ids, attention_mask=attention_mask,
#                         labels=labels)
#         # extract loss
#         loss = outputs.loss
#         # calculate loss for every parameter that needs grad update
#         loss.backward()
#         # update parameters
#         optim.step()
#         # print relevant info to progress bar
#         loop.set_description(f'Epoch {epoch}')
#         loop.set_postfix(loss=loss.item())
# trainer.train()

# trainer.evaluate()

epochs = 1

print("TESTING\n")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(test_loader, leave=True)
    sum_loss = 0
    count = 0
    # out_label_ids = np.empty((0), dtype=np.int64)
    # preds = np.empty((0), dtype=np.int64)
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
        # logits = outputs[1]
        sum_loss += loss.item()
        # calculate loss for every parameter that needs grad update
        # print relevant info to progress bar
        # preds = np.append(preds, logits.argmax(dim=1).detach().cpu().numpy(), axis=0)
        # out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
        print("avg_loss = {}", sum_loss / count)