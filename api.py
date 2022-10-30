from datasets import Dataset
import torch
from tqdm import tqdm
from datasets_generator import get_test_data, get_training_data


class MethodsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # return {key: val[idx].copy().detach() for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def get_tokenized_datasets(tokenizer):
    train_dict = get_training_data()
    test_dict = get_test_data()

    train_dict['label'] = train_dict['label'][:1024]
    train_dict['text'] = train_dict['text'][:1024]
#    test_dict['label'] = test_dict['label'][:128]
#    test_dict['text'] = test_dict['text'][:128]

    train_dataset = Dataset.from_dict({'text': train_dict['text'], 'label': train_dict['label']})
    test_dataset = Dataset.from_dict({'text': test_dict['text'], 'label': test_dict['label']})

    train_not_masked_text = train_dataset['label']
    train_masked_text = train_dataset['text']
    train_unmasked_inputs = tokenizer(train_not_masked_text, return_tensors='pt', max_length=512, truncation=True,
                                      padding=True)
    train_unmasked_inputs['labels'] = train_unmasked_inputs.input_ids.detach().clone()
    train_inputs = tokenizer(train_masked_text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    train_inputs['labels'] = train_unmasked_inputs.input_ids.detach().clone()
    assert (len(train_inputs['input_ids']) == len(train_inputs['labels']))
    for i in range(len(train_inputs['input_ids'])):
        assert (len(train_inputs['input_ids'][i]) == len(train_inputs['labels'][i]))

    test_not_masked_text = test_dataset['label']
    test_masked_text = test_dataset['text']
    test_unmasked_inputs = tokenizer(test_not_masked_text, return_tensors='pt', max_length=512, truncation=True,
                                     padding=True)
    test_unmasked_inputs['labels'] = test_unmasked_inputs.input_ids.detach().clone()

    test_inputs = tokenizer(test_masked_text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    test_inputs['labels'] = test_unmasked_inputs.input_ids.detach().clone()

    assert (len(test_inputs['input_ids']) == len(test_inputs['labels']))
    for i in range(len(test_inputs['input_ids'])):
        assert (len(test_inputs['input_ids'][i]) == len(test_inputs['labels'][i]))
    train_dataset = MethodsDataset(train_inputs)
    test_dataset = MethodsDataset(test_inputs)
    return train_dataset, test_dataset


def train(model, dataset, epochs, optimizer, batch_size=16):
    print("TRAINING")
    model.train()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        batches = tqdm(loader, leave=True)
        for batch in batches:
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids, attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            batches.set_description(f'Epoch {epoch}')
            batches.set_postfix(loss=loss.item())


def test(model, dataset, batch_size=16):
    print("TESTING\n")
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batches = tqdm(test_loader, leave=True)
    sum_loss = 0
    count = 0
    for batch in batches:
        count += 1
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        sum_loss += loss.item()
        batches.set_description('Testing')
        batches.set_postfix(loss=loss.item())
        print("avg_loss = {}", sum_loss / count)
