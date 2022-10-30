from torch.optim import AdamW
from transformers import RobertaTokenizer, RobertaForMaskedLM
from api import get_tokenized_datasets, train, test

model = RobertaForMaskedLM.from_pretrained('huggingface/CodeBERTa-small-v1')
tokenizer = RobertaTokenizer.from_pretrained('huggingface/CodeBERTa-small-v1')
optimizer = AdamW(model.parameters(), lr=3e-4)
train_dataset, test_dataset = get_tokenized_datasets(tokenizer)
train(model, train_dataset, epochs=3, optimizer=optimizer)
test(model, test_dataset)
