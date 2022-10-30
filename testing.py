import numpy as np
import torch
from tqdm import tqdm

from first_try import model, test_dataset, device
from transformers.data.metrics import simple_accuracy
from sklearn.metrics import f1_score
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

epochs = 1

print("TESTING\n")

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
    # acc = simple_accuracy(preds, out_label_ids)
    # f1 = f1_score(y_true=out_label_ids, y_pred=preds, average="macro")
    # print("=== Eval: loss ===", sum_loss / count)
    # print("=== Eval: acc. ===", acc)
    # print("=== Eval: f1 ===", f1)