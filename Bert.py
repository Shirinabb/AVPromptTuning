import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report
commands = [
"Turn left at the next intersection",
"Call emergency services",
"Increase the volume",
"What's the weather like?"
]
labels = [0, 1, 2, 3, 4]  # Example labels for "Traffic Management", "Entertainment Management", "Vehicles Routing","Emergency" ,"Parking Management"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and encode sequences
inputs = tokenizer(commands, return_tensors='pt', padding=True, truncation=True, max_length=128)
input_ids = inputs['input_ids']
attention_masks = inputs['attention_mask']
labels = torch.tensor(labels)
dataset = TensorDataset(input_ids, attention_masks, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
optimizer = AdamW(model.parameters(), lr=1e-5)
model.train()
for epoch in range(3):  # Number of training epochs
    {
      foreach batch_size in train_dataloader:
       {
          b_input_ids, attention_masks, b_labels = batch
optimizer.zero_grad()
outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
loss = outputs.loss
loss.backward()
optimizer.step()
print(f"Epoch {epoch+1} completed")
model.eval()
predictions, true_labels = [], []
    }
    }
with torch.no_grad():
for batch in val_dataloader:
    {
b_input_ids, b_attention_mask, b_labels = batch
outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask)
logits = outputs.logits
preds = torch.argmax(logits, dim=1).flatten()
predictions.extend(preds)
true_labels.extend(b_labels)
    }
print(classification_report(true_labels, predictions, target_names=["Traffic Management", "emergency", "Entertainment Management", "Vehicle Routing", "Parking Management"]))
# Example labels for "Traffic Management", "Entertainment Management", "Vehicles Routing", "Parking Management", "emergency"