import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

class Talk2CarDataset(Dataset):
def __init__(self, csv_file):
self.data = pd.read_csv(csv_file)

def __len__(self):
return len(self.data)

def __getitem__(self, idx):
item = self.data.iloc[idx]
image = np.load(item['image_path'])
command = item['command']
return image, command

def train_model(model, dataloader, epochs=10):
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
for images, commands in dataloader:
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, commands)
loss.backward()
optimizer.step()
print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Example usage
dataset = Talk2CarDataset('path_to_talk2car_data.csv')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Assuming you have a pre-defined model
# model = YourModel()
# train_model(model, dataloader)
