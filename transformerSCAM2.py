import torch 
import torch.nn as nn
import csv
import random
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader
dataset = load_dataset("wangyuancheng/discord-phishing-scam-clean")
ds = load_dataset("wangyuancheng/discord-phishing-scam")
all_texts = [row['msg_content'] for row in dataset['train']]
all_labels = [float(row['label']) for row in dataset['train']]
with open("scmas (2).csv","r",newline='',encoding="utf-8-sig") as f:
    reader=csv.DictReader(f)
    for row in reader:
        all_texts.append(row.get("message"))
        all_labels.append(float(row.get("label")))
for rowd in ds['train']:
    all_texts.append(rowd['msg_content'])
    all_labels.append(float(rowd['lable']))
with open("scam2.csv","r",newline='',encoding="utf-8-sig") as f:
    reader=csv.DictReader(f)
    for row in reader:
        all_texts.append(row.get("message"))
        all_labels.append(float(row.get("label")))
with open("discord_scam_dataset_v2.csv","r",newline='',encoding="utf-8-sig") as f:
    reader=csv.DictReader(f)
    for row in reader:
        all_texts.append(row.get("message"))
        all_labels.append(float(row.get("label")))
with open("discord_scam_dataset3.csv","r",newline='',encoding="utf-8-sig") as f:
    reader=csv.DictReader(f)
    for row in reader:
        all_texts.append(row.get("message"))
        all_labels.append(float(row.get("label")))
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')
encoded = tokenizer(
    all_texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)
input_ids = encoded['input_ids']
attention_mask = encoded['attention_mask']
labels = torch.tensor(all_labels).unsqueeze(1).float()
combined = list(zip(input_ids, attention_mask, labels))
random.shuffle(combined)
input_ids, attention_mask, labels = zip(*combined)
input_ids = torch.stack(list(input_ids))
attention_mask = torch.stack(list(attention_mask))
labels = torch.stack(list(labels))
split=int(len(input_ids) * 0.8)
input_ids_train=input_ids[:split]
input_ids_test=input_ids[split:]
labels_train=labels[:split]
labels_test=labels[split:]
mask_train = attention_mask[:split]
mask_test = attention_mask[split:]
class scamBert(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.bert=BertModel.from_pretrained('bert-base-multilingual-cased')
        self.classifier=nn.Linear(768,1)
    def forward(self, input_ids, attention_mask):
        output=self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = output.last_hidden_state[:, 0, :]
        return self.classifier(cls)
train_dataset = TensorDataset(
    input_ids_train, 
    mask_train, 
    labels_train
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
brain=scamBert()
creation=nn.BCEWithLogitsLoss()
edit=torch.optim.Adam(brain.parameters(),lr=0.0001)
for epoch in range(1):
    for i, (batch_ids, batch_mask, batch_labels) in enumerate(train_loader):
        edit.zero_grad()
        output=brain(batch_ids,batch_mask)
        loss=creation(output,batch_labels)
        loss.backward()
        edit.step()
        if i % 20 == 0:
            print(f"Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")
    if epoch % 1==0:
        print(f"epoch = {epoch+1} || loss={loss.item():.4f}")
        torch.save(brain.state_dict(),'bertscam.pth')
        print(f"brain saved sucesfuly")
brain.eval()
with torch.no_grad():
    output=brain(input_ids_test,mask_test)
    loss=creation(output,labels_test)
    predicted = (torch.sigmoid(output) > 0.5).float()
    accuracy = (predicted == labels_test).float().mean()
    print(f"Loss: {loss.item():.4f}")
    print(f"Accuracy: {accuracy.item() * 100:.2f}%")
torch.save(brain.state_dict(),'bertscam.pth')
