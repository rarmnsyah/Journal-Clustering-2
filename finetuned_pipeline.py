import torch
import pandas as pd
import numpy as np
import transformers
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.decomposition import PCA
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AutoModel

from Preprocessing import preprocess_text

class BertDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = 256

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['data_cleaned']
        labels = self.data.iloc[index][['label']].values.astype(int)
        encoding = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]

        # resize the tensors to the same size
        input_ids = nn.functional.pad(input_ids, (0, self.max_length - input_ids.shape[0]), value=0)
        attention_mask = nn.functional.pad(attention_mask, (0, self.max_length - attention_mask.shape[0]), value=0)
        return input_ids, attention_mask, torch.tensor(labels)
    
class BertClassifier(nn.Module):
    def __init__(self, num_labels, model_checkpoint):
        super(BertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_checkpoint)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs   
    
class BertClassifierEmbed(BertClassifier):
    def __init__(self, num_labels, model_checkpoint):
        super().__init__(num_labels, model_checkpoint)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs['last_hidden_state']
        return x 

class BertFinetuning():
    def __init__(self, dataset:pd.DataFrame, model_checkpoint:str, device, batch_size:int) -> None:
        self.dataset = dataset
        self.model_checkpoint = model_checkpoint
        self.device = device

        self.num_labels = dataset.eissn.nunique()
        self.model = BertClassifier(self.num_labels, model_checkpoint)
        self.tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
        
        self.model = self.model.to(self.device)
        self.dataset = BertDataset(self.dataset, self.tokenizer)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 2e-5)
        self.data_loader = DataLoader(self.dataset, batch_size, shuffle = True)

    def train(self, epochs:int):
        n_total_steps = len(self.data_loader)

        for epoch in range(epochs):
            for i, batch in enumerate (self.data_loader):

                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                labels = labels.view(-1)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(input_ids, attention_mask)

                # print(logits, labels.long())
                loss = self.criterion(logits, labels.long())
                loss.backward()
                self.optimizer.step()


                # if (i+1) % 100 == 0:
                print(f'epoch {epoch + 1}/ {epochs}, batch {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
                
    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)

class ScoopPredictor:
    def __init__(self, tokenizer, model, kmeans_model, threshold, pca_data, X_bert):
        self.tokenizer = tokenizer
        self.model = model
        self.kmeans_model = kmeans_model
        self.threshold = threshold
        self.pca_data = pca_data
        self.X_bert = X_bert

    def fit_new_data_to_pca(self, new_data):
        pca = PCA(n_components=self.pca_data.shape[1], random_state=0)
        pca.fit(self.X_bert)
        new_data_pca = pca.transform(new_data.reshape(1, -1))
        return new_data_pca

    def predict_scoop(self, title, abstract, translated = False):
        device = 'cpu'
        if torch.cuda.is_available() :
            device = 'cuda'

        self.model.to(device)

        processed_text = preprocess_text(title + abstract)
        if translated:
            if len(processed_text) > 5000:
                processed_text = processed_text[:5000]
            # processed_text = id_to_en(processed_text)
        # print(processed_text)
        encoded_dict = self.tokenizer.encode_plus(
            processed_text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoded_dict['input_ids'].to(device)
        attention_mask = encoded_dict['attention_mask'].to(device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            new_embedding = outputs.cpu().numpy().reshape(1, -1)

        new_data_pca = self.fit_new_data_to_pca(new_embedding)
        distance_to_centroid = np.sqrt(np.sum((new_data_pca - self.kmeans_model.cluster_centers_) ** 2, axis=1))
        prediction = "in scoop" if distance_to_centroid <= self.threshold else "out scoop"
        return prediction, new_data_pca, distance_to_centroid


    
