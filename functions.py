import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch        
import joblib
import torch.nn as nn

# from check_lang import id_to_en
from transformers import BertTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from Preprocessing import preprocess_text
from finetuned_pipeline import *

def plot_vector_distribution(pca_result, kmeans_model, scoop_labels, new_data_pca=None, check_outscoop = False):
    kmeans_labels = kmeans_model.labels_
    centroids = kmeans_model.cluster_centers_

    df_pca = pd.DataFrame(pca_result, columns=['Dimension 1', 'Dimension 2'])
    df_pca['kmeans_labels'] = kmeans_labels
    df_pca['scoop_labels'] = scoop_labels
    df_pca['plot_color'] = kmeans_labels

    if check_outscoop:
        df_pca.loc[df_pca.scoop_labels == -1, 'plot_color'] = max(kmeans_labels) + 1

    cluster_palette = sns.color_palette('tab10', n_colors=df_pca['plot_color'].nunique() + 1)
    plt.figure(figsize=(8, 6))
    plot = sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='plot_color', data=df_pca, palette=cluster_palette)
    plot.set(xlabel = None)
    plot.set(ylabel = None)
    plot.legend([],[], frameon=False)

    # Add centroids to the plot
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='^', c='red', s=50, label='Centroids')
    
    if new_data_pca is not None:
        plt.scatter(new_data_pca[:, 0], new_data_pca[:, 1], c='green')
    # Show position of new PCA data
    
    plt.title('PCA Latent Representation with Centroids')
    plt.show()

def save_data(jurnal_id, jurnal_type, kmeans_model, threshold, pca_data):
    pass

def load_data(jurnal_id, jurnal_type):
    file_path = f"./src/{jurnal_type}/{jurnal_id}"
    filename_kmeans = f"{file_path}/{jurnal_id}_kmeans.pkl"
    kmeans_model = joblib.load(filename_kmeans)
    threshold = np.load(f"{file_path}/{jurnal_id}_threshold.npy")
    pca_data = np.load(f"{file_path}/{jurnal_id}_pca_data.npy")
    X_bert = np.load(f"{file_path}/{jurnal_id}_bert_data.npy")
    df_res = pd.read_csv(f"{file_path}/{jurnal_id}_data_jurnal.csv")
    return kmeans_model, threshold, pca_data, X_bert, df_res

def embed(dataset, model_checkpoint, model=None, max_length=128, device='cpu'):
    if not model:
        model = AutoModel.from_pretrained(model_checkpoint)

    tokenizer = BertTokenizer.from_pretrained(model_checkpoint)

    dataset = BertDataset(dataset, tokenizer)
    batch_size = 32
    data_loader = DataLoader(dataset, batch_size)

    model.to(device)
    model.eval()

    embeddings = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            last_hidden_states = outputs.last_hidden_state
            embeddings.append(last_hidden_states.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)

    return embeddings

def centroid_dist(kmeans_model):
    