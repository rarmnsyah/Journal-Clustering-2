import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch        
import joblib
import os

# from check_lang import id_to_en
from transformers import BertTokenizer, AutoModel

from Preprocessing import preprocess_text
from finetuned_pipeline import *

# check_outscoop : default (False), warna plot berdasarkan label clustering, (True) warna untuk outscoop ditambah, ('focus') warna fokus untuk outscoop sama inscoop
def plot_vector_distribution(pca_result, kmeans_model, scoop_labels, new_data_pca = None, check_outscoop = False):
    kmeans_labels = kmeans_model.labels_
    centroids = kmeans_model.cluster_centers_

    df_pca = pd.DataFrame(pca_result, columns=['Dimension 1', 'Dimension 2'])
    df_pca['kmeans_labels'] = kmeans_labels
    df_pca['scoop_labels'] = scoop_labels
    df_pca['plot_color'] = kmeans_labels

    if check_outscoop == 'focus':
        df_pca['plot_color'] = df_pca['scoop_labels']
    elif check_outscoop:
        df_pca.loc[df_pca.scoop_labels == -1, 'plot_color'] = -1

    cluster_palette = sns.color_palette('hls', n_colors=df_pca['plot_color'].nunique() + 1)
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


def save_data(journal_id, journal_type, kmeans_model, threshold, X, embeddings, data, scoop_labels):
    filepath = f'./src/{journal_type}/{journal_id}'
    
    if not os.path.exists(filepath):
        os.mkdir(filepath)
                          
    joblib.dump(kmeans_model, f"{filepath}/{journal_id}_kmeans.pkl")
    print("Model Kmeans berhasil disimpan")

    # Simpan threshold
    np.save(f"{filepath}/{journal_id}_threshold.npy", threshold)
    print("Threshold telah disimpan.")

    # Simpan data sebaran PCA
    np.save(f"{filepath}/{journal_id}_pca_data.npy", X)
    print("Data sebaran PCA telah disimpan.")

    # Simpan data sebaran multibert
    np.save(f"{filepath}/{journal_id}_bert_data.npy", embeddings.reshape(embeddings.shape[0], -1))
    print("Data sebaran PCA telah disimpan.")

    df_res = pd.DataFrame({'Data': data,
                   'Label': scoop_labels})

    # Memisahkan data dalam scoop dan outscoop
    inScoop_df = df_res[df_res['Label'] == 1]
    outScoop_df = df_res[df_res['Label'] == -1]

    df_res.to_csv(f'{filepath}/{journal_id}_data_jurnal.csv')
    inScoop_df.to_csv(f'{filepath}/{journal_id}_inscoop_data_jurnal.csv')
    outScoop_df.to_csv(f'{filepath}/{journal_id}_outscoop_data_jurnal.csv')
        

def load_data(journal_id, journal_type):
    file_path = f"./src/{journal_type}/{journal_id}"
    filename_kmeans = f"{file_path}/{journal_id}_kmeans.pkl"
    kmeans_model = joblib.load(filename_kmeans)
    pca_data = np.load(f"{file_path}/{journal_id}_pca_data.npy")
    threshold = np.load(f"{file_path}/{journal_id}_threshold.npy")
    X_bert = np.load(f"{file_path}/{journal_id}_bert_data.npy")
    df_res = pd.read_csv(f"{file_path}/{journal_id}_data_jurnal.csv")
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

def centroid_dist(kmeans_model, X):
    centroids = kmeans_model.cluster_centers_
    cluster_labels = kmeans_model.labels_
    return np.array([np.sqrt(np.sum(x - centroids[cluster_labels[i]])**2) for i, x in enumerate(X)])
    
def outscoop_threshold(centroid_dist):
    return np.mean(centroid_dist) + 2 * np.std(centroid_dist)