import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch        
import joblib
import os

# from check_lang import id_to_en
from sklearn.cluster import KMeans
from transformers import BertTokenizer, AutoModel

from Preprocessing import preprocess_text
from finetuned_pipeline import *

# check_outscoop : default (False), warna plot berdasarkan label clustering, (True) warna untuk outscoop ditambah, ('focus') warna fokus untuk outscoop sama inscoop
def plot_vector_distribution(pca_result, kmeans_model, scoop_labels, new_data_pca = None, check_outscoop = False, lang = None, journal = None, label=None, scoop_labels_inject = None):
    if label is not None:
        kmeans_labels = label
    else:
        kmeans_labels = kmeans_model.labels_
        
    centroids = kmeans_model.cluster_centers_
    color_palette = 'hls'

    df_pca = pd.DataFrame(pca_result, columns=['Dimension 1', 'Dimension 2'])
    df_pca['kmeans_labels'] = kmeans_labels
    df_pca['scoop_labels'] = scoop_labels
    df_pca['plot_color'] = kmeans_labels

    if check_outscoop == 'focus':
        if df_pca['scoop_labels'].nunique() < 2:
            df_pca['plot_color'] = df_pca['scoop_labels'] + 1
            new_dummy_pca = df_pca.sample()
            new_dummy_pca.plot_color = 0
            new_dummy_pca['Dimension 1'] = centroids[0][0]
            new_dummy_pca['Dimension 2'] = centroids[0][1]
            df_pca = pd.concat([df_pca, new_dummy_pca], ignore_index=True, axis=0)
        else:
            df_pca['plot_color'] = df_pca['scoop_labels']
    elif check_outscoop == 'lang' :
        if lang == None:
            raise Exception('Jika memilih mode cek lang, maka diharuskan menyertakan language tiap data')
        bahasa_decoder = {
            'en' : 0,
            'id' : 1
        }
        lang = [bahasa_decoder[x] for x in lang]
        df_pca['plot_color'] = lang
        color_palette = 'Set2'
    elif check_outscoop == 'journal' :
        if journal == None:
            raise Exception('Jika memilih mode cek journal, maka diharuskan menyertakan journal dari tiap data')
        df_pca['plot_color'] = journal
    elif check_outscoop:
        df_pca.loc[df_pca.scoop_labels == -1, 'plot_color'] = -1
    else:
        df_pca['plot_color'] = df_pca['plot_color'] + 1
        new_dummy_pca = df_pca.sample()
        new_dummy_pca.plot_color = 0
        new_dummy_pca['Dimension 1'] = centroids[0][0]
        new_dummy_pca['Dimension 2'] = centroids[0][1]
        df_pca = pd.concat([df_pca, new_dummy_pca], ignore_index=True, axis=0)

    cluster_palette = sns.color_palette(color_palette, n_colors=df_pca['plot_color'].nunique() + 1)
    plt.figure(figsize=(8, 6))
    plot = sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='plot_color', data=df_pca, palette=cluster_palette)

    # Add centroids to the plot
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='^', c='red', s=50, label='Centroids')
    
    if new_data_pca is not None:
        sns.scatterplot(x = new_data_pca[:, 0], y = new_data_pca[:, 1], markers='x', hue = scoop_labels_inject, palette='Set2')
    # Show position of new PCA data
    plot.set(xlabel = None)
    plot.set(ylabel = None)
    plot.legend([],[], frameon=False)
    
    plt.title('PCA Latent Representation with Centroids')
    plt.show()


def save_data(journal_id, journal_type, kmeans_model, threshold, X, data, scoop_labels, pca):
    filepath = f'./src/{journal_type}/{journal_id}'
    
    if not os.path.exists(filepath):
        os.mkdir(filepath)
                          
    joblib.dump(kmeans_model, f"{filepath}/{journal_id}_kmeans.pkl")

    joblib.dump(pca, f"{filepath}/{journal_id}_pca.pkl")

    # Simpan threshold
    np.save(f"{filepath}/{journal_id}_threshold.npy", threshold)

    # Simpan data sebaran PCA
    np.save(f"{filepath}/{journal_id}_pca_data.npy", X)

    data.loc[:, 'scoop_labels'] = scoop_labels

    data.to_csv(f'{filepath}/{journal_id}_data_jurnal.csv')
        

def load_data(journal_id, journal_type):
    file_path = f"./src/{journal_type}/{journal_id}"
    kmeans_model = joblib.load(f"{file_path}/{journal_id}_kmeans.pkl")
    pca_model = joblib.load(f"{file_path}/{journal_id}_kmeans.pkl")
    pca_data = np.load(f"{file_path}/{journal_id}_pca_data.npy")
    threshold = np.load(f"{file_path}/{journal_id}_threshold.npy")
    df_res = pd.read_csv(f"{file_path}/{journal_id}_data_jurnal.csv")
    df_res_inject = pd.read_csv(f"{file_path}/{journal_id}_data_jurnal_inject.csv")
    pca_data_inject = np.load(f"{file_path}/{journal_id}_pca_data_inject.npy")
    return kmeans_model, threshold, pca_data, df_res, pca_model, df_res_inject, pca_data_inject

def embed(dataset, model_checkpoint, model=None, device='cpu'):
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
            embeddings.append(last_hidden_states.cpu().numpy().astype('float16'))

    embeddings = np.concatenate(embeddings, axis=0)

    return embeddings

def centroid_dist(kmeans_model, X, label=None):
    centroids = kmeans_model.cluster_centers_
    if label is not None:
        cluster_labels = label
    else:
        cluster_labels = kmeans_model.labels_
    return np.array([np.sqrt(np.sum(x - centroids[cluster_labels[i]])**2) for i, x in enumerate(X)])
    
    
def outscoop_threshold(centroid_dist):
    return np.mean(centroid_dist) + 2 * np.std(centroid_dist)

def determine_best_k(X, max_k=10):
    """
    Determine the best K for K-means clustering using the elbow method.

    Parameters:
    - X: The input data (features).
    - max_k: The maximum number of clusters to try.

    Returns:
    - The optimal number of clusters K.
    """
    inertia = []

    # Calculate inertia for each K
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    # Plot the elbow curve
    # plt.figure(figsize=(8, 5))
    # plt.plot(range(1, max_k + 1), inertia, marker='o')
    # plt.title('Elbow Method For Optimal K')
    # plt.xlabel('Number of clusters (K)')
    # plt.ylabel('Inertia')
    # plt.grid(True)
    # plt.show()

    # Identify the elbow point
    # Note: You can use more sophisticated methods to find the elbow point.
    # For simplicity, we'll use a basic approach here.
    # Find the "elbow" by checking the difference in inertia.
    diffs = np.diff(inertia)
    second_diffs = np.diff(diffs)
    optimal_k = np.argmin(second_diffs) + 2  # +2 because second_diffs is len(inertia)-2

    return optimal_k
