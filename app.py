import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch        
import joblib

# from transformers import BertTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from Preprocessing import *

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

    def predict_scoop(self, title, abstract):
        processed_text = preprocess_text(title + abstract)
        encoded_dict = self.tokenizer.encode_plus(
            processed_text,
            add_special_tokens=True,
            max_length=128,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoded_dict['input_ids'].to('cpu')
        attention_mask = encoded_dict['attention_mask'].to('cpu')

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            last_hidden_states = outputs.last_hidden_state
            new_embedding = last_hidden_states.cpu().numpy().reshape(1, -1)

        new_data_pca = self.fit_new_data_to_pca(new_embedding)
        distance_to_centroid = np.sqrt(np.sum((new_data_pca - self.kmeans_model.cluster_centers_) ** 2, axis=1))
        prediction = "in scoop" if distance_to_centroid <= self.threshold else "out scoop"
        return prediction, new_data_pca, distance_to_centroid


class StreamlitApp:
    def __init__(self):
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        # self.model = AutoModel.from_pretrained('bert-base-multilingual-cased')
        self.tokenizer = None
        self.model = None
        self.journal_types = os.listdir('src')
        self.jurnal_ids = None

    def load_data(self, jurnal_id, jurnal_type):
        file_path = f"./src/{jurnal_type}/{jurnal_id}"
        filename_kmeans = f"{file_path}/{jurnal_id}_kmeans.pkl"
        kmeans_model = joblib.load(filename_kmeans)
        threshold = np.load(f"{file_path}/{jurnal_id}_threshold.npy")
        pca_data = np.load(f"{file_path}/{jurnal_id}_pca_data.npy")
        X_bert = np.load(f"{file_path}/{jurnal_id}_bert_data.npy")
        df_res = pd.read_csv(f"{file_path}/{jurnal_id}_data_jurnal.csv")
        return kmeans_model, threshold, pca_data, X_bert, df_res

    def select_jurnal_types(self):
        return st.sidebar.selectbox('Select Journal Types', self.journal_types, key='select_journal_type')
    
    def select_jurnal_id(self):
        return st.sidebar.selectbox('Select Journal ID', self.jurnal_ids, key='select_journal_id')
    
    def plot_vector_distribution(self, vector_representation, kmeans_labels, scoop_labels, new_data_pca=None):
        pca = PCA(n_components=2, random_state=0)
        pca_result = pca.fit_transform(vector_representation)

        scoop_labels[scoop_labels == 1] = 2
        scoop_labels[scoop_labels == -1] = 3

        df_pca = pd.DataFrame(pca_result, columns=['Dimension 1', 'Dimension 2'])
        df_pca['Scoop Label'] = scoop_labels

        cluster_palette = sns.color_palette('tab10', n_colors=2)
        plt.figure(figsize=(8, 6))
        plot = sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='Scoop Label', data=df_pca, palette=cluster_palette)
        # handles, labels = plot.get_legend_handles_labels()
        # plot.legend(handles, label_)
        plot.set(xlabel = None)
        plot.set(ylabel = None)
        plot.legend([],[], frameon=False)
        
        # Add centroids to the plot
        centroids = []
        for label in np.unique(kmeans_labels):
            centroid = np.mean(pca_result[kmeans_labels == label], axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='^', c='red', s=50, label='Centroids')
        
        # Show position of new PCA data
        if new_data_pca is not None:
            plt.scatter(new_data_pca[:, 0], new_data_pca[:, 1], marker='o', c='black', s=50, label='New Data')
        
        # plt.title('PCA Latent Representation with Centroids')
        plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
        st.pyplot()

    def main(self):
        st.title('Jurnal Clustering with Mulltibert and K-Means')
        st.sidebar.title('Options')

        st.set_option('deprecation.showPyplotGlobalUse', False)

        jurnal_type = self.select_jurnal_types()
        self.jurnal_ids = os.listdir(os.path.join('src', jurnal_type))
        jurnal_id = self.select_jurnal_id()
        kmeans_model,  outscoop_threshold, pca_data, X_bert, df_res = self.load_data(jurnal_id, jurnal_type)
        predictor = ScoopPredictor(self.tokenizer, self.model, kmeans_model,  outscoop_threshold, pca_data, X_bert)

        # Memisahkan data dalam scoop dan outscoop
        inScoop_df = df_res[df_res['Label'] == 1]
        outScoop_df = df_res[df_res['Label'] == -1]

        # Mendapatkan koordinat pusat cluster
        centroid = kmeans_model.cluster_centers_

        # Menghitung jarak antara setiap titik data dengan centroid
        jarak_ke_centroid = np.sqrt(np.sum((pca_data - centroid)**2, axis=1))

        # Menentukan label untuk scoop dan outscoop
        scoop_labels = np.ones(len(pca_data))
        scoop_labels[jarak_ke_centroid > outscoop_threshold] = -1

        show_jurnal_cluster_info = st.sidebar.checkbox('Show Jurnal Cluster Information', value=False, key='select_clucter_info')
        
        if show_jurnal_cluster_info:
            st.subheader('Journal Cluster Information')
            st.subheader('Vector Distribution Plot')
            self.plot_vector_distribution(pca_data, kmeans_model.labels_, scoop_labels)

            st.subheader('Data Scoop Classification')
            col1, col2 = st.columns([2, 2])  

            with col1:
                st.write('**In Scoop Data**')
                st.dataframe(inScoop_df['Data'])

            with col2:
                st.write('**Out Scoop Data**')
                st.dataframe(outScoop_df['Data'])


        st.subheader('Scoop Prediction')
        judul_baru = st.text_input('Title', 'Sistem Pakar Dengan Metode Backward Chaining Untuk Pengujian Transistor Di Laboratorium Elektronika')
        abstrak_baru = st.text_area('Abstract', 'Sistem pakar juga merupakan kecerdasan buatan, sistem pakar adalah program untuk menyimpan dan proses pengetahuan untuk area ahusus, itu sebabnya mereka mampu untuk menjawab pertanyaan dan memecahkan masalah, sesuai dengan kesepakatan para ahli. Backward Chainning model secara terbalik dari hipotesa, sebuah potensi atau kesimpulan yang harus dibuktikan dengan fakta-fakta yang mendukung hipotesa. Backward Chainning juga juga dideskripsikan dalam bentuk penalaran mulai goal menuju subgoal dengan pemahaman mencapai goal berarti memenuhi subgoalnya. Pohon keputusan adalah hasil dari proses pelacakan yang dapat digunakan untuk menjelaskan jawaban dari pertanyaan pertanyan. Algoritma Backward Chainning menggunakan struktur data utama dalam pembentukan pohon keputusan. Pada knowledge base diperlukan aturan aturan yang akan di simpan dalam database. Untuk aturan–aturan pada pembentukan sistem pakar ini dapat dibuat berdasarkan fakta yang ada yang disebutkan di atas. Untuk rule tes pin aturan yang menghasilkan penggunaan multi meter untuk pengukuran Transistor yang benar dan salah. Rule Baca skala merupakan aturan hasil akhir dari pengujian Transistor apakah baik, bocor atau putus sehingga tidak dapat digunakan lagi.')

        if st.button('Predict'):
            scoop_prediction, new_data_pca, distance = predictor.predict_scoop(judul_baru, abstrak_baru)
            st.subheader('Prediction Result')
            if scoop_prediction == 'in scoop':
                st.success('Prediction: In Scoop')
            else:
                st.error('Prediction: Out Scoop')
            
            st.info('Distance to Centroid: {:.2f}'.format(distance[0]))
            # Plot vector distribution
            self.plot_vector_distribution(pca_data, kmeans_model.labels_, scoop_labels, new_data_pca)


if __name__ == "__main__":
    app = StreamlitApp()
    app.main()
