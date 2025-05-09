import pandas as pd

from sklearn.preprocessing import LabelEncoder

from functions import *
from Preprocessing import *

# pd.options.mode.chained_assignment = None  # default='warn'

le = LabelEncoder()

model_num = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

finetuned_data = pd.read_csv('data/data_sinta_cleaned_s1.csv')

# drop data
finetuned_data.dropna(inplace=True)
finetuned_data.drop_duplicates(inplace=True)

print(len(finetuned_data), device)

# split data by lang and create label
finetuned_data['label'] = le.fit_transform(finetuned_data.eissn)

model_checkpoint = 'google/multiberts-seed_3'

finetuning = BertFinetuning(finetuned_data, model_checkpoint, device, 32, f'model/skripsi_multiberts_rev2_{model_num}_{finetuned_data.label.nunique()}.pt', finetuned_data.label.nunique())
# finetuning = BertFinetuning(finetuned_data_id, model_checkpoint2, device, 32, f'model/test_indobert_pipeline3_{model_num}_{finetuned_data_id.label.nunique()}.pt', finetuned_data_id.label.nunique())

# finetuning = BertFinetuningFromCheckpoint(finetuned_data_id, model_checkpoint2, device, 32, f'model/test_indobert_pipeline3_{model_num}_{finetuned_data_id.label.nunique()}.pt', finetuned_data_id.label.nunique())

print(finetuning.model, device)

finetuning.train(3)

finetuning.save(f'model/skripsi_multiberts_rev2_{model_num}_{finetuned_data.label.nunique()}.pt')