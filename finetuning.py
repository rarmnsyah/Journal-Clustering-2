import pandas as pd

from sklearn.preprocessing import LabelEncoder

from check_lang import lang_checker_langdetect
from functions import *
from Preprocessing import *

# pd.options.mode.chained_assignment = None  # default='warn'

le = LabelEncoder()

model_num = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

finetuned_data = pd.read_csv('data/data_sinta_cleaned_s1_translated.csv')

# drop data
finetuned_data.dropna(inplace=True)
finetuned_data.drop_duplicates(inplace=True)

print(len(finetuned_data), device)

# split data by lang and create label
finetuned_data_en = finetuned_data[finetuned_data.lang == 'en']
# finetuned_data_id = finetuned_data[finetuned_data.lang == 'id']

finetuned_data_en['label'] = le.fit_transform(finetuned_data_en.journal)
# finetuned_data_id['label'] = le.fit_transform(finetuned_data_id.journal)

model_checkpoint = 'bert-base-cased'
model_checkpoint2 = 'indobenchmark/indobert-base-p1'

finetuning = BertFinetuning(finetuned_data_en, model_checkpoint, device, 32, f'model/test_bert_pipeline3_{model_num}_{finetuned_data_en.label.nunique()}.pt', finetuned_data_en.label.nunique())
# finetuning = BertFinetuning(finetuned_data_id, model_checkpoint2, device, 32, f'model/test_indobert_pipeline3_{model_num}_{finetuned_data_id.label.nunique()}.pt', finetuned_data_id.label.nunique())

# finetuning = BertFinetuningFromCheckpoint(finetuned_data_id, model_checkpoint2, device, 32, f'model/test_indobert_pipeline3_{model_num}_{finetuned_data_id.label.nunique()}.pt', finetuned_data_id.label.nunique())

print(finetuning.model, device)

finetuning.train(3)

finetuning.save(f'model/bert_pipeline3_{model_num}_{finetuned_data_en.label.nunique()}.pt')