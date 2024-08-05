import pandas as pd

from sklearn.preprocessing import LabelEncoder

from check_lang import lang_checker_langdetect
from functions import *
from Preprocessing import *

# pd.options.mode.chained_assignment = None  # default='warn'

le = LabelEncoder()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

finetuned_data = pd.read_csv('data/data_sinta_cleaned.csv')

finetuned_data.drop(columns=['jid_umum'], inplace=True)
finetuned_data.dropna(inplace=True)
finetuned_data.drop_duplicates(inplace=True)
finetuned_data['lang'] = finetuned_data.data_cleaned.apply(lang_checker_langdetect)

finetuned_data_en = finetuned_data[finetuned_data.lang == 'en']
finetuned_data_en['label'] = le.fit_transform(finetuned_data_en.jid)

finetuned_data_id = finetuned_data[finetuned_data.lang == 'id']
finetuned_data_id['label'] = le.fit_transform(finetuned_data_id.jid)

model_checkpoint = 'bert-base-cased'
model_checkpoint2 = 'indobenchmark/indobert-base-p1'

finetuning2 = BertFinetuning(finetuned_data_id, model_checkpoint2, device, 32, 'model/test_indobert_pipeline3_01.pt')
print(finetuning2.model, device)
finetuning2.train(3)