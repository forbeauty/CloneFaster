import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class PretrainDataset(Dataset):

    def __init__(self, config, sep='\t', overfit=False):
        super().__init__()
        df1 = pd.read_csv(config['dataset']['train_path'], names=['seq1', 'seq2', 'labels'], sep=sep, encoding='utf-8')
        df2 = pd.read_csv(config['dataset']['test_path'], names=['seq1', 'seq2'], sep=sep, encoding='utf-8')
        df2['labels'] = 2
        self.df = df1 + df2
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['model']['pretrained_model_name']
        )

        if overfit:
            self.df = self.df.iloc[:4, :]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        tokenized_inputs = self.tokenizer(
            text=[int(x) for x in self.df['seq1'].strip().split()],
            text_pair=[int(x) for x in self.df['seq2'].strip().split()],
            add_special_tokens=True,
            return_tensors=None,
            return_token_type_ids=True,
            max_length=32,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )

        ret = {
            'input_ids': tokenized_inputs['input_ids'],
            'token_type_ids': tokenized_inputs['token_type_ids'],
            'attention_mask': tokenized_inputs['attention_mask']
        }

        return ret

