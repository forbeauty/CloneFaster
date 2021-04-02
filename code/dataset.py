import pandas as pd
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold


class BaseDataset(Dataset):
    def __init__(self, df, tokenizer, split, overfit=False):

        super().__init__()
        self.tokenizer = tokenizer
        self.split = split
        self.df = df
        if overfit:
            self.df = self.df.iloc[:4, :]

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        text = [int(x) for x in self.df['seq1'][index].split()]
        text_pair = [int(x) for x in self.df['seq2'][index].split()]
        if self.split == 'pretrain':
            text, t1_label = self._random_word(text)
            text_pair, t2_label = self._random_word(text_pair)
            input_ids = [101] + text + [102] + text_pair + [102]
            lm_label_ids = [-1] + t1_label + [-1] + t2_label + [-1]
            # tokenize_results = self.tokenizer(text,
            #                                   text_pair,
            #                                   padding='longest',
            #                                   truncation=True,
            #                                   max_length=32,
            #                                   is_split_into_words=True,
            #                                   return_tensors="pt")

            ret = {
                'input_ids': input_ids,
                'lm_label_ids': lm_label_ids
            }
        elif self.split in ['train', 'val']:
            input_ids = [101] + text + [102] + text_pair + [102]
            label = int(self.df['label'][index])

            ret = {
                'input_ids': input_ids,
                'label': label
            }
        elif self.split == 'test':
            input_ids = [101] + text + [102] + text_pair + [102]
            ret = {
                'input_ids': input_ids
            }
        else:
            raise ValueError()

        return ret

    def _random_word(self, tokens):

        output_label = []

        for i, token in enumerate(tokens):
            prob = np.random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = "[MASK]"

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = np.random.choice(list(self.tokenizer.vocab.items()))[0]

                # -> rest 10% randomly keep current token

                # append current token to output (we will predict these later)
                try:
                    output_label.append(self.tokenizer.vocab[token])
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    output_label.append(self.tokenizer.vocab["[UNK]"])

            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return tokens, output_label


class PretrainDataModule(pl.LightningDataModule):

    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
        df1 = pd.read_csv(
            self.config['dataset']['train_path'],
            names=['seq1', 'seq2', 'labels'],
            sep='\t',
            encoding='utf-8',
            engine='python')
        df2 = pd.read_csv(
            self.config['dataset']['test_path'],
            names=['seq1', 'seq2'],
            sep='\t',
            encoding='utf-8',
            engine='python')
        self.df = df1['seq1', 'seq2'] + df2

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            self.train_dataset = BaseDataset(
                self.df,
                self.tokenizer,
                split='train',
                overfit=self.args.overfit
            )

    def prepare_data(self):
        pass

    def train_dataloader(self):

        return DataLoader(self.train_dataset,
                          batch_size=self.config['solver']['batch_size'],
                          shuffle=True,
                          num_workers=self.args.cpu_workers,
                          collate_fn=collate_fn_with_padding,
                          pin_memory=True
                          )


class TrainDataModule(pl.LightningDataModule):

    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['pretrained_model_path'])
        df = pd.read_csv(
            self.config['dataset']['train_path'],
            names=['seq1', 'seq2', 'labels'],
            sep='\t',
            encoding='utf-8',
            engine='python')
        self.df_test = pd.read_csv(
            self.config['dataset']['test_path'],
            names=['seq1', 'seq2'],
            sep='\t',
            encoding='utf-8',
            engine='python')

        if self.args.do_val:
            kf = StratifiedKFold(
                n_splits=config['solver']['k_fold'], shuffle=True, random_state=config['solver']['seed']
            )
            train_indexes, val_indexes = list(
                kf.split(df.index, y=df['labels'].astype('category'))
            )[args.fold]
            self.df_train = df.iloc[train_indexes].reset_index(drop=True)
            self.df_val = df.iloc[val_indexes].reset_index(drop=True)
        else:
            self.df_train = df


    def setup(self, stage=None):

        if stage == 'fit' or stage is None:

            self.train_dataset = BaseDataset(
                self.df_train,
                self.tokenizer,
                split='train',
                overfit=self.args.overfit
            )
            self.val_dataset = BaseDataset(
                self.df_val,
                self.tokenizer,
                split='val',
                overfit=self.args.overfit
            )

        if stage == 'test' or stage is None:

            self.test_dataset = BaseDataset(
                self.df_val,
                self.tokenizer,
                split='test',
                overfit=self.args.overfit
            )

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.config['solver']['batch_size'],
                          shuffle=True,
                          num_workers=self.args.cpu_workers,
                          collate_fn=collate_fn_with_padding,
                          pin_memory=True
                          )


    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.config['solver']['batch_size'],
                          shuffle=True,
                          num_workers=self.args.cpu_workers,
                          collate_fn=collate_fn_with_padding,
                          pin_memory=True
                          )

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=1,
                          shuffle=True,
                          num_workers=self.args.cpu_workers,
                          collate_fn=collate_fn_with_padding,
                          pin_memory=True
                          )


def pad_seq(encoded_inputs, max_length, pad_token_id=0):

    origin_length = len(encoded_inputs["input_ids"])
    difference = max_length - origin_length

    encoded_inputs["attention_mask"] = [1] * origin_length + [0] * difference
    encoded_inputs["token_type_ids"] = [1] * origin_length + [0] * difference
    encoded_inputs["input_ids"] = encoded_inputs["input_ids"] + [pad_token_id] * difference
    if 'lm_label_ids' in encoded_inputs:
        encoded_inputs['lm_label_ids'] = encoded_inputs['lm_label_ids'] + [-1] * difference
    return encoded_inputs


def collate_fn_with_padding(batch):

    max_len = max([len(sample['input_ids']) for sample in batch])
    padded_batch = [pad_seq(sample, max_len) for sample in batch]
    return default_collate(padded_batch)