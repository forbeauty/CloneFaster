import pandas as pd
import pytorch_lightning as pl
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold


class BaseDataset(Dataset):
    def __init__(self,
                 config,
                 df,
                 tokenizer,
                 use:str,
                 overfit=False
                 ):
        super.__init__()


        self.tokenizer = tokenizer
        self.df = df

        if use == 'pretrain':

        elif use == 'train':

        elif use == 'val':

        elif use == 'test':

        else:
            raise ValueError()





        if overfit:
            self.df = self.df.iloc[:4, :]


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):



class TrainDataModule(pl.LightningDataModule):

    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.args = args
        self.tokenizer = 'd'
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
                self.config,
                self.df_train,
                self.tokenizer,
                use='train',
                overfit=self.args.overfit
            )
            self.val_dataset = BaseDataset(
                self.config,
                self.df_val,
                self.tokenizer,
                use='val',
                overfit=self.args.overfit
            )

        if stage == 'test' or stage is None:

            self.test_dataset = BaseDataset(
                self.config,
                self.df_val,
                self.tokenizer,
                use='test',
                overfit=self.args.overfit
            )

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.config['solver']['batch_size'],
                          shuffle=True,
                          num_workers=self.args.cpu_workers,
                          collate_fn=collate_fn_with_padding(self.config['solver']['batch_size']))

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.config['solver']['batch_size'],
                          shuffle=True,
                          num_workers=self.args.cpu_workers,
                          collate_fn=collate_fn_with_padding(self.config['solver']['batch_size']))

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=1,
                          shuffle=True,
                          num_workers=self.args.cpu_workers,
                          collate_fn=collate_fn_with_padding(self.config['solver']['batch_size']))

    def convert_to_features(self):

        # # Either encode single sentence or sentence pairs
        # if len(self.text_fields) > 1:
        #     texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        # else:
        #     texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs=,
            max_length=,
            pad_to_max_length=True,
            truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features['labels'] = example_batch['label']

        return features


def pad_seq(encoded_inputs, max_length,
            pad_token_id=0,
            return_attention_mask=True):
    r"""
    Padding input sequence to max sequence length with `pad_token_id` and
    update other infomation if needed. Support both `List` and `np.ndarray`
    input sequence.

    Args:
        encoded_inputs (Dict): Dictionary of tokenized inputs (`List[int]` or
            `np.ndarray`) with 'input_ids' as key and additional information.
        max_length: maximum length of the returned list.
        pad_token_id (int): The id of pad token in the vocabulary. May
            specified by model. (Default 0)
        return_attention_mask (bool, optional): Set to False to avoid
            returning attention mask. (default: True)

    Returns:
        (Dict): Updated `encoded_inputs` with padded input_ids and attention
            mask if `return_attentioin_mask` if True.
    """
    origin_length = len(encoded_inputs["input_ids"])
    difference = max_length - origin_length
    if isinstance(encoded_inputs['input_ids'], list):
        if return_attention_mask:
            encoded_inputs["attention_mask"] = (
                [1] * origin_length + [0] * difference
            )
        if "token_type_ids" in encoded_inputs:
            encoded_inputs["token_type_ids"] = (
                encoded_inputs["token_type_ids"] + [0] * difference
            )
        encoded_inputs["input_ids"] = (
            encoded_inputs["input_ids"] + [pad_token_id] * difference
        )
    elif isinstance(encoded_inputs['input_ids'], np.ndarray):
        if return_attention_mask:
            attention_mask = np.zeros(max_length)
            attention_mask[:origin_length] = 1
            encoded_inputs["attention_mask"] = attention_mask
        if "token_type_ids" in encoded_inputs:
            token_type_ids = np.zeros(max_length).astype(np.int64)
            token_type_ids[:origin_length] = encoded_inputs['token_type_ids']
            encoded_inputs["token_type_ids"] = token_type_ids
        input_ids = np.full(max_length, pad_token_id).astype(np.int64)
        input_ids[:origin_length] = encoded_inputs['input_ids']
        encoded_inputs["input_ids"] = input_ids
    return encoded_inputs



def collate_fn_with_padding(batch):
    r"""
    Padding every sample in a list of samples to max length of these samples
    and merge them to form a mini-batch of Tensor(s). Each sample is an
    encoded inputs. This function padding each sample's input_ids to max
    length and update other information if needed. Then, for each item in a
    sample, puts each data field into a tensor with outer dimension batch size.

    Args:
        batch (List[Dict]): List of samples.

    Returns:
        (Dict): Padded and merged batch.
    """
    max_len = max([sample['length'] for sample in batch])
    padded_batch = [pad_seq(sample, max_len) for sample in batch]
    return default_collate(padded_batch)