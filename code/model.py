import math
import torch
from torch import nn
from torch.nn import Module
from transformers import AutoModel


class BertModel(Module):

    def __init__(self, config):
        super().__init__()
        self.pretrained_model_name = config['model']['pretrained_model_name']
        self.model = AutoModel.from_pretrained(self.pretrained_model_name)

    def forward(self, batch):
        r"""
        Forward pass.

        Args:
            batch (Dict): The input batch. A Dictionary with at least keys
                `input_ids`, `attention_mask` for all kind of models and
                `token_type_ids` for some models.

        Returns:
            (:class:`~transformers.file_utils.ModelOutput`): The dictionary-
                like model outputs with keys `last_hidden_state`,
                `hidden_states`, `attentions`, `length` for all kind of
                models.
        """

        ret = self.model.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            output_hidden_states=True,
            return_dict=True
        )
        return ret

class InnoModel(Module):

    def __init__(self, config):
        super().__init__()

        self.model = BertModel(config)

    def forward(self, batch):
        return self.model(batch)