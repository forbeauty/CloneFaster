import torch
import pytorch_lightning as pl
import transformers
from torch import nn
from torch.nn import functional
from torch.nn import Module
from transformers import AutoModel, AutoModelForMaskedLM


class BertMaskedLM(Module):

    def __init__(self, config):
        super().__init__()
        self.bert = AutoModelForMaskedLM.from_pretrained(config['model']['name'])

    def forward(self, batch):

        logits = self.bert(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            output_hidden_states=True,
            return_dict=True
        )
        return logits


class BertSemanticS(Module):

    def __init__(self, config):

        super().__init__()

        self.bert = AutoModel.from_pretrained(config['model']['address'])
        self.dropout = nn.Dropout(0.02)
        self.classifier  = nn.Linear(config['model']['hidden_size'], 2)

    def forward(self, batch):

        outputs = self.bert(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids'],
            return_dict=True)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        return logits


class PretrainInnoModel(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config['model']['name'] == 'bert-base-multilingual-uncased':
            self.model = BertMaskedLM(config)
        else:
            raise("haven't support")
        self.optimizer_class = getattr(torch.optim, config['solver']['optimizer'])
        self.scheduler_fn = getattr(transformers, config['solver']['lr_schedule'])
        self.criterion_fn = getattr(torch.nn.functional, config['solver']['criterion'])
    def forward(self, batch):

        return self.model(batch)

    def training_step(self, batch, batch_idx):

        logits = self(batch)
        loss = self.criterion_fn(logits, batch['labels'])
        return loss

    def configure_optimizers(self):

        optimizer = self.optimizer_class(self.parameters(), lr=self.config['solver']['pretrain_initial_lr'])
        scheduler = self.scheduler_fn(optimizer=optimizer, self.config['solver'][], total_steps)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}




class TrainInnoModel(pl.LightningModule):

    def __init__(self, config, train_dataloader=None, val_dataloader=None):
        super().__init__()
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        if config['model']['name'] == 'bert-base-multilingual-uncased':
            self.model = BertSemanticS(config)
        else:
            raise ("haven't support")
        self.optimizer_class = getattr(torch.optim, config['solver']['optimizer'])
        self.scheduler_fn = getattr(transformers, config['solver']['lr_schedule'])
        self.criterion_fn = getattr(torch.nn.functional, config['solver']['criterion'])
        self.metric = pl.metrics.AUC()

    def forward(self, batch):

        return self.model(batch)

    def training_step(self, batch, batch_idx):

        logits = self.model(batch)
        loss = self.criterion_fn(logits, batch['labels'])
        auc = self.metric(logits, batch['labels'])
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_auc', auc, prog_bar=True, logger=True)
        return loss, auc

    def validation_step(self, batch, batch_idx):

        logits = self.model(batch)
        loss = self.criterion_fn(logits, batch['labels'])
        auc = self.metric(logits, batch['labels'])
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_auc', auc, prog_bar=True, logger=True)
        return loss, auc

    def test_step(self, batch, batch_idx):

        logits = self.model(batch)
        loss = self.criterion_fn(logits, batch['labels'])
        auc = self.metric(logits, batch['labels'])
        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_auc', auc, prog_bar=True, logger=True)
        return loss, auc

    # def setup(self, stage):
    #     if stage == 'fit':
    #         # Get dataloader by calling it - train_dataloader() is called after setup() by default
    #         train_loader = self.train_dataloader()
    #
    #         # Calculate total steps
    #         self.total_steps = (
    #             (len(train_loader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.gpus)))
    #             // self.hparams.accumulate_grad_batches
    #             * float(self.hparams.max_epochs)
    #         )

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        grouped_parameters = [
            # need decay
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay': self.config['solver']['weight_decay'],
            },
            # not decay
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0,
            }
        ]

        optimizer = self.optimizer_class(grouped_parameters,
                                         lr=self.config['solver']['pretrain_initial_lr'],
                                         weight_decay=self.config['solver']['weight_decay'])
        total_steps = len(self.train_dataloader)
        warmup_steps = self.config['solver']['warmup_fraction'] * total_steps
        scheduler = self.scheduler_fn(optimizer=optimizer,
                                      num_warmup_steps=warmup_steps,
                                      num_tranining_steps=total_steps)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}