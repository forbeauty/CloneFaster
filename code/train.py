import os
import yaml
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from .dataset import PretrainDataModule, TrainDataModule
from .model import PretrainInnoModel, TrainInnoModel
from .utils.args import get_args


def pretrain(config):

    datamodule = PretrainDataModule(config, args)
    datamodule.setup()
    model = PretrainInnoModel(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dirpath,
        filename='{epoch}_{val_loss:.4f}',
        save_top_k=3,
        verbose=False,
        monitor='val_loss',
        mode='min',
        prefix='train'
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        verbose=False,
        min_delta=0.,
        patience=3
    )

    trainer = pl.Trainer(accumulate_grad_batches=config['solver']['accumulation_steps'],
                         gpus=args.device[1],
                         tpu_cores=args.device[1],
                         max_epochs=config['solver']['pretrain_num_epochs'],
                         callbacks=[checkpoint_callback, early_stopping_callback],
                         gradient_clip_val=1,
                         deterministic=True
                         )
    trainer.fit(model=model, datamodule=datamodule)


def train(config):

    datamodule = TrainDataModule(config, args)
    datamodule.setup()
    model = TrainInnoModel(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dirpath,
        filename='{epoch}_{val_loss:.4f}',
        save_top_k=3,
        verbose=False,
        monitor='val_loss',
        mode='min',
        prefix='train'
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        verbose=False,
        min_delta=0.,
        patience=3
    )

    trainer = pl.Trainer(accumulate_grad_batches=config['solver']['accumulation_steps'],
                         gpus=args.device[1],
                         tpu_cores=args.device[1],
                         max_epochs=config['solver']['train_num_epochs'],
                         callbacks=[checkpoint_callback, early_stopping_callback],
                         gradient_clip_val=1,
                         deterministic=True
                         )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(ckpt_path=config['solver']['ckpt_path'], datamodule=datamodule)


    # model.freeze()  # eval
    # pred_lists = []

    # for inputs, row_id in prod_dataloader:
    #     if DEVICE == "tpu":
    #         device = xm.xla_device()
    #     elif DEVICE == "gpu":
    #         device = torch.device("cuda")
    #     else:
    #         device = torch.device("cpu")
    #
    #     inputs["input_ids"] = inputs["input_ids"].to(device)
    #     inputs["token_type_ids"] = inputs["token_type_ids"].to(device)
    #     inputs["attention_mask"] = inputs["attention_mask"].to(device)
    #
    #     predictions = model.forward(inputs)
    #     pred_label = torch.argmax(predictions[0], dim=1).cpu().numpy()
    #     pred_lists.append([row_id[0], pred_label[0]])
    #
    # pred_pd = pd.DataFrame(pred_lists, columns=["id", "prediction"])
    # pred_pd.to_csv('submission.csv', index=False)

    # 5-fold
    # Submission0 = pd.read_csv('./submission0.csv')
    # Submission1 = pd.read_csv('./submission1.csv')
    # Submission2 = pd.read_csv('./submission2.csv')
    # Submission3 = pd.read_csv('./submission3.csv')
    # Submission4 = pd.read_csv('./submission4.csv')
    #
    # Submission = pd.concat([Submission0, Submission1, Submission2, Submission3, Submission4]).groupby('image_name').mean().reset_index()
    # header = ["image_name","target"]
    # Submission.to_csv(f'submission.csv', columns = header, index=False)

if __name__ == '__main__':

    args = get_args()
    os.makedirs(args.save_dirpath, exist_ok=True)
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    pl.seed_everything(config['solver']['seed'])
    if config['solver']['mode'] == 'pretrain':

        pretrain(config)
    elif config['solver']['mode'] == 'train':

        train(config)
    else:

        raise ValueError