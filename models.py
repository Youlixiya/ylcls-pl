import os
import time
import timm
import torch
import wandb
import platform
import numpy as np
import pandas as pd
from torch import nn
from torch.optim import *
from metrics import Metrics
from argparse import Namespace
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import transforms as T
from datasets import COVIDDataset, get_images_path, get_index2label



class COVIDModel(pl.LightningModule):
    def __init__(self, args : Namespace):
        super(COVIDModel, self).__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.model = timm.create_model(**args.model_args)
        self.metrics = Metrics(args.model_args['num_classes'])
        self.index2label = get_index2label(args.root_path)
        # self.train_loss = MeanMetric()
        # self.val_loss = MeanMetric()
        # self.train_acc = Accuracy(task='multiclass', num_classes=2)
        # self.val_acc = Accuracy(task='multiclass', num_classes=2)
        self.loss_fn = nn.CrossEntropyLoss()
    def setup(self, stage: str) -> None:
        self.train_dataset = COVIDDataset(self.args, 'train')
        self.val_dataset = COVIDDataset(self.args, 'valid')
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                                batch_size=self.args.batch_size,
                                shuffle=True,
                                num_workers=self.args.workers,
                                pin_memory=True,
                                drop_last=True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                                batch_size=self.args.batch_size,
                                shuffle=False,
                                num_workers=self.args.workers,
                                pin_memory=True,
                                drop_last=True)
    def configure_optimizers(self):
        self.optimizer = eval(self.args.optimizer)(self.model.parameters(), lr=self.args.lr)
        return self.optimizer
    def training_step(self, batch, batch_nb):
        train_images, train_targets = batch
        b = train_images.shape[0]
        train_preds = self.model(train_images)
        train_loss = self.loss_fn(train_preds, train_targets.reshape(b))
        self.metrics.callback_on_per_step(train_preds, train_targets.reshape(b), train_loss, 'train')
        self.log('train_loss', train_loss, prog_bar=True, on_step=True, on_epoch=False,
                 logger=True if self.args.use_wandb else False)
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        self.log("men", mem, prog_bar=True, on_step=True, on_epoch=False,
                 logger=True if self.args.use_wandb else False,
                 )
        return train_loss
    def validation_step(self, batch, batch_nb):
        valid_images, valid_targets = batch
        b = valid_images.shape[0]
        valid_preds = self.model(valid_images)
        valid_loss = self.loss_fn(valid_preds, valid_targets.reshape(b))
        self.metrics.callback_on_per_step(valid_preds, valid_targets.reshape(b), valid_loss, 'valid')
        self.log('val_loss',valid_loss, prog_bar=True, on_step=True, on_epoch=False,
                 logger=True if self.args.use_wandb else False)
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        self.log("men", mem, prog_bar=True, on_step=True, on_epoch=False,
                 logger=True if self.args.use_wandb else False,
                 )
        return valid_loss
    def on_validation_epoch_start(self) -> None:
        train_accuracy, train_precision, train_recall, train_f1, train_mean_loss = self.metrics.callback_on_per_epoch_end(
            'train')
        self.metrics.callback_on_per_epoch_start('valid')
        self.log('train_mean_loss', train_mean_loss, prog_bar=True, on_step=False, on_epoch=True,
                 logger=True if self.args.use_wandb else False)
        self.log('train_accuracy', train_accuracy, prog_bar=True, on_step=False, on_epoch=True,
                 logger=True if self.args.use_wandb else False)
        self.log('train_precision', train_precision, prog_bar=True, on_step=False, on_epoch=True,
                 logger=True if self.args.use_wandb else False)
        self.log('train_recall', train_recall, prog_bar=True, on_step=False, on_epoch=True,
                 logger=True if self.args.use_wandb else False)
        self.log('train_f1', train_f1, prog_bar=True, on_step=False, on_epoch=True,
                 logger=True if self.args.use_wandb else False)
    def on_validation_epoch_end(self) -> None:
        valid_accuracy, valid_precision, valid_recall, valid_f1, valid_mean_loss = self.metrics.callback_on_per_epoch_end('valid')
        self.metrics.callback_on_per_epoch_start('train')
        self.log('val_mean_loss', valid_mean_loss, prog_bar=True, on_step=False, on_epoch=True,
                 logger=True if self.args.use_wandb else False)
        self.log('val_accuracy', valid_accuracy, prog_bar=True, on_step=False, on_epoch=True,
                 logger=True if self.args.use_wandb else False)
        self.log('val_precision', valid_precision, prog_bar=True, on_step=False, on_epoch=True,
                 logger=True if self.args.use_wandb else False)
        self.log('val_recall', valid_recall, prog_bar=True, on_step=False, on_epoch=True,
                 logger=True if self.args.use_wandb else False)
        self.log('val_f1', valid_f1, prog_bar=True, on_step=False, on_epoch=True,
                 logger=True if self.args.use_wandb else False)
    def on_train_end(self) -> None:
        self.test(train = True)
        self.save_metrics_csv()
        self.save_metrics_curves()
    def test(self, save_img = True, train = False):
        save_path = f'{self.args.exp_name}/detection'
        os.makedirs(save_path, exist_ok=True)
        images_list, _ = get_images_path(self.args.root_path, 'test')
        transformers = T.Compose(
            [
                T.Resize((self.args.image_size, self.args.image_size)),
                T.ToTensor(),
                T.Normalize(0.5, 0.5)
            ]
        )
        cnt = 0
        if self.args.use_wandb and train:
            detect_table = wandb.Table(columns=['detect_image', 'pred_label', 'conf', 'detect_time(ms)'])
        else:
            detect_table = None
        for img in images_list:
            image_name = img.split('/')[-1]
            img = Image.open(img).convert('RGB')
            img_tensor = transformers(img).unsqueeze(0).to(self.device)
            t1 = time.time()
            out = self.model(img_tensor).softmax(-1).squeeze().cpu()
            t2 = time.time()
            t = int((t2 - t1) * 1000)
            arg = out.argmax().item()
            conf = round(out[arg].item() * 100, 2)
            draw = ImageDraw.Draw(img)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = int(25 * self.args.image_size / 224)
            plat = platform.platform()
            # if ('Windows' in plat):
            #     font_type = ImageFont.truetype("simhei.ttf", font_size)
            # else:
            font_type = ImageFont.truetype(fm.findfont(fm.FontProperties(family='simhei.ttf')), font_size)
            # img_detected = cv2.putText(imgcv2, f'label:{self.index2label[arg]}', (10, 30), font, 1, (255, 255, 255), 1)
            # img_detected = cv2.putText(img_detected, f'conf:{conf}', (10, 60), font, 1, (255, 255, 255), 1)
            box1 = (int(10 * self.args.image_size / 224), int(10 * self.args.image_size / 224))
            box2 = (int(10 * self.args.image_size / 224), int(40 * self.args.image_size / 224))
            draw.text(box1, f'label:{self.index2label[arg]}', font=font_type, fill=(144, 144, 144))
            draw.text(box2, f'conf:{conf}', font=font_type, fill=(144, 144, 144))
            print(f'image_index:{image_name} detect_time:{t}ms label:{self.index2label[arg]} conf:{conf}')
            if save_img:
                image_save_path = f'{save_path}/{image_name}'
                img.save(image_save_path)
                # cv2.imwrite(image_save_path, img_detected)
            if self.args.use_wandb and train:
                Img = wandb.Image(img, caption=f"detect_image_{cnt}")
                detect_table.add_data(Img, self.index2label[arg], conf, t)
    def save_metrics_csv(self):
        loss_df = pd.DataFrame({'train_loss': self.metrics.metrics_dict['train_loss_list'],
                                'valid_loss': self.metrics.metrics_dict['valid_loss_list'],
                                'train_accuracy': self.metrics.metrics_dict['train_acc_list'],
                                'valid_accuracy': self.metrics.metrics_dict['valid_acc_list'],
                                'train_precision': self.metrics.metrics_dict['train_total_precision_list'],
                                'valid_precision': self.metrics.metrics_dict['valid_total_precision_list'],
                                'train_recall': self.metrics.metrics_dict['train_total_recall_list'],
                                'valid_recall': self.metrics.metrics_dict['valid_total_recall_list'],
                                'train_f1': self.metrics.metrics_dict['train_total_f1_list'],
                                'valid_f1': self.metrics.metrics_dict['valid_total_f1_list'],
                                'train_mean_precision': self.metrics.metrics_dict['train_mean_precision_list'],
                                'valid_mean_precision': self.metrics.metrics_dict['valid_mean_precision_list'],
                                'train_mean_recall': self.metrics.metrics_dict['train_mean_recall_list'],
                                'valid_mean_recall': self.metrics.metrics_dict['valid_mean_recall_list'],
                                'train_mean_f1': self.metrics.metrics_dict['train_mean_f1_list'],
                                'valid_mean_f1': self.metrics.metrics_dict['valid_mean_f1_list'],
                                })
        if self.args.use_wandb:
            metrics_table = wandb.Table(
                columns=['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy', 'train_precision',
                         'valid_precision', 'train_recall', 'valid_recall', 'train_f1', 'valid_f1',
                         'train_mean_precision',
                         'valid_mean_precision', 'train_mean_recall', 'valid_mean_recall', 'train_mean_f1',
                         'valid_mean_f1'])
            for i in range(loss_df.shape[0]):
                metrics_table.add_data(*(loss_df.iloc[i, :].tolist()))
            wandb.log({
                "metrics_table": metrics_table
            })
        loss_df.to_csv(f'{self.args.exp_name}/metrics.csv', index=False)

    def save_metrics_curves(self):
        fig, ax = plt.subplots(3, 2, figsize=(24, 36))
        ax[0][0].set_title('Loss_curve', fontsize=20, fontweight='bold')
        ax[0][1].set_title('Accuracy_curve', fontsize=20, fontweight='bold')
        ax[1][0].set_title('Precision_curve', fontsize=20, fontweight='bold')
        ax[1][1].set_title('Recall_curve', fontsize=20, fontweight='bold')
        ax[2][0].set_title('F1_curve', fontsize=20, fontweight='bold')
        ax[2][1].set_title('Confusion_matrix', fontsize=20, fontweight='bold')
        for i in range(3):
            for j in range(2):
                # x轴标签
                if i == 2 and j == 1:
                    ax[i][j].set_xlabel('Predicted label', fontsize=20, fontweight='bold')
                else:
                    ax[i][j].set_xlabel('epochs', fontsize=20, fontweight='bold')
                    # 展示网格线
                    ax[i][j].grid()
        # y轴标签
        ax[0][0].set_ylabel('loss', fontsize=20, fontweight='bold')
        ax[0][1].set_ylabel('accuracy', fontsize=20, fontweight='bold')
        ax[1][0].set_ylabel('precision', fontsize=20, fontweight='bold')
        ax[1][1].set_ylabel('recall', fontsize=20, fontweight='bold')
        ax[2][0].set_ylabel('f1', fontsize=20, fontweight='bold')
        ax[2][1].set_ylabel('True label', fontsize=20, fontweight='bold')
        # 绘制
        x = np.arange(0, self.args.num_epochs)
        ax[0][0].plot(x, self.metrics.metrics_dict['train_loss_list'], color='blue', label='train_loss')
        ax[0][0].plot(x, self.metrics.metrics_dict['valid_loss_list'], color='green', label='valid_loss')
        ax[0][1].plot(x, self.metrics.metrics_dict['train_acc_list'], color='red', label='train_accuracy')
        ax[0][1].plot(x, self.metrics.metrics_dict['valid_acc_list'], color='yellow', label='valid_accuracy')
        ax[1][0].plot(x, self.metrics.metrics_dict['train_mean_precision_list'], color='blue',
                      label='train_mean_precision')
        ax[1][0].plot(x, self.metrics.metrics_dict['valid_mean_precision_list'], color='green',
                      label='valid_mean_precision')
        ax[1][1].plot(x, self.metrics.metrics_dict['train_mean_recall_list'], color='cyan',
                      label='train_mean_recall')
        ax[1][1].plot(x, self.metrics.metrics_dict['valid_mean_recall_list'], color='magenta',
                      label='valid_mean_recall')
        ax[2][0].plot(x, self.metrics.metrics_dict['train_mean_f1_list'], color='blue',
                      label='train_mean_f1')
        ax[2][0].plot(x, self.metrics.metrics_dict['valid_mean_f1_list'], color='green',
                      label='valid_mean_f1')
        ax21 = ax[2][1].matshow(self.metrics.metrics_dict['valid_confusion_matrix'], cmap=plt.cm.Blues)
        fig.colorbar(ax21, ax=ax[2][1])
        for i in range(self.args.model_args['num_classes']):
            for j in range(self.args.model_args['num_classes']):
                ax[2][1].annotate(self.metrics.metrics_dict['valid_confusion_matrix'][j, i], xy=(i, j),
                                  horizontalalignment='center', verticalalignment='center')
        for i in range(3):
            for j in range(2):
                if i != 2 or j != 1:
                    ax[i][j].legend(loc='upper right', fontsize=20)  # 设置图表图例在右上角
        plt.savefig(f'{self.args.exp_name}/metrics_curves.png', bbox_inches='tight', dpi=300)
        if self.args.use_wandb:
            metrics_curves = Image.open(f'{self.args.exp_name}/metrics_curves.png').convert('RGB')
            wandb.log({
                "metrics_curves": wandb.Image(metrics_curves)
            })

