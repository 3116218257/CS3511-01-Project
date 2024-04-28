import os
import numpy as np
import argparse

from pytorch_lightning import Trainer, loggers
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging

from src.data_modules.task1_dm import Task1DM
from src.lightning_modules.task1_lm import Task1LM
from src.utils import set_seed

def run(conf):
    dm = Task1DM('segment', conf.data_dir, conf.input_size, conf.batch_size, conf.num_workers, conf.balanced_sampling, conf.target)
    lm = Task1LM(conf.lr, conf.backbone, target=conf.target)
    tb_logger = loggers.TensorBoardLogger(save_dir=conf.save_dir)

    ckpt_callback = ModelCheckpoint(
        dirpath=conf.save_dir,
        monitor='val/Dice_avg',
        filename=f"{conf.seed}_class_avg_" + 'epoch={epoch:02d}-Dice={val/Dice_avg:.4f}',
        mode='max'
    )
    ckpt_callback1 = ModelCheckpoint(
        dirpath=conf.save_dir,
        monitor='val/Dice_1',
        filename=f"{conf.seed}_class_1_" + 'epoch={epoch:02d}-Dice={val/Dice_1:.4f}',
        mode='max'
    )
    ckpt_callback2 = ModelCheckpoint(
        dirpath=conf.save_dir,
        monitor='val/Dice_2',
        filename=f"{conf.seed}_class_2_" + 'epoch={epoch:02d}-Dice={val/Dice_2:.4f}',
        mode='max'
    )
    ckpt_callback3 = ModelCheckpoint(
        dirpath=conf.save_dir,
        monitor='val/Dice_3',
        filename=f"{conf.seed}_class_3_" + 'epoch={epoch:02d}-Dice={val/Dice_3:.4f}',
        mode='max'
    )
    lr_callback = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(logger=tb_logger, 
                      callbacks=[ckpt_callback, ckpt_callback1, ckpt_callback2, ckpt_callback3, lr_callback],
                      default_root_dir=conf.save_dir,
                      max_epochs=conf.max_epochs,
                      log_every_n_steps=1,
                      gpus = [0],
                      plugins=DDPPlugin(find_unused_parameters=True) if len(conf.gpu_id.split(',')) > 1 else None,
                      precision=16,
                      sync_batchnorm=True if len(conf.gpu_id.split(',')) > 1 else False,
                      num_sanity_val_steps=0,
                      )
    trainer.fit(lm, datamodule=dm)
    return ckpt_callback.best_model_score


def parse_argument():
    parser = argparse.ArgumentParser()
    boolean_flags = ['true', 'yes', '1', 't', 'y']
    is_boolean = lambda x: x.lower() in boolean_flags
    
    parser.add_argument('--pname', type=str, default='debug')
    parser.add_argument('--save_dir', type=str, default='./outputs_drac/task1')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--balanced_sampling', type=is_boolean, default=False)
    
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd', 'radam'])

    parser.add_argument('--gpu_id', type=str, default='1')
    parser.add_argument('--max_epochs', type=int, default=500)
    
    parser.add_argument('--target', default=3, type=int)
    
    return parser.parse_args()

if __name__ == '__main__':
    conf = parse_argument()
    #set_seed(conf.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpu_id
    conf.save_dir = os.path.join(conf.save_dir, conf.pname, f"class_{conf.target}")
    os.makedirs(conf.save_dir, exist_ok=True)

    if conf.target in [1]:
        conf.backbone = 'u2net_lite'
    elif conf.target in [2, 3]:
        conf.backbone = 'u2net_full'
    else:
        raise ValueError
    
    for seed in [42, 43, 44, 45, 46]:
        conf.seed = seed
        set_seed(conf.seed)
        score = run(conf)

        print(f'{conf.seed} Best Dice Similarity Coefficient: {score}')
        with open(os.path.join(conf.save_dir, f'{conf.seed}_performance.txt'), 'w') as f:
            f.write(f'{score}')