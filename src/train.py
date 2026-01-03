
import torch
import sys

# --- PATCH FOR PYTORCH 2.6+ (Security Fix) ---
_original_load = torch.load
def strict_load_patch(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = strict_load_patch
# ---------------------------------------------

import wandb
import argparse
import pytorch_lightning as pl
import os
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from os.path import join
import warnings

warnings.filterwarnings("ignore")

from sgmse.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()
torch.set_float32_matmul_precision('high')

from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module import SpecsDataModule
from sgmse.sdes import SDERegistry
from sgmse.model import ScoreModel

def get_argparse_groups(parser):
    groups = {}
    for group in parser._action_groups:
        group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
        groups[group.title] = argparse.Namespace(**group_dict)
    return groups

if __name__ == '__main__':
    base_parser = ArgumentParser(add_help=False)
    parser = ArgumentParser()
    for parser_ in (base_parser, parser):
        parser_.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")
        parser_.add_argument("--sde", type=str, choices=SDERegistry.get_all_names(), default="ouve")
        parser_.add_argument("--nolog", action='store_true', help="Turn off logging.")
        parser_.add_argument("--wandb_name", type=str, default=None, help="Name for wandb logger.")
        parser_.add_argument("--ckpt", type=str, default=None, help="Resume from checkpoint.")
        parser_.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs.")
        parser_.add_argument("--save_ckpt_interval", type=int, default=1, help="Save checkpoint interval.")
        
    temp_args, _ = base_parser.parse_known_args()

    backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
    sde_class = SDERegistry.get_by_name(temp_args.sde)
    trainer_parser = parser.add_argument_group("Trainer", description="Lightning Trainer")
    trainer_parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator type.")
    trainer_parser.add_argument("--devices", default="auto", help="How many gpus to use.")
    trainer_parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients.")
    trainer_parser.add_argument("--max_epochs", type=int, default=-1, help="Number of epochs.")
    
    ScoreModel.add_argparse_args(parser.add_argument_group("ScoreModel", description=ScoreModel.__name__))
    sde_class.add_argparse_args(parser.add_argument_group("SDE", description=sde_class.__name__))
    backbone_cls.add_argparse_args(parser.add_argument_group("Backbone", description=backbone_cls.__name__))
    data_module_cls = SpecsDataModule
    data_module_cls.add_argparse_args(parser.add_argument_group("DataModule", description=data_module_cls.__name__))

    args = parser.parse_args()
    arg_groups = get_argparse_groups(parser)

    model = ScoreModel(
        backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
        **{**vars(arg_groups['ScoreModel']), **vars(arg_groups['SDE']), **vars(arg_groups['Backbone']), **vars(arg_groups['DataModule'])}
    )

    if args.nolog:
        logger = None
        save_version = "local_run"
    else:
        logger = WandbLogger(project="sgmse", log_model=False, save_dir="logs", name=args.wandb_name)
        save_version = f"{logger.version}-{args.wandb_name}" if args.wandb_name else str(logger.version)

    ckpt_dir = join(args.log_dir, save_version)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # --- CALLBACKS ---
    callbacks = [ModelCheckpoint(dirpath=ckpt_dir, save_last=True)]
    callbacks += [ModelCheckpoint(dirpath=ckpt_dir, filename='ckpt-epoch-{epoch}', monitor='epoch', mode='max', save_top_k=5)]

    if hasattr(args, 'num_eval_files') and args.num_eval_files:
        callbacks += [ModelCheckpoint(dirpath=ckpt_dir, save_top_k=1, monitor="pesq", mode="max", filename='best-pesq-{epoch}')]

    # --- TRAINER ---
    # REVERTED TO 32-BIT (Standard Precision) for Stability
    trainer = pl.Trainer(
        **vars(arg_groups['Trainer']),
        logger=logger,
        log_every_n_steps=10, 
        num_sanity_val_steps=0,
        callbacks=callbacks,
        precision=32  
    )

    print(f"Starting training... Resume path: {args.ckpt}")
    trainer.fit(model, ckpt_path=args.ckpt)