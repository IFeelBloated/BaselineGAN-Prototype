# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import click
import re
import json
import tempfile
import torch

import fsspec
import dnnlib
from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops


#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    # TODO Fix this for fsspec
    # dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    local_rank=rank
    num_nodes = int(os.environ.get('NUM_NODES', '1'))
    node_rank = int(os.environ.get('NODE_RANK', '1'))
    if num_nodes > 1 and  "MASTER_ADDR" not in os.environ:
        raise AssertionError("MASTER_ADDR must be specified")
    if num_nodes > 1 and  "MASTER_PORT" not in os.environ:
        raise AssertionError("MASTER_PORT must be specified")
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', '1'))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "8888")
    global_rank = rank + (node_rank * local_world_size)
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, 
                rank=global_rank, world_size=c.num_gpus)
        else:
            #init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', 
                #init_method=init_method, 
                rank=global_rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=torch.distributed.get_rank(), sync_device=sync_device)
    if torch.distributed.get_rank() != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **c)

#----------------------------------------------------------------------------

def load_latest_checkpoint(directory_path):
    """
    Load the latest checkpoint from a specified directory using a filename pattern.
    This function is designed to work with any fsspec-compatible file system.

    Args:
    directory_path (str): The URLpath to the directory containing the checkpoint files.
                          It should include the protocol prefix (e.g., 's3://', 'file://').

    Returns:
    The path of the latest checkpoint file.
    """
    # Infer the appropriate filesystem from the given directory_path
    fs, _, paths = fsspec.get_fs_token_paths(directory_path)
    
    directory = paths[0]
    
    # Construct a glob pattern for the checkpoint files
    pattern = os.path.join(directory, "network-snapshot-*.pkl")

    # Use glob to find all files matching the pattern
    checkpoints = fs.glob(pattern)

    # If no checkpoints are found, return None
    if not checkpoints:
        print("No checkpoint files found.")
        return None

    # Sort the checkpoints to find the latest one
    # This step assumes that the filenames contain sortable timestamps or indices
    latest_checkpoint = sorted(checkpoints)[-1]

    # Return the latest checkpoint path
    return f"{fs.protocol[0]}://{latest_checkpoint}"

def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)
    
    if not c.no_subdir:
        prev_run_dirs = []
        if os.path.isdir(outdir):
            prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    else:
        c.run_dir = outdir
        c.resume_pkl = load_latest_checkpoint(outdir)
        print(f'Autoresuming from : {c.resume_pkl}')

        #print(c.resume_pkl)
        #assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    #os.makedirs(c.run_dir)
    with fsspec.open(os.path.join(c.run_dir, 'training_options.json'), 'wt', auto_mkdir=True) as f:
        json.dump(c, f, indent=2)

    c.pop('no_subdir') # removes no_subdir

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    num_nodes = int(os.environ.get('NUM_NODES', 1))
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', '1'))
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=local_world_size)

#----------------------------------------------------------------------------

def init_dataset_kwargs(data):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)
@click.option('--preset',       help='Preset configs', metavar='STR',                           type=str, required=True)

# Optional features.
@click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--mirror',       help='Enable dataset x-flips', metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--aug',          help='Augmentation mode',                                       type=click.Choice(['noaug', 'ada', 'fixed']), default='ada', show_default=True)
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)

# Misc hyperparameters.
@click.option('--p',            help='Probability for --aug=fixed', metavar='FLOAT',            type=click.FloatRange(min=0, max=1), default=0.2, show_default=True)
@click.option('--g-batch-gpu',  help='Limit batch size per GPU for G', metavar='INT',           type=click.IntRange(min=1))
@click.option('--d-batch-gpu',  help='Limit batch size per GPU for D', metavar='INT',           type=click.IntRange(min=1))

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=10000000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=3, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)

@click.option('--no-subdir', is_flag=True, help='Do not create subdirectories')


def main(**kwargs):
    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    
    c.G_kwargs = dnnlib.EasyDict(class_name='training.networks_baseline.Generator')
    c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_baseline.Discriminator')
    
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0], eps=1e-8)
    
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.BaselineGANLoss')
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror
    c.no_subdir = opts.no_subdir
    # s3://mosaicml-internal-checkpoints-shared/aaron/learned-diffusion/v0/vdm/
    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.g_batch_gpu = opts.g_batch_gpu or opts.batch // opts.gpus
    c.d_batch_gpu = opts.d_batch_gpu or opts.batch // opts.gpus
    
    
    if opts.preset == 'FFHQ256':
        WidthPerStage = [3 * x // 4 for x in [1024, 1024, 1024, 1024, 512, 256, 128]]
        BlocksPerStage = [2 * x for x in [1, 1, 1, 1, 1, 1, 1]]
        CardinalityPerStage = [3 * x for x in [32, 32, 32, 32, 16, 8, 4]]
        FP16Stages = [-1, -2, -3, -4]
        NoiseDimension = 64
        
        ema_nimg = 500 * 1000
        
    elif opts.preset == 'cifar':
        WidthPerStage = [3 * x // 4 for x in [1024, 1024, 1024, 1024]]
        BlocksPerStage = [2 * x for x in [1, 1, 1, 1]]
        CardinalityPerStage = [3 * x for x in [32, 32, 32, 32]]
        FP16Stages = [-1, -2, -3]
        NoiseDimension = 64
        
        c.G_kwargs.ConditionEmbeddingDimension = NoiseDimension
        c.D_kwargs.ConditionEmbeddingDimension = WidthPerStage[0]
        
        ema_nimg = 5000 * 1000
        decay_nimg = 2e7
        
        c.ema_scheduler = { 'base_value': 0, 'final_value': ema_nimg, 'total_nimg': decay_nimg }
        c.ada_scheduler = { 'base_value': 1.01, 'final_value': 0.35, 'total_nimg': decay_nimg }
        c.lr_scheduler = { 'base_value': 2e-4, 'final_value': 5e-5, 'total_nimg': decay_nimg }
        c.gamma_scheduler = { 'base_value': 0.1, 'final_value': 0.01, 'total_nimg': decay_nimg }
        c.beta_scheduler = { 'base_value': 0.9, 'final_value': 0.999, 'total_nimg': decay_nimg }
    
    elif opts.preset == 'imagenet':
        WidthPerStage = [6 * x // 4 for x in [1024, 1024, 1024, 1024, 512]]
        BlocksPerStage = [2 * x for x in [1, 1, 1, 1, 1]]
        CardinalityPerStage = [3 * x for x in [32, 32, 32, 32, 16]]
        FP16Stages = [-1, -2, -3, -4]
        NoiseDimension = 64
        
        c.G_kwargs.ConditionEmbeddingDimension = NoiseDimension
        c.D_kwargs.ConditionEmbeddingDimension = WidthPerStage[0]
        
        ema_nimg = 50000 * 1000
        decay_nimg = 2e8
        
        c.ema_scheduler = { 'base_value': 0, 'final_value': ema_nimg, 'total_nimg': decay_nimg }
        c.ada_scheduler = { 'base_value': 1.01, 'final_value': 0.35, 'total_nimg': decay_nimg }
        c.lr_scheduler = { 'base_value': 2e-4, 'final_value': 5e-5, 'total_nimg': decay_nimg }
        c.gamma_scheduler = { 'base_value': 1, 'final_value': 0.1, 'total_nimg': decay_nimg }
        c.beta_scheduler = { 'base_value': 0.9, 'final_value': 0.999, 'total_nimg': decay_nimg }
    
    elif opts.preset == 'imagenet32':
        WidthPerStage = [6 * x // 4 for x in [1024, 1024, 1024, 1024]]
        BlocksPerStage = [2 * x for x in [1, 1, 1, 1]]
        CardinalityPerStage = [3 * x for x in [32, 32, 32, 32]]
        FP16Stages = [-1, -2, -3]
        NoiseDimension = 64
        
        c.G_kwargs.ConditionEmbeddingDimension = NoiseDimension
        c.D_kwargs.ConditionEmbeddingDimension = WidthPerStage[0]
        
        ema_nimg = 50000 * 1000
        decay_nimg = 2e8
        
        c.ema_scheduler = { 'base_value': 0, 'final_value': ema_nimg, 'total_nimg': decay_nimg }
        c.ada_scheduler = { 'base_value': 1.01, 'final_value': 0.35, 'total_nimg': decay_nimg }
        c.lr_scheduler = { 'base_value': 2e-4, 'final_value': 5e-5, 'total_nimg': decay_nimg }
        c.gamma_scheduler = { 'base_value': 0.5, 'final_value': 0.05, 'total_nimg': decay_nimg }
        c.beta_scheduler = { 'base_value': 0.9, 'final_value': 0.999, 'total_nimg': decay_nimg }
    
    c.G_kwargs.NoiseDimension = NoiseDimension
    c.G_kwargs.WidthPerStage = WidthPerStage
    c.G_kwargs.CardinalityPerStage = CardinalityPerStage
    c.G_kwargs.BlocksPerStage = BlocksPerStage
    c.G_kwargs.ExpansionFactor = 2
    c.G_kwargs.FP16Stages = FP16Stages
    
    c.D_kwargs.WidthPerStage = [*reversed(WidthPerStage)]
    c.D_kwargs.CardinalityPerStage = [*reversed(CardinalityPerStage)]
    c.D_kwargs.BlocksPerStage = [*reversed(BlocksPerStage)]
    c.D_kwargs.ExpansionFactor = 2
    c.D_kwargs.FP16Stages = [x + len(FP16Stages) for x in FP16Stages]
    
    
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.g_batch_gpu) != 0 or c.batch_size % (c.num_gpus * c.d_batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

            
    # Augmentation.
    if opts.aug != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=0.5, contrast=0.5, lumaflip=0.5, hue=0.5, saturation=0.5, cutout=1)
        if opts.aug == 'fixed':
            c.augment_p = opts.p

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume

    # Performance-related toggles.
    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
