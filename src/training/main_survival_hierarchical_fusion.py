"""
Main entry point for survival downstream tasks
"""

from __future__ import print_function

import argparse
import pdb
import os
from os.path import join as j_
import sys
import re

import numpy as np

# internal imports
from src.utils.file_utils import save_pkl
from src.utils.utils import (seed_torch, array2list, merge_dict, read_splits,
                         parse_model_name, get_current_time, extract_patching_info, resolve_existing_path)

from src.training.trainer_hierarchical_fusion import train
from src.wsi_datasets import WSIOmicsSurvivalDataset
# pytorch imports
import torch
from torch.utils.data import DataLoader

import pandas as pd
import json

from src.wsi_datasets.dataset_utils import collate_fn
from src.wsi_datasets.wsi_survival import WSIOmicsTextSurvivalDataset
PROTO_MODELS = ['PANTHER', 'OT', 'H2T', 'ProtoCount']

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_SPLIT_DIR = os.path.join(PROJECT_ROOT, 'src', 'splits', 'tcga-coadread', 'TCGA_COADREAD_overall_survival_k=4')
DEFAULT_OMICS_DIR = os.path.join(PROJECT_ROOT, 'src', 'data_csvs', 'rna')
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')


def infer_split_k(split_name):
    match = re.search(r'_k=(\d+)$', split_name)
    return int(match.group(1)) if match else 0


def infer_cancer_type_from_split(split_name):
    match = re.match(r'TCGA_([A-Za-z0-9]+)_', split_name)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot infer cancer type from split name: {split_name}")


def sanitize_path_component(name):
    # Windows forbidden chars: <>:"/\\|?*
    return re.sub(r'[<>:\"/\\\\|?*]', '_', str(name))


def infer_source_tag(path):
    path = os.path.normpath(str(path))
    base = os.path.basename(path)
    parent = os.path.basename(os.path.dirname(path))
    if base in {'feats_pt', 'feats_h5', 'reports_pt'} and parent:
        return parent
    return base


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

def build_datasets(csv_splits, model_type, batch_size=1, num_workers=2, train_kwargs={}, val_kwargs={}):
    """
    Construct dataloaders from the data splits
    """
    dataset_splits = {}
    label_bins = None
    for k in csv_splits.keys(): # ['train', 'val', 'test']
        df = csv_splits[k]
        dataset_kwargs = train_kwargs.copy() if (k == 'train') else val_kwargs.copy()
        dataset_kwargs['label_bins'] = label_bins
        dataset = WSIOmicsTextSurvivalDataset(df_histo=df['histo'], df_gene=df['gene'], **dataset_kwargs)

        # If prototype methods, each WSI will have same feature bag dimension and is batchable
        # Otherwise, we need to use batch size of 1 to accommodate to different bag size for each WSI.
        # Alternatively, we can sample same number of patch features per WSI to have larger batch.
        if model_type not in PROTO_MODELS:
            batch_size = batch_size if dataset_kwargs.get('bag_size', -1) > 0 else 1

        if k == 'train':
            scaler = dataset.get_scaler()

        assert scaler is not None, "Omics scaler from train split required"
        dataset.apply_scaler(scaler)

        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=dataset_kwargs['shuffle'], num_workers=num_workers, collate_fn=collate_fn)
        dataset_splits[k] = dataloader
        print(f'split: {k}, n: {len(dataset)}')
        if (args.loss_fn == 'nll') and (k == 'train'):
            label_bins = dataset.get_label_bins()
    return dataset_splits


def main(args):
    if args.train_bag_size == -1:
        args.train_bag_size = args.bag_size
    if args.val_bag_size == -1:
        args.val_bag_size = args.bag_size
    if args.loss_fn != 'nll':
        args.n_label_bins = 0

    if args.model_mm_type.lower() == 'survpath':
        assert args.model_histo_type.lower() == 'mil', "To use SurvPath, the model_type needs to be mil"

    censorship_col = args.target_col.split('_')[0] + '_censorship'
    
    # Specify omics dir
    cancer_type = infer_cancer_type_from_split(args.split_name_clean)
    args.omics_dir = j_(args.omics_dir, args.type_of_path, cancer_type)

    train_kwargs = dict(data_source=args.data_source,
                        reports_dir=args.reports_dir,
                        survival_time_col=args.target_col,
                        censorship_col=censorship_col,
                        n_label_bins=args.n_label_bins,
                        label_bins=None,
                        bag_size=args.train_bag_size,
                        shuffle=True,
                        omics_dir=args.omics_dir,
                        omics_modality=args.omics_modality
                        )

    # use the whole bag at test time
    val_kwargs = dict(data_source=args.data_source,
                      reports_dir=args.reports_dir,
                      survival_time_col=args.target_col,
                      censorship_col=censorship_col,
                      n_label_bins=args.n_label_bins,
                      label_bins=None,
                      bag_size=args.val_bag_size,
                      shuffle=False,
                      omics_dir=args.omics_dir,
                      omics_modality=args.omics_modality
                      )

    all_results, all_dumps = {}, {}

    seed_torch(args.seed)
    csv_splits = read_splits(args)
    print('successfully read splits for: ', list(csv_splits.keys()))
    dataset_splits = build_datasets(csv_splits, 
                                    model_type=args.model_histo_type,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    train_kwargs=train_kwargs,
                                    val_kwargs=val_kwargs)

    # NOTE: eval-only mode (loading a checkpoint via --eval/--checkpoint_path and
    # skipping training) is not supported for this two-stage model -- the legacy
    # `eval.py` helper builds the original single-stage architecture, not this one.
    # Training always goes through `train()`.
    fold_results, fold_dumps = train(dataset_splits, args)

    # Save results
    for split, split_results in fold_results.items():
        all_results[split] = merge_dict({}, split_results) if (split not in all_results.keys()) else merge_dict(all_results[split], split_results)
        save_pkl(j_(args.results_dir, f'{split}_results.pkl'), fold_dumps[split]) # saves per-split, per-fold results to pkl
    
    final_dict = {}
    for split, split_results in all_results.items():
        final_dict.update({f'{metric}_{split}': array2list(val) for metric, val in split_results.items()})
    final_df = pd.DataFrame(final_dict)
    save_name = 'summary.csv'
    final_df.to_csv(j_(args.results_dir, save_name), index=False)
    with open(j_(args.results_dir, save_name + '.json'), 'w') as f:
        f.write(json.dumps(final_dict, sort_keys=True, indent=4))
    
    dump_path = j_(args.results_dir, 'all_dumps.h5')
    save_pkl(dump_path, fold_dumps)

    return final_dict

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
### optimizer settings ###
FEATURE_ENCODER = 'MSTAR_CONCH_CLS'
print('Feature tag (default path only):', FEATURE_ENCODER)
tissue_type = 'COADREAD'
split = 4

parser.add_argument('--max_epochs', type=int, default=20,
                    help='maximum number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate')
parser.add_argument('--wd', type=float, default=0.00001,
                    help='weight decay')
parser.add_argument('--accum_steps', type=int, default=1,
                    help='grad accumulation steps')
parser.add_argument('--opt', type=str, default='adamW',
                    choices=['adamW', 'sgd', 'RAdam'])
parser.add_argument('--lr_scheduler', type=str,
                    choices=['cosine', 'linear', 'constant'], default='cosine')
parser.add_argument('--warmup_steps', type=int,
                    default=-1, help='warmup iterations')
parser.add_argument('--warmup_epochs', type=int,
                    default=1, help='warmup epochs')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--eval', type=bool, default=False)
parser.add_argument('--checkpoint_path', type=str, default=None)
#

### misc ###
parser.add_argument('--print_every', default=100,
                    type=int, help='how often to print')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--num_workers', type=int, default=8)

### Earlystopper args ###
parser.add_argument('--early_stopping', type=int,
                    default=1, help='enable early stopping')
parser.add_argument('--es_min_epochs', type=int, default=3,
                    help='early stopping min epochs')
parser.add_argument('--es_patience', type=int, default=4,
                    help='early stopping min patience')
parser.add_argument('--es_metric', type=str, default='loss',
                    help='early stopping metric')

### model args ###
parser.add_argument('--model_histo_type', type=str, choices=['H2T', 'OT', 'PANTHER', 'ProtoCount', 'MIL'],
                    default='MIL', help='type of histology model')
parser.add_argument('--ot_eps', default=0.1, type=float,
                    help='Strength for entropic constraint regularization for OT')
parser.add_argument('--model_histo_config', type=str,
                    default='PANTHER_default', help="name of model config file")
parser.add_argument('--n_fc_layers', type=int, default=0)
parser.add_argument('--em_iter', type=int, default=1)
parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--out_type', type=str, default='allcat')

# Multimodal args ###
parser.add_argument('--num_coattn_layers', default=1, type=int)
parser.add_argument('--model_mm_type', default='twostage_cls',
                    help='Multimodal model type')
parser.add_argument('--attn_mode', default='full_all_es',
                    help='Attention mode type')
parser.add_argument('--pathway_dropout', type=float, default=0.0,
                    help='Pathway token dropout probability applied on omics tokens before co-attention')
parser.add_argument("--use_gated_fusion", type=str2bool, default=False,
                    help="Enable gated fusion for tri-modal embeddings")
parser.add_argument("--contrastive_weight", type=float, default=0.002,
                    help="Weight for contrastive alignment loss")
parser.add_argument("--contrastive_temp", type=float, default=0.07,
                    help="Temperature for batch InfoNCE contrastive loss")
parser.add_argument("--contrastive_pairs", type=str, default="histo_text",
                    help="Comma-separated modality pairs for contrastive alignment")
parser.add_argument("--force_gene_histo_contrastive", type=str2bool, default=False,
                    help="Force contrastive loss to gene_histo only (legacy compatibility flag)")
parser.add_argument("--use_pathway_attnpool", type=str2bool, default=True,
                    help="Use attention pooling for pathway tokens (False => mean pooling).")

# Pathology Q-Former args (CONCH features are pre-extracted and frozen offline)
parser.add_argument("--use_path_qformer", type=str2bool, default=False,
                    help="Enable lightweight Q-Former on pathology token branch")
parser.add_argument("--qformer_num_queries", type=int, default=16,
                    help="Number of learnable Q-Former queries")
parser.add_argument("--qformer_depth", type=int, default=2,
                    help="Number of Q-Former blocks")
parser.add_argument("--qformer_num_heads", type=int, default=4,
                    help="Q-Former attention heads")
parser.add_argument("--qformer_mlp_ratio", type=float, default=2.0,
                    help="Q-Former FFN expansion ratio")
parser.add_argument("--qformer_dropout", type=float, default=0.1,
                    help="Q-Former dropout")
parser.add_argument("--freeze_non_qformer", type=str2bool, default=False,
                    help="Freeze non-Q-Former modules and train Q-Former only")
parser.add_argument("--freeze_conch", type=str2bool, default=True,
                    help="No-op in this pipeline: CONCH embeddings are pre-extracted and frozen")
parser.add_argument('--append_prob', action='store_true', default=False)
parser.add_argument('--histo_agg', default='mean')
parser.add_argument('--omics_dir', default=DEFAULT_OMICS_DIR)
parser.add_argument('--omics_modality', default='pathway')
parser.add_argument('--type_of_path', default='hallmarks')

parser.add_argument('--residual', default=False)
parser.add_argument('--residual_type', default='all', choices=['all', 'path+hist'])

# Text args ###
# parser.add_argument('--process_text', type=lambda x: x.lower() == 'true', default=False,help="Enable or disable text processing (True or False)")
parser.add_argument('--process_text', default=True, help="Enable or disable text processing (True or False)")
parser.add_argument('--text_target_length', type=int)
parser.add_argument('--text_max_length', type=int)
parser.add_argument('--text_resizing_model', type=str, default='SA_sampling')

parser.add_argument('--net_indiv', default=True)
parser.add_argument('--net_text_combined', default=False)
parser.add_argument('--group_prototype', default=False)
parser.add_argument('--append_embed', type=str, default='random',
                    choices=['none', 'modality', 'proto', 'random', 'random_xavier', 'uniform'])

# Prototype related
parser.add_argument('--load_proto', action='store_true', default=False)
parser.add_argument('--proto_path', type=str)
parser.add_argument('--fix_proto', default=True)
parser.add_argument('--n_proto', type=int, default=16)
parser.add_argument('--pathway_token_count', type=int, default=16,
                    help='number of compressed RNA/pathway tokens before multimodal fusion')

parser.add_argument('--in_dim', default=512, type=int,
                    help='dim of input vision features')
parser.add_argument('--text_in_dim', default=-1, type=int,
                    help='dim of input text features; <=0 means infer from report .pt files')
parser.add_argument('--path_proj_dim', default=256, type=int,
                    help='shared latent dim after separate vision/text projections')
parser.add_argument('--cls_depth', default=2, type=int,
                    help='number of shallow CLS transformer blocks')
parser.add_argument('--cls_num_heads', default=4, type=int,
                    help='attention heads in the shallow CLS transformer')
parser.add_argument('--cls_mlp_ratio', default=2.0, type=float,
                    help='MLP expansion ratio in the shallow CLS transformer')
parser.add_argument('--cls_dropout', default=0.1, type=float,
                    help='dropout in the shallow CLS transformer')
parser.add_argument('--bag_size', type=int, default=16)
parser.add_argument('--train_bag_size', type=int, default='-1')
parser.add_argument('--val_bag_size', type=int, default='-1')
parser.add_argument('--loss_fn', type=str, default='cox', choices=['nll', 'cox', 'sumo', 'ipcwls', 'rank'],
                    help='which loss function to use')
parser.add_argument('--nll_alpha', type=float, default=0.5,
                    help='Balance between censored / uncensored loss')

# experiment task / label args ###
parser.add_argument('--exp_code', type=str, default=None,
                    help='experiment code for saving results')
parser.add_argument('--task', type=str, default=f'{tissue_type}_survival')
parser.add_argument('--target_col', type=str, default='dss_survival_days')
parser.add_argument('--n_label_bins', type=int, default=4,
                    help='number of bins for event time discretization')

# dataset / split args ###
parser.add_argument('--data_source', type=str, default=f'/user/Data/TCGA/{tissue_type}/{FEATURE_ENCODER}_features/feats_pt',
                    help='manually specify the data source')
parser.add_argument('--reports_dir', type=str, default=f'/user/TCGA/{tissue_type}/{FEATURE_ENCODER}_TCGA_Reports/',
                    help='manually specify the reports dir')
parser.add_argument('--split_dir', type=str, default=DEFAULT_SPLIT_DIR,
                    help='manually specify the set of splits to use')
parser.add_argument('--split_names', type=str, default='train,test',
                    help='delimited list for specifying names within each split')
parser.add_argument('--overwrite', default=True,
                    help='overwrite existing results')

# logging args ###
parser.add_argument('--results_dir', default=DEFAULT_RESULTS_DIR,
                    help='results directory (default: ./results)')
parser.add_argument('--tags', nargs='+', type=str, default=None,
                    help='tags for logging')

parser.add_argument('--wandb_project', default='mmp_final')
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    print('task: ', args.task)
    args.split_dir = resolve_existing_path(
        args.split_dir,
        search_roots=[PROJECT_ROOT, os.path.join(PROJECT_ROOT, 'src'), os.path.join(PROJECT_ROOT, 'src', 'splits')]
    )
    args.omics_dir = resolve_existing_path(
        args.omics_dir,
        search_roots=[PROJECT_ROOT, os.path.join(PROJECT_ROOT, 'src')]
    )
    print('split_dir: ', args.split_dir)
    args.split_name_clean = os.path.basename(os.path.normpath(args.split_dir))
    args.split_k = infer_split_k(args.split_name_clean)

    if args.load_proto:
        assert os.path.isfile(args.proto_path), f"Path {args.proto_path} doesn't exist!"
        args.proto_fname = os.path.join(os.path.basename(os.path.dirname(args.proto_path)), os.path.basename(args.proto_path))
        proto_fname_clean = '--'.join(args.proto_fname[:-4].split(os.sep))

    if args.eval and not args.checkpoint_path:
        raise ValueError("`--checkpoint_path` is required when `--eval=True`.")

    if args.use_path_qformer and args.model_histo_type.upper() != 'MIL':
        print(f"[QFormer] model_histo_type is {args.model_histo_type}; overriding to MIL.")
        args.model_histo_type = 'MIL'

    if args.force_gene_histo_contrastive:
        args.contrastive_pairs = 'gene_histo'

    if args.loss_fn != 'cox':
        print(f"[QFormer] loss_fn is {args.loss_fn}; overriding to cox.")
        args.loss_fn = 'cox'

    if args.freeze_conch:
        print('[QFormer] CONCH is pre-extracted as fixed input features in this pipeline.')

    ### Allows you to pass in multiple data sources (separated by comma). If single data source, no change.
    args.data_source = [src for src in args.data_source.split(',')]
    check_params_same = []
    vision_source_tags = []
    for src in args.data_source: 
        ### assert data source exists + extract feature name ###
        print('data source: ', src)
        assert os.path.isdir(src), f"data source must be a directory: {src} invalid"

        ### parse patching info ###
        feat_name = os.path.basename(src)
        print('resolved feature source: ', feat_name)
        vision_source_tags.append(infer_source_tag(src))


        #### parse model name ####
        parsed = parse_model_name(feat_name)
        parsed.update({'patch_mag': 20, 'patch_size': 224})

        
    ### Updated parsed mdoel parameters in args.Namespace ###
    for key, val in parsed.items():
        setattr(args, key, val)

    args.vision_source_tag = '__'.join(vision_source_tags) if vision_source_tags else 'unknown_vision'
    args.text_source_tag = infer_source_tag(args.reports_dir)

    ### Updated text in args.Namespace ###
    csv_file = os.path.join(args.split_dir, 'train.csv')

    # Read the case IDs from the CSV file
    case_ids = pd.read_csv(csv_file)['case_id'].astype(str).tolist()  # Assuming column is named 'case_id'

    # Filter .pt files that match the case IDs
    pt_files = [os.path.join(args.reports_dir, f) for f in os.listdir(args.reports_dir)
                if f.endswith('.pt') and os.path.splitext(f)[0] in case_ids]

    if not pt_files:
        raise FileNotFoundError("No matching files found in the directory.")

    # Extract lengths and embedding dimensions
    lengths = []
    text_dims = []
    for file in pt_files:
        embedding = torch.load(file, weights_only=False)  # Load .pt file
        if len(embedding.shape) != 2:
            raise ValueError(f"Expected 2D text embeddings in {file}, got shape={tuple(embedding.shape)}")
        lengths.append(embedding.shape[0])  # Get number of chunks (first dimension)
        text_dims.append(embedding.shape[1])

    unique_text_dims = sorted(set(int(x) for x in text_dims))
    if len(unique_text_dims) != 1:
        raise ValueError(f"Inconsistent text feature dims found: {unique_text_dims}")
    inferred_text_in_dim = unique_text_dims[0]
    if int(args.text_in_dim) <= 0:
        args.text_in_dim = inferred_text_in_dim
    elif int(args.text_in_dim) != inferred_text_in_dim:
        raise ValueError(
            f"Configured text_in_dim={args.text_in_dim} but inferred {inferred_text_in_dim} from reports_dir={args.reports_dir}"
        )
    else:
        args.text_in_dim = int(args.text_in_dim)

    # Analyze length statistics
    max_length = max(lengths)
    mean_length = int(np.mean(lengths))
    median_length = int(np.median(lengths))
    percentile_90 = int(np.percentile(lengths, 90))

    text_parameters = {
            "text_max_length": max_length,
            "text_target_length": mean_length,
        }

    for key, val in text_parameters.items():
        setattr(args, key, val)

    print(f'Text Target Length : {args.text_target_length}')
    print(f'Text Max Length : {args.text_max_length}')
    print(f'Text Input Dim : {args.text_in_dim}')
    print(f'Vision Input Dim : {getattr(args, "feat_dim", args.in_dim)}')
    print(f'Shared Projection Dim : {args.path_proj_dim}')

    vision_name = getattr(args, 'vision_source_tag', 'vision').replace('features_', '')
    text_name = getattr(args, 'text_source_tag', 'text').replace('features_', '')
    mixed_feature_tag = f"v_{vision_name}__t_{text_name}"

    ### setup results dir ### es = extra softmax
    if args.exp_code is None:
        if args.process_text:
            if args.model_mm_type == 'text_baseline':
                exp_code = f"{args.split_name_clean}::{args.model_histo_config}::{mixed_feature_tag}::{args.model_mm_type}"
            else:
                exp_code = (f"{args.split_name_clean}::{args.model_histo_config}::{mixed_feature_tag}::{args.model_mm_type}:"
                        f"{args.text_target_length}::td_{args.text_in_dim}::net_indiv_{args.net_indiv}::append_embed_{args.append_embed}::attn_mode_{args.attn_mode}::pdrop_{args.pathway_dropout}::gf_{int(args.use_gated_fusion)}::cw_{args.contrastive_weight}::ct_{args.contrastive_temp}::cp_{args.contrastive_pairs}::pap_{int(args.use_pathway_attnpool)}::qf_{int(args.use_path_qformer)}::qn_{args.qformer_num_queries}::qd_{args.qformer_depth}::qh_{args.qformer_num_heads}::qfr_{int(args.freeze_non_qformer)}::clsd_{args.cls_depth}::clsh_{args.cls_num_heads}::clsm_{args.cls_mlp_ratio}::clsdp_{args.cls_dropout}::ptok_{args.pathway_token_count}")
        else:
            exp_code = f"{args.split_name_clean}::{args.model_histo_config}::{mixed_feature_tag}::{args.model_mm_type}"
    else:
        exp_code = args.exp_code


    args.results_dir = j_(args.results_dir, 
                          args.task, 
                          f'k={args.split_k}', 
                          sanitize_path_component(exp_code))

    os.makedirs(args.results_dir, exist_ok=True)


    print("\n################### Settings ###################")
    for key, val in vars(args).items():
        print("{}:  {}".format(key, val))

    with open(j_(args.results_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))

    #### train ####
    results = main(args)

    print("FINISHED!\n\n\n")
