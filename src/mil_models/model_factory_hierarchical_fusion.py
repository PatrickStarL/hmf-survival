import os
import torch

from os.path import join as j_

from src.mil_models.model_PANTHER import PANTHER
from src.mil_models.model_h2t import H2T
from src.mil_models.model_protocount import ProtoCount
from src.mil_models.model_configs import PANTHERConfig, ProtoCountConfig, H2TConfig
# OT/OTConfig depend on an optional third-party optimal-transport submodule that
# is not vendored in this repo (the released mainline only uses model_type='PANTHER').
# They are imported lazily below, only if model_type == 'OT' is actually requested.
from src.mil_models.models.text_baseline import DAttention_Text
from src.mil_models.model_multimodal_hierarchical_fusion import coattn_text_twostage
from src.utils.file_utils import save_pkl, load_pkl


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_embedding_model(args, mode='classification', config_dir=None):
    if config_dir is None:
        config_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'configs'))
    config_name = getattr(args, 'model_histo_config', None) or getattr(args, 'model_config', None)
    if config_name is None:
        raise ValueError("Could not find model config name in args (`model_histo_config` or `model_config`).")

    config_path = os.path.join(config_dir, config_name, 'config.json')
    assert os.path.exists(config_path), f"Config path {config_path} doesn't exist!"

    model_type = getattr(args, 'model_histo_type', None) or getattr(args, 'model_type', None)
    if model_type is None:
        raise ValueError("Could not find model type in args (`model_histo_type` or `model_type`).")

    update_dict = {
        'in_dim': args.in_dim,
        'out_size': args.n_proto,
        'load_proto': args.load_proto,
        'fix_proto': args.fix_proto,
        'proto_path': args.proto_path,
    }

    if mode == 'classification':
        update_dict.update({'n_classes': args.n_classes})
    elif mode == 'survival':
        if args.loss_fn == 'nll':
            update_dict.update({'n_classes': args.n_label_bins})
        elif args.loss_fn in ['cox', 'rank']:
            update_dict.update({'n_classes': 1})
    elif mode == 'emb':
        pass
    else:
        raise NotImplementedError(f"Not implemented for {mode}...")

    if model_type == 'PANTHER':
        update_dict.update({'out_type': args.out_type})
        config = PANTHERConfig.from_pretrained(config_path, update_dict=update_dict)
        model = PANTHER(config=config, mode=mode)
    elif model_type == 'OT':
        from src.mil_models.model_OT import OT
        from src.mil_models.model_configs import OTConfig
        update_dict.update({'out_type': args.out_type})
        config = OTConfig.from_pretrained(config_path, update_dict=update_dict)
        model = OT(config=config, mode=mode)
    elif model_type == 'H2T':
        config = H2TConfig.from_pretrained(config_path, update_dict=update_dict)
        model = H2T(config=config, mode=mode)
    elif model_type == 'ProtoCount':
        config = ProtoCountConfig.from_pretrained(config_path, update_dict=update_dict)
        model = ProtoCount(config=config, mode=mode)
    else:
        raise NotImplementedError(f"Not implemented for {model_type}!")

    return model


def create_multimodal_survival_model(args, omic_sizes=[]):
    if args.loss_fn == 'nll':
        num_classes = args.n_label_bins
    elif args.loss_fn in ['cox', 'rank']:
        num_classes = 1
    else:
        raise NotImplementedError(f"Unsupported loss_fn={args.loss_fn}")

    twostage_types = [
        'twostage_cls',
        'full_twostage',
        'histo_only_twostage',
        'histo_text_twostage',
        'histo_rna_twostage',
        'cls_transformer_twostage',
        'coattn_text_twostage',
    ]
    if args.model_mm_type in twostage_types:
        model = coattn_text_twostage(
            omic_sizes=omic_sizes,
            histo_in_dim=args.feat_dim,
            text_in_dim=getattr(args, 'text_in_dim', args.feat_dim),
            path_proj_dim=getattr(args, 'path_proj_dim', 256),
            num_classes=num_classes,
            num_coattn_layers=args.num_coattn_layers,
            modality=args.model_mm_type,
            histo_model=args.model_histo_type,
            append_embed=args.append_embed,
            group_prototype=args.group_prototype,
            net_indiv=args.net_indiv,
            net_text_combined=args.net_text_combined,
            numOfproto=getattr(args, 'n_proto', 16),
            pathway_token_count=getattr(args, 'pathway_token_count', 16),
            text_target_length=args.text_target_length,
            text_max_length=args.text_max_length,
            text_resizing_model=args.text_resizing_model,
            attn_mode=args.attn_mode,
            residual=args.residual,
            residual_type=args.residual_type,
            pathway_dropout=getattr(args, 'pathway_dropout', 0.0),
            use_gated_fusion=getattr(args, 'use_gated_fusion', False),
            contrastive_weight=getattr(args, 'contrastive_weight', 0.05),
            contrastive_temp=getattr(args, 'contrastive_temp', 0.07),
            contrastive_pairs=getattr(args, 'contrastive_pairs', 'gene_histo'),
            use_path_qformer=getattr(args, 'use_path_qformer', False),
            qformer_num_queries=getattr(args, 'qformer_num_queries', getattr(args, 'n_proto', 16)),
            qformer_depth=getattr(args, 'qformer_depth', 2),
            qformer_num_heads=getattr(args, 'qformer_num_heads', 4),
            qformer_mlp_ratio=getattr(args, 'qformer_mlp_ratio', 2.0),
            qformer_dropout=getattr(args, 'qformer_dropout', 0.1),
            freeze_non_qformer=getattr(args, 'freeze_non_qformer', False),
            force_gene_histo_contrastive=getattr(args, 'force_gene_histo_contrastive', False),
            freeze_conch=getattr(args, 'freeze_conch', True),
            use_pathway_attnpool=getattr(args, 'use_pathway_attnpool', True),
            cls_depth=getattr(args, 'cls_depth', 2),
            cls_num_heads=getattr(args, 'cls_num_heads', 4),
            cls_mlp_ratio=getattr(args, 'cls_mlp_ratio', 2.0),
            cls_dropout=getattr(args, 'cls_dropout', 0.1),
        )
    elif args.model_mm_type == 'text_baseline':
        model = DAttention_Text(n_classes=4, dropout=0.25, act='relu', n_features=getattr(args, 'text_in_dim', 512))
    else:
        raise NotImplementedError(f"Not implemented for model_mm_type={args.model_mm_type}")

    return model



def prepare_emb(datasets, args, mode='classification'):
    model_type = getattr(args, 'model_histo_type', None) or getattr(args, 'model_type', None)
    if model_type is None:
        raise ValueError("Could not find model type in args (`model_histo_type` or `model_type`).")

    print('\nConstructing unsupervised slide embedding...', end=' ')
    embeddings_kwargs = {
        'feats': os.path.basename(os.path.dirname(args.data_source[0])),
        'model_type': model_type,
        'out_size': args.n_proto,
    }

    fpath = "{feats}_plip_{model_type}_embeddings_proto_{out_size}".format(**embeddings_kwargs)
    if model_type == 'PANTHER':
        PANTHER_kwargs = {'tau': args.tau, 'out_type': args.out_type, 'eps': args.ot_eps, 'em_step': args.em_iter}
        name = '_{out_type}_em_{em_step}_eps_{eps}_tau_{tau}'.format(**PANTHER_kwargs)
        fpath += name
    elif model_type == 'OT':
        OT_kwargs = {'out_type': args.out_type, 'eps': args.ot_eps}
        name = '_{out_type}_eps_{eps}'.format(**OT_kwargs)
        fpath += name
    embeddings_fpath = j_(args.split_dir, 'embeddings', fpath + '.pkl')

    if os.path.isfile(embeddings_fpath):
        embeddings = load_pkl(embeddings_fpath)
        for k, loader in datasets.items():
            print(f'\n\tEmbedding already exists! Loading {k}', end=' ')
            loader.dataset.X, loader.dataset.y = embeddings[k]['X'], embeddings[k]['y']
    else:
        os.makedirs(j_(args.split_dir, 'embeddings'), exist_ok=True)
        model = create_embedding_model(args, mode=mode).to(device)
        embeddings = {}
        for split, loader in datasets.items():
            print(f"\nAggregating {split} set features...")
            X, y = model.predict(loader, use_cuda=torch.cuda.is_available())
            loader.dataset.X, loader.dataset.y = X, y
            embeddings[split] = {'X': X, 'y': y}
        save_pkl(embeddings_fpath, embeddings)

    return datasets, embeddings_fpath
