import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.utils.utils import safe_list_to
from .components import SNN_Block, FeedForward, FeedForwardEnsemble, process_surv, Attn_Net_Gated, \
    CoAttentionLayer, FeedForwardEnsemble_combined_text
from .text_processing import SelfAttentionResizer, interpolate_to_fixed_length


def init_per_path_model(omic_sizes, hidden_dim=256):
    """
    Create a list of SNNs, one for each pathway

    Args:
        omic_sizes: List of integers, each indicating number of genes per prototype
    """
    hidden = [hidden_dim, hidden_dim]
    sig_networks = []
    for input_dim in omic_sizes:
        fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
        sig_networks.append(nn.Sequential(*fc_omic))
    sig_networks = nn.ModuleList(sig_networks)

    return sig_networks

def agg_histo(X, agg_mode='mean'):
    """
    Aggregating histology
    """
    if agg_mode == 'mean':
        out = torch.mean(X, dim=1)
    elif agg_mode == 'cat':
        out = X.reshape(X.shape[0], -1)
    else:
        raise NotImplementedError(f"Not implemented for {agg_mode}")

    return out



def construct_proto_embedding(path_proj_dim, append_embed='modality', numOfproto_histo=16, numOfproto_omics=50):
    """
    Per-prototype learnable/non-learnable embeddings to append to the original prototype embeddings 
    """
    if append_embed == 'modality':  # One-hot encoding for two modalities
        path_proj_dim_new = path_proj_dim + 2

        histo_embedding = torch.tensor([[[1, 0]]]).repeat(1, numOfproto_histo, 1)  # (1, numOfproto, 2)
        gene_embedding = torch.tensor([[[0, 1]]]).repeat(1, numOfproto_omics, 1)  # (1, len(omic_sizes),2 )

    elif append_embed == 'proto':
        path_proj_dim_new = path_proj_dim + numOfproto_histo + numOfproto_omics
        embedding = torch.eye(numOfproto_histo + numOfproto_omics).unsqueeze(0)

        histo_embedding = embedding[:, :numOfproto_histo, :]  # (1, numOfproto, numOftotalproto)
        gene_embedding = embedding[:, numOfproto_histo:, :]  # (1, len(omic_sizes), numOftotalproto)

    elif append_embed == 'random':
        append_dim = 32
        path_proj_dim_new = path_proj_dim + append_dim

        histo_embedding = torch.nn.Parameter(torch.randn(1, numOfproto_histo, append_dim), requires_grad=True)
        gene_embedding = torch.nn.Parameter(torch.randn(1, numOfproto_omics, append_dim), requires_grad=True)

    else:
        path_proj_dim_new = path_proj_dim
        histo_embedding = None
        gene_embedding = None

    return path_proj_dim_new, histo_embedding, gene_embedding

def construct_proto_embedding_text(path_proj_dim, append_embed='modality', numOfproto_histo=16, numOfproto_omics=50, numOfproto_text=43):
    """
    Per-prototype learnable/non-learnable embeddings to append to the original prototype embeddings
    """
    if append_embed == 'modality':  # One-hot encoding for two modalities
        path_proj_dim_new = path_proj_dim + 3  # Add 3 dimensions for three modalities

        histo_embedding = torch.tensor([[[1, 0, 0]]]).repeat(1, numOfproto_histo,
                                                             1)  # Histology (1, numOfproto_histo, 3)
        gene_embedding = torch.tensor([[[0, 1, 0]]]).repeat(1, numOfproto_omics, 1)  # Omics (1, numOfproto_omics, 3)
        text_embedding = torch.tensor([[[0, 0, 1]]]).repeat(1, numOfproto_text, 1)  # New modality

    elif append_embed == 'proto':
        path_proj_dim_new = path_proj_dim + numOfproto_histo + numOfproto_omics + numOfproto_text
        embedding = torch.eye(numOfproto_histo + numOfproto_omics+ numOfproto_text).unsqueeze(0)

        histo_embedding = embedding[:, :numOfproto_histo, :]  # (1, numOfproto_histo, numOftotalproto)
        gene_embedding = embedding[:, numOfproto_histo:numOfproto_histo + numOfproto_omics, :]  # (1, numOfproto_omics, numOftotalproto)
        text_embedding = embedding[:, numOfproto_histo + numOfproto_omics:, :]  # (1, numOfproto_new_modality, numOftotalproto)

    elif append_embed == 'random':
        append_dim = 32
        path_proj_dim_new = path_proj_dim + append_dim

        histo_embedding = torch.nn.Parameter(torch.randn(1, numOfproto_histo, append_dim), requires_grad=True)
        gene_embedding = torch.nn.Parameter(torch.randn(1, numOfproto_omics, append_dim), requires_grad=True)
        text_embedding = torch.nn.Parameter(torch.randn(1, numOfproto_text, append_dim), requires_grad=True)

    elif append_embed == 'random_xavier':
        append_dim = 32
        path_proj_dim_new = path_proj_dim + append_dim
        histo_embedding = torch.nn.Parameter(torch.empty(1, numOfproto_histo, append_dim), requires_grad=True)
        gene_embedding = torch.nn.Parameter(torch.empty(1, numOfproto_omics, append_dim), requires_grad=True)
        text_embedding = torch.nn.Parameter(torch.empty(1, numOfproto_text, append_dim), requires_grad=True)
        nn.init.xavier_uniform_(histo_embedding)
        nn.init.xavier_uniform_(gene_embedding)
        nn.init.xavier_uniform_(text_embedding)

    elif append_embed == 'uniform':
        append_dim = 32
        path_proj_dim_new = path_proj_dim + append_dim
        histo_embedding = torch.nn.Parameter(torch.empty(1, numOfproto_histo, append_dim).uniform_(-0.1, 0.1), requires_grad=True)
        gene_embedding = torch.nn.Parameter(torch.empty(1, numOfproto_omics, append_dim).uniform_(-0.1, 0.1),requires_grad=True)
        text_embedding = torch.nn.Parameter(torch.empty(1, numOfproto_text, append_dim).uniform_(-0.1, 0.1),requires_grad=True)

    else:
        path_proj_dim_new = path_proj_dim
        histo_embedding = None
        gene_embedding = None
        text_embedding = None

    return path_proj_dim_new, histo_embedding, gene_embedding, text_embedding


def construct_proto_embedding_grouped_text(path_proj_dim, append_embed='modality', numOfproto_histo=16, numOfproto_omics=50, numOfproto_text=1, text_batch_size=20):
    """
    Per-prototype learnable/non-learnable embeddings to append to the original prototype embeddings
    """
    if append_embed == 'modality':  # One-hot encoding for two modalities
        path_proj_dim_new = path_proj_dim + 3  # Add 3 dimensions for three modalities

        histo_embedding = torch.tensor([[[1, 0, 0]]]).repeat(1, numOfproto_histo, 1)  # Histology (1, numOfproto_histo, 3)
        gene_embedding = torch.tensor([[[0, 1, 0]]]).repeat(1, numOfproto_omics, 1)  # Omics (1, numOfproto_omics, 3)
        text_embedding = torch.tensor([[[0, 0, 1]]]).repeat(1, text_batch_size, 1)  # New modality

    elif append_embed == 'proto':
        path_proj_dim_new = path_proj_dim + numOfproto_histo + numOfproto_omics + numOfproto_text
        embedding = torch.eye(numOfproto_histo + numOfproto_omics + numOfproto_text).unsqueeze(0)

        histo_embedding = embedding[:, :numOfproto_histo, :]  # (1, numOfproto_histo, numOftotalproto)
        gene_embedding = embedding[:, numOfproto_histo:numOfproto_histo + numOfproto_omics, :]  # (1, numOfproto_omics, numOftotalproto)
        text_embedding = embedding[:, numOfproto_histo + numOfproto_omics:, :].repeat(1, text_batch_size, 1)

    elif append_embed == 'random':
        append_dim = 32
        path_proj_dim_new = path_proj_dim + append_dim

        histo_embedding = torch.nn.Parameter(torch.randn(1, numOfproto_histo, append_dim), requires_grad=True)
        gene_embedding = torch.nn.Parameter(torch.randn(1, numOfproto_omics, append_dim), requires_grad=True)
        text_embedding = torch.nn.Parameter(torch.randn(1, numOfproto_text, append_dim), requires_grad=True).repeat(1, text_batch_size, 1)

    else:
        path_proj_dim_new = path_proj_dim
        histo_embedding = None
        gene_embedding = None
        text_embedding = None

    return path_proj_dim_new, histo_embedding, gene_embedding, text_embedding



class LightweightQFormerBlock(nn.Module):
    # Lightweight Q-Former block with cross-attention + self-attention + FFN.
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm_q_cross = nn.LayerNorm(dim)
        self.norm_kv_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        hidden_dim = int(dim * mlp_ratio)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_tokens, context_tokens):
        q = self.norm_q_cross(query_tokens)
        kv = self.norm_kv_cross(context_tokens)
        cross_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        query_tokens = query_tokens + self.dropout(cross_out)

        q2 = self.norm_q_self(query_tokens)
        self_out, _ = self.self_attn(q2, q2, q2, need_weights=False)
        query_tokens = query_tokens + self.dropout(self_out)

        query_tokens = query_tokens + self.dropout(self.ffn(self.norm_ffn(query_tokens)))
        return query_tokens


class LightweightQFormer(nn.Module):
    # A compact Q-Former for pathology tokens.
    def __init__(self, in_dim, num_queries=16, depth=2, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.num_queries = int(num_queries)
        self.query_tokens = nn.Parameter(torch.randn(1, self.num_queries, in_dim) * 0.02)
        self.blocks = nn.ModuleList([
            LightweightQFormerBlock(
                dim=in_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(max(1, int(depth)))
        ])
        self.final_norm = nn.LayerNorm(in_dim)

    def forward(self, x_path):
        # x_path: (B, N_patch, C)
        batch_size = x_path.size(0)
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        for block in self.blocks:
            query_tokens = block(query_tokens, x_path)
        return self.final_norm(query_tokens)

################################
# Multimodal fusion approaches #
################################

class coattn_text(nn.Module):
    def __init__(
            self,
            omic_sizes=[100, 200, 300, 400, 500, 600],
            histo_in_dim=1024,
            text_in_dim=None,
            dropout=0.1,
            num_classes=4,
            path_proj_dim=256,
            num_coattn_layers=1,
            modality='both',
            histo_agg='mean',
            histo_model='PANTHER',
            append_embed='none',
            group_prototype = False,
            mult=1,
            net_indiv=False,
            net_text_combined = False,
            numOfproto=16,
            text_target_length = 43,
            text_max_length = 60,
            text_resizing_model = None,
            attn_mode = None,
            residual = False,
            residual_type = 'all',
            pathway_dropout=0.0,
            use_gated_fusion=True,
            contrastive_weight=0.05,
            contrastive_temp=0.07,
            contrastive_pairs='gene_histo,gene_text,text_histo',
            use_path_qformer=False,
            qformer_num_queries=16,
            qformer_depth=2,
            qformer_num_heads=4,
            qformer_mlp_ratio=2.0,
            qformer_dropout=0.1,
            freeze_non_qformer=False,
            force_gene_histo_contrastive=False,
            freeze_conch=True,
            use_pathway_attnpool=True,
    ):
        """
        The central co-attention module where you can do it all!

        Args:
            omic_sizes: List of integers, each indicating number of genes per prototype
            histo_in_dim: Dimension of histology feature embedding
            num_classes: 4 if we are using NLL, 1 if we are using Cox/Ranking loss
            path_proj_dim: Dimension of the embedding space where histology and pathways are fused
            modality: ['gene','histo','coattn', 'partial'] 'coattn' accounts for both modalities
                If 'histo' or 'gene', unimodal self-attention
            histo_agg: ['mean', 'cat'] Take average of post-attention embeddings ('mean') or concatenate ('cat')
            histo_model: ['mil','PANTHER']: 'mil' is for non-prototype-based methods
            net_indiv (bool): If True, create FFN for each prototype
            numOfproto: Number of histology prototypes
        """

        super().__init__()

        self.num_pathways = len(omic_sizes)
        self.num_coattn_layers = num_coattn_layers

        self.histo_in_dim = histo_in_dim
        self.out_mult = mult
        self.net_indiv = net_indiv
        self.net_text_combined = net_text_combined
        self.group_prototype = group_prototype
        self.modality = modality
        self.pathway_dropout = max(0.0, min(float(pathway_dropout), 0.95))
        self.use_gated_fusion = bool(use_gated_fusion)
        self.contrastive_weight = max(0.0, float(contrastive_weight))
        self.contrastive_temp = max(1e-4, float(contrastive_temp))
        self.force_gene_histo_contrastive = bool(force_gene_histo_contrastive)
        if self.force_gene_histo_contrastive:
            self.contrastive_pairs = ['gene_histo']
        else:
            self.contrastive_pairs = [p.strip().lower() for p in str(contrastive_pairs).split(",") if p.strip()]
        self.use_pathway_attnpool = bool(use_pathway_attnpool)

        self.histo_agg = histo_agg

        self.numOfproto = numOfproto
        self.num_classes = num_classes
        self.freeze_non_qformer = bool(freeze_non_qformer)
        self.freeze_conch = bool(freeze_conch)  # no-op: CONCH is pre-extracted offline in this pipeline

        self.histo_model = histo_model.lower()
        self.use_path_qformer = bool(use_path_qformer)
        self.qformer_num_queries = int(qformer_num_queries) if int(qformer_num_queries) > 0 else int(self.numOfproto)
        self.num_histo_tokens = int(self.qformer_num_queries if self.use_path_qformer else self.numOfproto)
        if self.use_path_qformer and self.histo_model != 'mil':
            raise ValueError("`use_path_qformer=True` currently supports `model_histo_type=MIL` only.")

        self.sig_networks = init_per_path_model(omic_sizes)
        self.identity = nn.Identity()  # use this layer to calculate ig

        self.append_embed = append_embed

        self.text_target_length = text_target_length
        self.max_text_length = text_max_length
        self.text_resizing_model = text_resizing_model
        self.text_in_dim = int(text_in_dim) if text_in_dim is not None else self.histo_in_dim
        self.text_resize_dim = self.text_in_dim

        if self.text_resizing_model == 'SA_sampling':
            self.text_resizer = SelfAttentionResizer(input_dim=self.text_in_dim, max_length=self.max_text_length, target_length=self.text_target_length, aggregation_method='sampling')

        if self.histo_model == 'panther':  # Uses prob/mean/cov
            self.path_proj_net = nn.Sequential(nn.Linear(self.histo_in_dim * 2 + 1, path_proj_dim))
        else:
            self.path_proj_net = nn.Sequential(nn.Linear(self.histo_in_dim, path_proj_dim))
        if self.use_path_qformer:
            self.path_qformer = LightweightQFormer(
                in_dim=self.histo_in_dim,
                num_queries=self.qformer_num_queries,
                depth=qformer_depth,
                num_heads=qformer_num_heads,
                mlp_ratio=qformer_mlp_ratio,
                dropout=qformer_dropout,
            )
        else:
            self.path_qformer = None

        self.text_proj_net = nn.Sequential(nn.Linear(self.text_resize_dim, path_proj_dim))

        if self.histo_model != "mil":
            if self.group_prototype:
                self.path_proj_dim, self.histo_embedding, self.gene_embedding, self.text_embedding = (construct_proto_embedding_grouped_text
                                                                                                      (path_proj_dim,
                                                                                                      self.append_embed,
                                                                                                      self.numOfproto,
                                                                                                      len(omic_sizes),
                                                                                                       1,
                                                                                                       text_target_length))
            else:
                self.path_proj_dim, self.histo_embedding, self.gene_embedding, self.text_embedding = construct_proto_embedding_text(path_proj_dim,
                                                                                                      self.append_embed,
                                                                                                      self.numOfproto,
                                                                                                      len(omic_sizes),
                                                                                                      text_target_length)
        else:
            self.path_proj_dim = path_proj_dim
            self.histo_embedding = None
            self.gene_embedding = None
            self.text_embedding = None

        coattn_list = []
        if self.num_coattn_layers == 0:
            out_dim = self.path_proj_dim

            if self.net_indiv:  # Individual MLP per prototype
                feed_forward = FeedForwardEnsemble(out_dim,
                                                   self.out_mult,
                                                   dropout=dropout,
                                                   num=self.num_histo_tokens + len(omic_sizes)+ self.text_target_length)
            else:
                feed_forward = FeedForward(out_dim, self.out_mult, dropout=dropout)

            layer_norm = nn.LayerNorm(int(out_dim * self.out_mult))
            coattn_list.extend([feed_forward, layer_norm])
        else:
            out_dim = self.path_proj_dim // 2
            out_mult = self.out_mult

            if self.modality in ['histo', 'gene', 'text']:  # If we want to use only single modality + self-attention
                attn_mode = 'self'
            else:
                attn_mode = attn_mode  # Otherwise, perform self & cross attention
            print('Attn Mode: ', attn_mode)
            cross_attender = CoAttentionLayer(
                dim=self.path_proj_dim,
                dim_head=out_dim,
                heads=1,
                residual=residual,
                dropout=0.1,
                num_pathways=self.num_pathways,
                num_prototypes=self.num_histo_tokens,
                attn_mode=attn_mode,
                residual_type='all'
            )

            if self.net_indiv:
                if self.net_text_combined:
                    feed_forward = FeedForwardEnsemble_combined_text(out_dim, # Individual MLP per prototype but combined for text
                                                   out_mult,
                                                   dropout=dropout,
                                                   num=self.num_histo_tokens + len(omic_sizes))
                else:
                    feed_forward = FeedForwardEnsemble(out_dim, # Individual MLP per prototype and for text token
                                                   out_mult,
                                                   dropout=dropout,
                                                   num=self.num_histo_tokens + len(omic_sizes) + self.text_target_length)
            else:
                feed_forward = FeedForward(out_dim, out_mult, dropout=dropout)

            layer_norm = nn.LayerNorm(int(out_dim * out_mult))
            coattn_list.extend([cross_attender, feed_forward, layer_norm])

        self.coattn = nn.Sequential(*coattn_list)

        out_dim_final = int(out_dim * self.out_mult)
        self.pathway_attn_pool = nn.Linear(out_dim_final, 1)
        histo_final_dim = out_dim_final * self.num_histo_tokens if self.histo_agg == 'cat' else out_dim_final
        gene_final_dim = out_dim_final
        text_final_dim = out_dim_final
        self.gene_final_dim = gene_final_dim
        self.histo_final_dim = histo_final_dim
        self.text_final_dim = text_final_dim

        if self.modality == 'histo':
            in_dim = histo_final_dim
        elif self.modality == 'gene':
            in_dim = gene_final_dim
        elif self.modality == 'text':
            in_dim = text_final_dim
        elif self.modality == 'histo+text':
            in_dim = histo_final_dim + text_final_dim
        elif self.modality == 'pathways+text':
            in_dim = gene_final_dim + text_final_dim
        else:
            in_dim = histo_final_dim + gene_final_dim + text_final_dim

        self.classifier = nn.Linear(in_dim, self.num_classes, bias=False)
        self.gene_gate_proj = nn.Linear(self.gene_final_dim, out_dim_final)
        self.histo_gate_proj = nn.Linear(self.histo_final_dim, out_dim_final)
        self.text_gate_proj = nn.Linear(self.text_final_dim, out_dim_final)
        self.gate_mlp = nn.Sequential(
            nn.Linear(out_dim_final * 3, out_dim_final),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim_final, 3)
        )

        self.contrastive_gene_proj = nn.Linear(self.gene_final_dim, out_dim_final)
        self.contrastive_histo_proj = nn.Linear(self.histo_final_dim, out_dim_final)
        self.contrastive_text_proj = nn.Linear(self.text_final_dim, out_dim_final)
        self._configure_trainable_params()

    def _configure_trainable_params(self):
        # Optionally freeze everything except pathology Q-Former.
        if not self.freeze_non_qformer:
            return
        if self.path_qformer is None:
            raise ValueError("freeze_non_qformer=True requires use_path_qformer=True.")

        qformer_param_ids = {id(p) for p in self.path_qformer.parameters()}
        for _, param in self.named_parameters():
            param.requires_grad = id(param) in qformer_param_ids

    @staticmethod
    def _normalize_pair_name(name):
        aliases = {
            'gene': 'gene',
            'path': 'gene',
            'pathway': 'gene',
            'pathways': 'gene',
            'histo': 'histo',
            'histology': 'histo',
            'wsi': 'histo',
            'text': 'text',
        }
        return aliases.get(name, name)

    def _pair_infonce(self, emb_a, emb_b):
        if emb_a.size(0) < 2 or emb_b.size(0) < 2:
            return emb_a.new_zeros(())

        emb_a = F.normalize(emb_a, p=2, dim=1)
        emb_b = F.normalize(emb_b, p=2, dim=1)

        logits = torch.matmul(emb_a, emb_b.t()) / self.contrastive_temp
        targets = torch.arange(logits.size(0), device=logits.device)
        loss_ab = F.cross_entropy(logits, targets)
        loss_ba = F.cross_entropy(logits.t(), targets)
        return 0.5 * (loss_ab + loss_ba)

    def _compute_contrastive_loss(self, gene_embed, histo_embed, text_embed):
        if gene_embed.size(0) < 2:
            return gene_embed.new_zeros(())

        embed_map = {
            'gene': self.contrastive_gene_proj(gene_embed),
            'histo': self.contrastive_histo_proj(histo_embed),
            'text': self.contrastive_text_proj(text_embed),
        }

        losses = []
        for pair in self.contrastive_pairs:
            clean_pair = pair.replace('-', '_')
            parts = [self._normalize_pair_name(p.strip()) for p in clean_pair.split('_') if p.strip()]
            if len(parts) != 2:
                continue
            left, right = parts
            if left not in embed_map or right not in embed_map or left == right:
                continue
            losses.append(self._pair_infonce(embed_map[left], embed_map[right]))

        if not losses:
            return gene_embed.new_zeros(())

        return torch.stack(losses).mean()

    def _resize_mil_histo_tokens(self, x_path):
        """
        Keep MIL pathology tokens compatible with fixed-size co-attention blocks.

        Training usually samples a fixed `bag_size`, but test-time `val_bag_size=0`
        can pass the whole slide bag with variable token counts. We compress those
        variable-length patch tokens to `self.num_histo_tokens` so the downstream
        co-attention stack can keep using a fixed token budget.
        """
        if self.histo_model != 'mil' or self.path_qformer is not None:
            return x_path

        current_tokens = x_path.size(1)
        target_tokens = self.num_histo_tokens
        if current_tokens == target_tokens:
            return x_path

        x_path_t = x_path.transpose(1, 2)
        if current_tokens > target_tokens:
            x_path_t = F.adaptive_avg_pool1d(x_path_t, target_tokens)
        else:
            x_path_t = F.interpolate(x_path_t, size=target_tokens, mode='linear', align_corners=False)
        return x_path_t.transpose(1, 2)

    def forward_no_loss(self, x_path, x_omics, x_text, return_attn=False):
        """
        Args:
            x_path: (B, numOfproto, in_dim) in_dim = [prob, mean, cov] (PANTHER prototype statistics)
            x_omics:
            return_attn:

        """
        device = x_path.device

        ## Pathway embeddings
        h_omic = []  ## each omic signature goes through it's own FC layer
        for idx, sig_feat in enumerate(x_omics):
            omic_feat = self.sig_networks[idx](sig_feat.float())  # (B, d)
            h_omic.append(omic_feat)
        h_omic = torch.stack(h_omic, dim=1)  # [batch_size, 50, out_feats = 256]

        if self.gene_embedding is not None:  # Append gene prototype encoding
            arr = []
            for idx in range(len(h_omic)):
                arr.append(torch.cat([h_omic[idx:idx + 1], self.gene_embedding.to(device)], dim=-1))
            h_omic = torch.cat(arr, dim=0)  # [batch_size, 50, out_feats + prototype encoding = 288 ]

        if self.training and self.pathway_dropout > 0:
            keep_prob = 1.0 - self.pathway_dropout
            pathway_keep = (torch.rand(h_omic.size(0), h_omic.size(1), device=device) < keep_prob).float()
            all_dropped = pathway_keep.sum(dim=1) == 0
            if all_dropped.any():
                random_idx = torch.randint(low=0, high=h_omic.size(1), size=(int(all_dropped.sum().item()),), device=device)
                pathway_keep[all_dropped, random_idx] = 1.0
            h_omic = h_omic * pathway_keep.unsqueeze(-1) / keep_prob

        ## Histology embeddings
        x_path = self._resize_mil_histo_tokens(x_path)
        if self.path_qformer is not None:
            x_path = self.path_qformer(x_path)

        # Project wsi to smaller dimension (same as pathway dimension)
        h_path = self.path_proj_net(x_path)  # [batch_size, 16, out_feats = 256 ]

        if self.histo_embedding is not None:  # Append histo prototype encoding
            arr = []
            for idx in range(len(h_path)):
                arr.append(torch.cat([h_path[idx:idx + 1], self.histo_embedding.to(device)], dim=-1))
            h_path = torch.cat(arr, dim=0)  # [batch_size, 50, out_feats + prototype encoding = 288]

        if self.text_resizing_model == 'Interpolation':
            h_text = interpolate_to_fixed_length(x_text, self.text_target_length) # [batch_size, text_target_length, out_feats = 512]
        else:
            h_text = self.text_resizer(x_text, device)  # [batch_size, text_target_length, out_feats = 768]

        h_text = safe_list_to(h_text, device)

        h_text = self.text_proj_net(h_text) # [batch_size, text_target_length, out_feats = 256]

        if self.text_embedding is not None:
            arr = []
            for idx in range(len(h_text)):
                arr.append(torch.cat([h_text[idx:idx + 1], self.text_embedding.to(device)], dim=-1)) # [batch_size, text_length, out_feats + prototype encoding = 288]
            h_text = torch.cat(arr, dim=0)  # [batch_size, text_target_length, out_feats + prototype encoding = 288]

        tokens = torch.cat([h_omic, h_path, h_text], dim=1)  # (B, N_p+N_h, d)
        tokens = self.identity(tokens)  # []

        # Required for visualization
        if return_attn:
            with torch.no_grad():
                #_, attn_pathways, cross_attn_pathways, cross_attn_histology = self.coattn[0](x=tokens, mask=None,return_attention=True)
                _, attn_pathways, cross_attn_pathways, cross_attn_histology, cross_attn_histology_text, cross_attn_pathways_text, cross_attn_text_histology, cross_attn_text_pathways = \
                self.coattn[0](x=tokens, mask=None, return_attention=True)

        # Pass the token set through co-attention network
        mm_embed = self.coattn(tokens)

        # ---> aggregate
        # Pathways
        paths_postSA_tokens = mm_embed[:, :self.num_pathways, :]
        if self.use_pathway_attnpool:
            pathway_attn_logits = self.pathway_attn_pool(paths_postSA_tokens).squeeze(-1)
            pathway_attn_weights = torch.softmax(pathway_attn_logits, dim=1)
            paths_postSA_embed = torch.sum(paths_postSA_tokens * pathway_attn_weights.unsqueeze(-1), dim=1)
        else:
            # Mean pooling fallback for pathway tokens to support ablation without attention pooling.
            paths_postSA_embed = torch.mean(paths_postSA_tokens, dim=1)
            pathway_attn_weights = torch.full(
                (paths_postSA_tokens.size(0), paths_postSA_tokens.size(1)),
                1.0 / max(1, paths_postSA_tokens.size(1)),
                dtype=paths_postSA_tokens.dtype,
                device=paths_postSA_tokens.device,
            )

        # Histology
        start_histology = self.num_pathways
        end_histology = self.num_pathways + h_path.shape[1]
        wsi_postSA_embed = mm_embed[:, start_histology:end_histology, :]

        # Text
        text_postSA_embed = mm_embed[:, end_histology:, :]  # Remaining tokens are for text
        text_postSA_embed = torch.mean(text_postSA_embed, dim=1)

        if self.histo_model == 'mil':
            wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)  # For non-prototypes, we just take the mean
        else:
            wsi_postSA_embed = agg_histo(wsi_postSA_embed, self.histo_agg)
        gates = None

        if self.modality == 'histo':  # Just use histo for prediction
            embedding = wsi_postSA_embed
        elif self.modality == 'gene':  # Just use gene for prediction
            embedding = paths_postSA_embed
        elif self.modality == 'text':  # Use only text
            embedding = text_postSA_embed
        elif self.modality == 'histo+text':
            embedding = torch.cat([wsi_postSA_embed, text_postSA_embed], dim=1)
        elif self.modality == 'pathways+text':
            embedding = torch.cat([paths_postSA_embed, text_postSA_embed], dim=1)
        else:  # Use all modalities
            if self.use_gated_fusion:
                gate_gene = self.gene_gate_proj(paths_postSA_embed)
                gate_histo = self.histo_gate_proj(wsi_postSA_embed)
                gate_text = self.text_gate_proj(text_postSA_embed)
                gate_logits = self.gate_mlp(torch.cat([gate_gene, gate_histo, gate_text], dim=1))
                gates = torch.sigmoid(gate_logits)
                paths_for_cls = paths_postSA_embed * gates[:, 0:1]
                wsi_for_cls = wsi_postSA_embed * gates[:, 1:2]
                text_for_cls = text_postSA_embed * gates[:, 2:3]
            else:
                paths_for_cls = paths_postSA_embed
                wsi_for_cls = wsi_postSA_embed
                text_for_cls = text_postSA_embed
            embedding = torch.cat([paths_for_cls, wsi_for_cls, text_for_cls], dim=1)

        logits = self.classifier(embedding)
        out = {'logits': logits,
               'gene_embed': paths_postSA_embed,
               'histo_embed': wsi_postSA_embed,
               'text_embed': text_postSA_embed}
        if gates is not None:
            out['fusion_gates'] = gates
        if return_attn:
            out['omic_attn'] = attn_pathways
            out['cross_attn'] = cross_attn_pathways
            out['path_attn'] = cross_attn_histology
            #extra
            out['cross_attn_histology_text'] = cross_attn_histology_text
            out['cross_attn_pathways_text'] = cross_attn_pathways_text
            out['cross_attn_text_histology'] = cross_attn_text_histology
            out['cross_attn_text_pathways'] = cross_attn_text_pathways
            out['pathway_token_attn'] = pathway_attn_weights

        return out

    def forward(self, x_path, x_omics, x_text, return_attn=False, attn_mask=None, label=None, censorship=None, loss_fn=None):

        out = self.forward_no_loss(x_path, x_omics, x_text, return_attn)
        results_dict, log_dict = process_surv(out['logits'], label, censorship, loss_fn)
        if ("loss" in results_dict) and (results_dict["loss"] is not None) and (self.contrastive_weight > 0):
            contrastive_loss = self._compute_contrastive_loss(
                out["gene_embed"], out["histo_embed"], out["text_embed"]
            )
            total_loss = results_dict["loss"] + self.contrastive_weight * contrastive_loss
            results_dict["loss"] = total_loss
            log_dict["contrastive_loss"] = contrastive_loss.item()
            log_dict["loss"] = total_loss.item()

            if "fusion_gates" in out:
                mean_gates = out["fusion_gates"].detach().mean(dim=0)
                log_dict["gate_gene"] = mean_gates[0].item()
                log_dict["gate_histo"] = mean_gates[1].item()
                log_dict["gate_text"] = mean_gates[2].item()
        if return_attn:
            results_dict['omic_attn'] = out['omic_attn']
            results_dict['cross_attn'] = out['cross_attn']
            results_dict['path_attn'] = out['path_attn']
            #extra
            results_dict['cross_attn_histology_text'] = out['cross_attn_histology_text']
            results_dict['cross_attn_pathways_text'] = out['cross_attn_pathways_text']
            results_dict['cross_attn_text_histology'] = out['cross_attn_text_histology']
            results_dict['cross_attn_text_pathways'] = out['cross_attn_text_pathways']
            results_dict['pathway_token_attn'] = out['pathway_token_attn']


        results_dict.update(out)
        return results_dict, log_dict

