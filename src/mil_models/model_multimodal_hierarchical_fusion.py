import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.utils import safe_list_to

from .components import (
    FeedForward,
    FeedForwardEnsemble,
    FeedForwardEnsemble_combined_text,
    CoAttentionLayer,
    process_surv,
)
from .text_processing import SelfAttentionResizer, interpolate_to_fixed_length
from .model_multimodal_encoders import (
    LightweightQFormer,
    construct_proto_embedding_grouped_text,
    construct_proto_embedding_text,
    init_per_path_model,
)
from .model_multimodal_cls_fusion import (
    ShallowCLSTransformerEncoder,
    agg_histo,
)


class AttentionPool1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        logits = self.score(x).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class coattn_text_twostage(nn.Module):
    def __init__(
        self,
        omic_sizes=[100, 200, 300, 400, 500, 600],
        histo_in_dim=1024,
        text_in_dim=None,
        dropout=0.1,
        num_classes=4,
        path_proj_dim=256,
        num_coattn_layers=1,
        modality="twostage_cls",
        histo_model="PANTHER",
        append_embed="none",
        group_prototype=False,
        mult=1,
        net_indiv=False,
        net_text_combined=False,
        numOfproto=16,
        pathway_token_count=16,
        text_target_length=43,
        text_max_length=60,
        text_resizing_model=None,
        attn_mode=None,
        residual=False,
        residual_type="all",
        pathway_dropout=0.0,
        use_gated_fusion=False,
        contrastive_weight=0.05,
        contrastive_temp=0.07,
        contrastive_pairs="gene_histo,gene_text,text_histo",
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
        cls_depth=2,
        cls_num_heads=4,
        cls_mlp_ratio=2.0,
        cls_dropout=0.1,
    ):
        super().__init__()

        self.raw_num_pathways = len(omic_sizes)
        requested_pathway_tokens = int(pathway_token_count) if int(pathway_token_count) > 0 else self.raw_num_pathways
        self.num_pathways = requested_pathway_tokens
        self.num_coattn_layers = int(num_coattn_layers)

        self.histo_in_dim = histo_in_dim
        self.out_mult = mult
        self.net_indiv = net_indiv
        self.net_text_combined = net_text_combined
        self.group_prototype = group_prototype
        self.modality = modality
        self.active_modalities = self._resolve_active_modalities(modality)
        self.histo_agg = "mean"
        self.pathway_dropout = max(0.0, min(float(pathway_dropout), 0.95))
        self.use_gated_fusion = bool(use_gated_fusion)
        self.contrastive_weight = max(0.0, float(contrastive_weight))
        self.contrastive_temp = max(1e-4, float(contrastive_temp))
        self.force_gene_histo_contrastive = bool(force_gene_histo_contrastive)
        if self.force_gene_histo_contrastive:
            self.contrastive_pairs = ["gene_histo"]
        else:
            self.contrastive_pairs = [p.strip().lower() for p in str(contrastive_pairs).split(",") if p.strip()]
        self.use_pathway_attnpool = bool(use_pathway_attnpool)

        self.numOfproto = int(numOfproto)
        self.num_classes = int(num_classes)
        self.freeze_non_qformer = bool(freeze_non_qformer)
        self.freeze_conch = bool(freeze_conch)

        self.histo_model = histo_model.lower()
        self.use_path_qformer = bool(use_path_qformer)
        self.qformer_num_queries = int(qformer_num_queries) if int(qformer_num_queries) > 0 else int(self.numOfproto)
        self.num_histo_tokens = int(self.qformer_num_queries if self.use_path_qformer else self.numOfproto)
        if self.use_path_qformer and self.histo_model != "mil":
            raise ValueError("`use_path_qformer=True` currently supports `model_histo_type=MIL` only.")

        self.sig_networks = init_per_path_model(omic_sizes)
        self.identity = nn.Identity()
        self.append_embed = append_embed

        self.text_target_length = int(text_target_length)
        self.max_text_length = int(text_max_length)
        self.text_resizing_model = text_resizing_model
        self.text_in_dim = int(text_in_dim) if text_in_dim is not None else self.histo_in_dim
        self.text_resize_dim = self.text_in_dim

        if self.text_resizing_model == "SA_sampling":
            self.text_resizer = SelfAttentionResizer(
                input_dim=self.text_in_dim,
                max_length=self.max_text_length,
                target_length=self.text_target_length,
                aggregation_method="sampling",
            )
        else:
            self.text_resizer = None

        if self.histo_model == "panther":
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
                self.path_proj_dim, self.histo_embedding, self.gene_embedding, self.text_embedding = (
                    construct_proto_embedding_grouped_text(
                        path_proj_dim,
                        self.append_embed,
                        self.numOfproto,
                        self.num_pathways,
                        1,
                        self.text_target_length,
                    )
                )
            else:
                self.path_proj_dim, self.histo_embedding, self.gene_embedding, self.text_embedding = (
                    construct_proto_embedding_text(
                        path_proj_dim,
                        self.append_embed,
                        self.numOfproto,
                        self.num_pathways,
                        self.text_target_length,
                    )
                )
        else:
            self.path_proj_dim = path_proj_dim
            self.histo_embedding = None
            self.gene_embedding = None
            self.text_embedding = None

        coattn_list = []
        if self.num_coattn_layers == 0:
            out_dim = self.path_proj_dim
            if self.net_indiv:
                feed_forward = FeedForwardEnsemble(
                    out_dim,
                    self.out_mult,
                    dropout=dropout,
                    num=self.num_histo_tokens + self.num_pathways + self.text_target_length,
                )
            else:
                feed_forward = FeedForward(out_dim, self.out_mult, dropout=dropout)
            layer_norm = nn.LayerNorm(int(out_dim * self.out_mult))
            coattn_list.extend([feed_forward, layer_norm])
        else:
            out_dim = self.path_proj_dim // 2
            out_mult = self.out_mult
            cross_attender = CoAttentionLayer(
                dim=self.path_proj_dim,
                dim_head=out_dim,
                heads=1,
                residual=residual,
                dropout=0.1,
                num_pathways=self.num_pathways,
                num_prototypes=self.num_histo_tokens,
                attn_mode=attn_mode,
                residual_type=residual_type,
            )
            if self.net_indiv:
                if self.net_text_combined:
                    feed_forward = FeedForwardEnsemble_combined_text(
                        out_dim,
                        out_mult,
                        dropout=dropout,
                        num=self.num_histo_tokens + self.num_pathways,
                    )
                else:
                    feed_forward = FeedForwardEnsemble(
                        out_dim,
                        out_mult,
                        dropout=dropout,
                        num=self.num_histo_tokens + self.num_pathways + self.text_target_length,
                    )
            else:
                feed_forward = FeedForward(out_dim, out_mult, dropout=dropout)
            layer_norm = nn.LayerNorm(int(out_dim * out_mult))
            coattn_list.extend([cross_attender, feed_forward, layer_norm])

        self.coattn = nn.Sequential(*coattn_list)

        out_dim_final = int(out_dim * self.out_mult)
        self.out_dim_final = out_dim_final
        self.summary_dim = out_dim_final
        self.full_token_dim = self.path_proj_dim

        self.gene_summary_pool = AttentionPool1D(self.summary_dim)
        self.histo_summary_pool = AttentionPool1D(self.summary_dim)
        self.text_summary_pool = AttentionPool1D(self.summary_dim)
        self.full_gene_summary_pool = AttentionPool1D(self.full_token_dim)
        self.full_histo_summary_pool = AttentionPool1D(self.full_token_dim)
        self.full_text_summary_pool = AttentionPool1D(self.full_token_dim)

        self.summary_cls_transformer = ShallowCLSTransformerEncoder(
            dim=self.summary_dim,
            num_pathways=1,
            num_histo_tokens=1,
            num_text_tokens=1,
            depth=cls_depth,
            num_heads=cls_num_heads,
            mlp_ratio=cls_mlp_ratio,
            dropout=cls_dropout,
        )
        self.full_token_cls_transformer = ShallowCLSTransformerEncoder(
            dim=self.full_token_dim,
            num_pathways=self.num_pathways,
            num_histo_tokens=self.num_histo_tokens,
            num_text_tokens=self.text_target_length,
            depth=cls_depth,
            num_heads=cls_num_heads,
            mlp_ratio=cls_mlp_ratio,
            dropout=cls_dropout,
        )

        summary_variants = {
            "twostage_cls",
            "full_twostage",
            "histo_only_twostage",
            "histo_text_twostage",
            "histo_rna_twostage",
        }
        if self.modality in summary_variants:
            classifier_in_dim = self.summary_dim
            gate_dim = self.summary_dim
        elif self.modality == "coattn_text_twostage":
            classifier_in_dim = self.summary_dim * 3
            gate_dim = self.summary_dim
        elif self.modality == "cls_transformer_twostage":
            classifier_in_dim = self.full_token_dim
            gate_dim = self.full_token_dim
        else:
            raise NotImplementedError(f"Unsupported strict twostage modality: {self.modality}")

        self.classifier = nn.Linear(classifier_in_dim, self.num_classes, bias=False)
        self.gene_gate_proj = nn.Linear(gate_dim, gate_dim)
        self.histo_gate_proj = nn.Linear(gate_dim, gate_dim)
        self.text_gate_proj = nn.Linear(gate_dim, gate_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_dim * 3, gate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gate_dim, 3),
        )

        self.contrastive_gene_proj = nn.Linear(gate_dim, gate_dim)
        self.contrastive_histo_proj = nn.Linear(gate_dim, gate_dim)
        self.contrastive_text_proj = nn.Linear(gate_dim, gate_dim)
        self._configure_trainable_params()

    def _configure_trainable_params(self):
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
            "gene": "gene",
            "path": "gene",
            "pathway": "gene",
            "pathways": "gene",
            "histo": "histo",
            "histology": "histo",
            "wsi": "histo",
            "text": "text",
        }
        return aliases.get(name, name)

    @staticmethod
    def _resolve_active_modalities(modality):
        modality_map = {
            "twostage_cls": {"gene", "histo", "text"},
            "full_twostage": {"gene", "histo", "text"},
            "coattn_text_twostage": {"gene", "histo", "text"},
            "cls_transformer_twostage": {"gene", "histo", "text"},
            "histo_only_twostage": {"histo"},
            "histo_text_twostage": {"histo", "text"},
            "histo_rna_twostage": {"gene", "histo"},
        }
        if modality not in modality_map:
            raise NotImplementedError(f"Unsupported strict twostage modality: {modality}")
        return modality_map[modality]

    @staticmethod
    def _empty_token_slice(embed):
        return embed.new_zeros(embed.size(0), 0, embed.size(1))

    def _mask_inactive_modalities(self, h_omic, h_path, h_text):
        if "gene" not in self.active_modalities:
            h_omic = torch.zeros_like(h_omic)
        if "histo" not in self.active_modalities:
            h_path = torch.zeros_like(h_path)
        if "text" not in self.active_modalities:
            h_text = torch.zeros_like(h_text)
        return h_omic, h_path, h_text

    def _select_summary_tokens(self, gene_embed, histo_embed, text_embed):
        gene_tokens = gene_embed.unsqueeze(1) if "gene" in self.active_modalities else self._empty_token_slice(gene_embed)
        histo_tokens = histo_embed.unsqueeze(1) if "histo" in self.active_modalities else self._empty_token_slice(histo_embed)
        text_tokens = text_embed.unsqueeze(1) if "text" in self.active_modalities else self._empty_token_slice(text_embed)
        return gene_tokens, histo_tokens, text_tokens

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
            "gene": self.contrastive_gene_proj(gene_embed),
            "histo": self.contrastive_histo_proj(histo_embed),
            "text": self.contrastive_text_proj(text_embed),
        }

        losses = []
        for pair in self.contrastive_pairs:
            clean_pair = pair.replace("-", "_")
            parts = [self._normalize_pair_name(p.strip()) for p in clean_pair.split("_") if p.strip()]
            if len(parts) != 2:
                continue
            left, right = parts
            if left not in self.active_modalities or right not in self.active_modalities:
                continue
            if left not in embed_map or right not in embed_map or left == right:
                continue
            losses.append(self._pair_infonce(embed_map[left], embed_map[right]))

        if not losses:
            return gene_embed.new_zeros(())

        return torch.stack(losses).mean()

    def _resize_mil_histo_tokens(self, x_path):
        if self.histo_model != "mil" or self.path_qformer is not None:
            return x_path

        current_tokens = x_path.size(1)
        target_tokens = self.num_histo_tokens
        if current_tokens == target_tokens:
            return x_path

        x_path_t = x_path.transpose(1, 2)
        if current_tokens > target_tokens:
            x_path_t = F.adaptive_avg_pool1d(x_path_t, target_tokens)
        else:
            x_path_t = F.interpolate(x_path_t, size=target_tokens, mode="linear", align_corners=False)
        return x_path_t.transpose(1, 2)

    def _resize_pathway_tokens(self, h_omic):
        current_tokens = h_omic.size(1)
        target_tokens = self.num_pathways
        if current_tokens == target_tokens:
            return h_omic

        h_omic_t = h_omic.transpose(1, 2)
        if current_tokens > target_tokens:
            h_omic_t = F.adaptive_avg_pool1d(h_omic_t, target_tokens)
        else:
            h_omic_t = F.interpolate(h_omic_t, size=target_tokens, mode="linear", align_corners=False)
        return h_omic_t.transpose(1, 2)

    def _pool_tokens(self, x, pooler):
        if x.size(1) == 1:
            weights = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
            return x[:, 0, :], weights
        return pooler(x)

    def _split_modal_tokens(self, mm_embed, histo_token_count):
        start_histology = self.num_pathways
        end_histology = self.num_pathways + histo_token_count
        gene_tokens = mm_embed[:, :self.num_pathways, :]
        histo_tokens = mm_embed[:, start_histology:end_histology, :]
        text_tokens = mm_embed[:, end_histology:, :]
        return gene_tokens, histo_tokens, text_tokens

    def _collect_token_attn(self, tokens):
        if self.num_coattn_layers == 0:
            return {}

        with torch.no_grad():
            (
                _,
                attn_pathways,
                cross_attn_pathways,
                cross_attn_histology,
                cross_attn_histology_text,
                cross_attn_pathways_text,
                cross_attn_text_histology,
                cross_attn_text_pathways,
            ) = self.coattn[0](x=tokens, mask=None, return_attention=True)
        return {
            "omic_attn": attn_pathways,
            "cross_attn": cross_attn_pathways,
            "path_attn": cross_attn_histology,
            "cross_attn_histology_text": cross_attn_histology_text,
            "cross_attn_pathways_text": cross_attn_pathways_text,
            "cross_attn_text_histology": cross_attn_text_histology,
            "cross_attn_text_pathways": cross_attn_text_pathways,
        }

    def _summarize_postcoattn_tokens(self, gene_tokens, histo_tokens, text_tokens):
        gene_embed, gene_pool_attn = self._pool_tokens(gene_tokens, self.gene_summary_pool)
        histo_embed, histo_pool_attn = self._pool_tokens(histo_tokens, self.histo_summary_pool)
        text_embed, text_pool_attn = self._pool_tokens(text_tokens, self.text_summary_pool)
        return {
            "gene_embed": gene_embed,
            "histo_embed": histo_embed,
            "text_embed": text_embed,
            "gene_pool_attn": gene_pool_attn,
            "histo_pool_attn": histo_pool_attn,
            "text_pool_attn": text_pool_attn,
        }

    def _summarize_full_token_outputs(self, mm_embed, histo_token_count):
        gene_tokens, histo_tokens, text_tokens = self._split_modal_tokens(mm_embed, histo_token_count)
        gene_embed, gene_pool_attn = self._pool_tokens(gene_tokens, self.full_gene_summary_pool)
        histo_embed, histo_pool_attn = self._pool_tokens(histo_tokens, self.full_histo_summary_pool)
        text_embed, text_pool_attn = self._pool_tokens(text_tokens, self.full_text_summary_pool)
        return {
            "gene_tokens": gene_tokens,
            "histo_tokens": histo_tokens,
            "text_tokens": text_tokens,
            "gene_embed": gene_embed,
            "histo_embed": histo_embed,
            "text_embed": text_embed,
            "gene_pool_attn": gene_pool_attn,
            "histo_pool_attn": histo_pool_attn,
            "text_pool_attn": text_pool_attn,
        }

    def _encode_pathways(self, x_omics, device):
        h_omic = []
        for idx, sig_feat in enumerate(x_omics):
            omic_feat = self.sig_networks[idx](sig_feat.float())
            h_omic.append(omic_feat)
        h_omic = torch.stack(h_omic, dim=1)
        h_omic = self._resize_pathway_tokens(h_omic)

        if self.gene_embedding is not None:
            arr = []
            for idx in range(len(h_omic)):
                arr.append(torch.cat([h_omic[idx:idx + 1], self.gene_embedding.to(device)], dim=-1))
            h_omic = torch.cat(arr, dim=0)

        if self.training and self.pathway_dropout > 0:
            keep_prob = 1.0 - self.pathway_dropout
            pathway_keep = (torch.rand(h_omic.size(0), h_omic.size(1), device=device) < keep_prob).float()
            all_dropped = pathway_keep.sum(dim=1) == 0
            if all_dropped.any():
                random_idx = torch.randint(
                    low=0,
                    high=h_omic.size(1),
                    size=(int(all_dropped.sum().item()),),
                    device=device,
                )
                pathway_keep[all_dropped, random_idx] = 1.0
            h_omic = h_omic * pathway_keep.unsqueeze(-1) / keep_prob

        return h_omic

    def _encode_histology(self, x_path, device):
        x_path = self._resize_mil_histo_tokens(x_path)
        if self.path_qformer is not None:
            x_path = self.path_qformer(x_path)

        h_path = self.path_proj_net(x_path)
        if self.histo_embedding is not None:
            arr = []
            for idx in range(len(h_path)):
                arr.append(torch.cat([h_path[idx:idx + 1], self.histo_embedding.to(device)], dim=-1))
            h_path = torch.cat(arr, dim=0)
        return h_path

    def _encode_text(self, x_text, device):
        if self.text_resizing_model == "Interpolation":
            h_text = interpolate_to_fixed_length(x_text, self.text_target_length)
        else:
            h_text = self.text_resizer(x_text, device)

        h_text = safe_list_to(h_text, device)
        h_text = self.text_proj_net(h_text)

        if self.text_embedding is not None:
            arr = []
            for idx in range(len(h_text)):
                arr.append(torch.cat([h_text[idx:idx + 1], self.text_embedding.to(device)], dim=-1))
            h_text = torch.cat(arr, dim=0)
        return h_text

    def _forward_coattn_variant(self, h_omic, h_path, h_text, return_attn=False):
        tokens = torch.cat([h_omic, h_path, h_text], dim=1)
        token_attn_payload = self._collect_token_attn(tokens) if return_attn else {}

        mm_embed = self.coattn(tokens)
        gene_tokens, histo_tokens, text_tokens = self._split_modal_tokens(mm_embed, h_path.shape[1])
        pooled = self._summarize_postcoattn_tokens(gene_tokens, histo_tokens, text_tokens)

        gene_embed = pooled["gene_embed"]
        histo_embed = pooled["histo_embed"]
        text_embed = pooled["text_embed"]

        out = {
            "gene_embed": gene_embed,
            "histo_embed": histo_embed,
            "text_embed": text_embed,
            "gene_pool_attn": pooled["gene_pool_attn"],
            "histo_pool_attn": pooled["histo_pool_attn"],
            "text_pool_attn": pooled["text_pool_attn"],
            "pathway_token_attn": pooled["gene_pool_attn"],
        }

        gates = None
        if self.modality == "coattn_text_twostage":
            if self.use_gated_fusion:
                gate_gene = self.gene_gate_proj(gene_embed)
                gate_histo = self.histo_gate_proj(histo_embed)
                gate_text = self.text_gate_proj(text_embed)
                gate_logits = self.gate_mlp(torch.cat([gate_gene, gate_histo, gate_text], dim=1))
                gates = torch.sigmoid(gate_logits)
                embedding = torch.cat([
                    gene_embed * gates[:, 0:1],
                    histo_embed * gates[:, 1:2],
                    text_embed * gates[:, 2:3],
                ], dim=1)
            else:
                embedding = torch.cat([gene_embed, histo_embed, text_embed], dim=1)
            logits = self.classifier(embedding)
            out["logits"] = logits
        else:
            gene_embed_for_fusion = gene_embed
            histo_embed_for_fusion = histo_embed
            text_embed_for_fusion = text_embed

            if self.use_gated_fusion:
                gate_gene = self.gene_gate_proj(gene_embed)
                gate_histo = self.histo_gate_proj(histo_embed)
                gate_text = self.text_gate_proj(text_embed)
                gate_logits = self.gate_mlp(torch.cat([gate_gene, gate_histo, gate_text], dim=1))
                gates = torch.sigmoid(gate_logits)
                gene_embed_for_fusion = gene_embed * gates[:, 0:1]
                histo_embed_for_fusion = histo_embed * gates[:, 1:2]
                text_embed_for_fusion = text_embed * gates[:, 2:3]

            summary_gene_tokens, summary_histo_tokens, summary_text_tokens = self._select_summary_tokens(
                gene_embed_for_fusion,
                histo_embed_for_fusion,
                text_embed_for_fusion,
            )
            summary_tokens, summary_attn_payload = self.summary_cls_transformer(
                gene_tokens=summary_gene_tokens,
                histo_tokens=summary_histo_tokens,
                text_tokens=summary_text_tokens,
                return_attn=return_attn,
            )
            cls_embed = summary_tokens[:, 0, :]
            out["cls_embed"] = cls_embed
            out["logits"] = self.classifier(cls_embed)

            if return_attn:
                if "cls_attn" in summary_attn_payload:
                    out["cls_attn"] = summary_attn_payload["cls_attn"]
                if "encoder_attn" in summary_attn_payload:
                    out["encoder_attn"] = summary_attn_payload["encoder_attn"]

        if gates is not None:
            out["fusion_gates"] = gates

        if return_attn:
            out.update(token_attn_payload)

        return out

    def _forward_cls_variant(self, h_omic, h_path, h_text, return_attn=False):
        mm_tokens, attn_payload = self.full_token_cls_transformer(
            gene_tokens=h_omic,
            histo_tokens=h_path,
            text_tokens=h_text,
            return_attn=return_attn,
        )
        cls_embed = mm_tokens[:, 0, :]
        mm_embed = mm_tokens[:, 1:, :]
        pooled = self._summarize_full_token_outputs(mm_embed, h_path.shape[1])

        gene_embed = pooled["gene_embed"]
        histo_embed = pooled["histo_embed"]
        text_embed = pooled["text_embed"]

        gates = None
        if self.use_gated_fusion:
            gate_gene = self.gene_gate_proj(gene_embed)
            gate_histo = self.histo_gate_proj(histo_embed)
            gate_text = self.text_gate_proj(text_embed)
            gate_logits = self.gate_mlp(torch.cat([gate_gene, gate_histo, gate_text], dim=1))
            gates = torch.sigmoid(gate_logits)
            embedding = cls_embed + (
                gate_gene * gates[:, 0:1]
                + gate_histo * gates[:, 1:2]
                + gate_text * gates[:, 2:3]
            )
        else:
            embedding = cls_embed

        out = {
            "logits": self.classifier(embedding),
            "cls_embed": cls_embed,
            "gene_embed": gene_embed,
            "histo_embed": histo_embed,
            "text_embed": text_embed,
            "gene_pool_attn": pooled["gene_pool_attn"],
            "histo_pool_attn": pooled["histo_pool_attn"],
            "text_pool_attn": pooled["text_pool_attn"],
            "pathway_token_attn": pooled["gene_pool_attn"],
        }
        if gates is not None:
            out["fusion_gates"] = gates

        if return_attn:
            if "cls_attn" in attn_payload:
                out["cls_attn"] = attn_payload["cls_attn"]
            if "encoder_attn" in attn_payload:
                out["encoder_attn"] = attn_payload["encoder_attn"]
            out["omic_attn"] = pooled["gene_pool_attn"]
            out["cross_attn"] = attn_payload.get("encoder_attn", pooled["gene_pool_attn"])
            out["path_attn"] = attn_payload.get("cls_attn", pooled["gene_pool_attn"])

        return out

    def forward_no_loss(self, x_path, x_omics, x_text, return_attn=False):
        device = x_path.device

        h_omic = self.identity(self._encode_pathways(x_omics, device))
        h_path = self.identity(self._encode_histology(x_path, device))
        h_text = self.identity(self._encode_text(x_text, device))
        h_omic, h_path, h_text = self._mask_inactive_modalities(h_omic, h_path, h_text)
        if self.modality == "cls_transformer_twostage":
            return self._forward_cls_variant(h_omic, h_path, h_text, return_attn=return_attn)
        if self.modality in {
            "twostage_cls",
            "full_twostage",
            "coattn_text_twostage",
            "histo_only_twostage",
            "histo_text_twostage",
            "histo_rna_twostage",
        }:
            return self._forward_coattn_variant(h_omic, h_path, h_text, return_attn=return_attn)
        raise NotImplementedError(f"Unsupported strict twostage modality: {self.modality}")

    def forward(
        self,
        x_path,
        x_omics,
        x_text,
        return_attn=False,
        attn_mask=None,
        label=None,
        censorship=None,
        loss_fn=None,
    ):
        del attn_mask

        out = self.forward_no_loss(x_path, x_omics, x_text, return_attn)
        results_dict, log_dict = process_surv(out["logits"], label, censorship, loss_fn)

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
            for key in [
                "cls_attn",
                "encoder_attn",
                "omic_attn",
                "cross_attn",
                "path_attn",
                "cross_attn_histology_text",
                "cross_attn_pathways_text",
                "cross_attn_text_histology",
                "cross_attn_text_pathways",
                "pathway_token_attn",
                "gene_pool_attn",
                "histo_pool_attn",
                "text_pool_attn",
            ]:
                if key in out:
                    results_dict[key] = out[key]

        results_dict.update(out)
        return results_dict, log_dict
