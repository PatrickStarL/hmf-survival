from collections import OrderedDict
from os.path import join
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mil_models.models.model_utils import *

from src.mil_models.components import process_surv
from src.mil_models.models.model_utils import Attn_Net_Gated

"""

Implement Attention MIL for the unimodal (WSI only) and multimodal setting (pathways + WSI). The combining of modalities 
can be done using bilinear fusion or concatenation. 

Mobadersany, Pooya, et al. "Predicting cancer outcomes from histology and genomics using convolutional networks." Proceedings of the National Academy of Sciences 115.13 (2018): E2970-E2979.

"""

################################
# Attention MIL Implementation #
################################
class ABMIL(nn.Module):
    def __init__(self,size_arg = "small", dropout=0.25, n_classes=4, df_comp=None, dim_per_path_1=16, dim_per_path_2=64, device="cpu"):
        r"""
        Attention MIL Implementation

        Args:
            omic_input_dim (int): Dimension size of genomic features.
            fusion (str): Fusion method (Choices: concat, bilinear, or None)
            size_arg (str): Size of NN architecture (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(ABMIL, self).__init__()
        self.device = device
        self.size_dict_path = {"small": [512, 256, 256], "big": [512, 512, 384]}

        ### Deep Sets Architecture Construction
        size = self.size_dict_path[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        self.df_comp = df_comp
        self.dim_per_path_1 = dim_per_path_1
        self.num_pathways = self.df_comp.shape[1]
        self.dim_per_path_2 = dim_per_path_2

        self.classifier = nn.Linear(size[2], n_classes)
        self.classifier = self.classifier.to(self.device)


    def forward_no_loss(self, x_path):

        x_path = x_path #---> need to do this to make it work with this set up
        A, h_path = self.attention_net(x_path)  
        A = torch.transpose(A, 1, 0)
        A_raw = A 
        A = F.softmax(A, dim=1) 
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path).squeeze()
        
        h = h_path # [256] vector
        
        logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector
        print(logits.shape)

        out = {'logits': logits}

        return out

    def forward(self, x_path, x_omics, return_attn=False, attn_mask=None, label=None, censorship=None, loss_fn=None):
        out = self.forward_no_loss(x_path)
        results_dict, log_dict = process_surv(out['logits'], label, censorship, loss_fn)
        results_dict.update(out)

        return results_dict, log_dict
