import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from src.mil_models.components import process_surv


class DAttention_Text(nn.Module):
    def __init__(self, n_classes, dropout, act, n_features=512):
        super(DAttention_Text, self).__init__()
        self.L = 256
        self.D = 128
        self.K = 1
        self.feature = [nn.Linear(n_features, 256)]

        if act.lower() == "gelu":
            self.feature += [nn.GELU()]
        else:
            self.feature += [nn.ReLU()]

        if dropout:
            self.feature += [nn.Dropout(0.25)]

        self.feature = nn.Sequential(*self.feature)

        self.attention = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K))
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, n_classes),
        )

    def forward_no_loss(self, x):
        x = x[0]
        feature = self.feature(x)
        #feature = feature.squeeze()
        A = self.attention(feature)
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # KxL
        logits = self.classifier(M)

        out = {'logits': logits}

        return out

    def forward(self, x_path, x_omics, x_text, return_attn=False, attn_mask=None, label=None, censorship=None, loss_fn=None):
        out = self.forward_no_loss(x_text)
        results_dict, log_dict = process_surv(out['logits'], label, censorship, loss_fn)
        results_dict.update(out)

        return results_dict, log_dict