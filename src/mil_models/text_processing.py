import torch
import torch.nn.functional as F
import torch.nn as nn


def interpolate_to_fixed_length(embeddings, target_length):
    """
    Resize embeddings to a fixed length via interpolation.
    Args:
        embeddings: List of tensors of shape [N_i, D] (variable-length embeddings)
        target_length: Desired fixed length (number of chunks)
    Returns:
        Tensor of shape [batch_size, target_length, D]
    """
    batch_size = len(embeddings)  # Number of embeddings
    feature_dim = embeddings[0].shape[1]  # Dimensionality of each embedding (D)
    resized_embeddings = torch.zeros(batch_size, target_length, feature_dim)  # Output tensor

    for i, embedding in enumerate(embeddings):
        original_length = embedding.shape[0]  # Original number of chunks
        if original_length != target_length:
            # Permute to [C, L] for interpolation (where C = feature_dim, L = original_length)
            embedding = embedding.permute(1, 0).unsqueeze(0)  # [1, C, L]
            # Interpolate along the sequence length
            resized = F.interpolate(
                embedding,
                size=target_length,  # Target length (T)
                mode='linear',
                align_corners=False
            ).squeeze(0).permute(1, 0)  # Back to [T, C]
            resized_embeddings[i] = resized
        else:
            # No resizing needed
            resized_embeddings[i] = embedding

    return resized_embeddings

class SelfAttentionResizer(nn.Module):
    def __init__(self, input_dim, target_length, aggregation_method, max_length, num_heads=1):
        super(SelfAttentionResizer, self).__init__()
        self.target_length = target_length

        # Self-attention components
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.num_heads = num_heads
        self.aggregation_method = aggregation_method

        # Final output projection to maintain input dimension size
        self.output_projection = nn.Linear(input_dim, input_dim)
        self.max_length = max_length

    def forward(self, embeddings, device):
        """
        Args:
            embeddings: List of tensors with shape [N_i, input_dim] (variable-length embeddings).
        Returns:
            Resized embeddings of shape [batch_size, target_length, input_dim].
        """
        # Step 1: Pad inputs to the maximum length in the batch
        batch_size = len(embeddings)
        batch_max_length = max([embedding.shape[0] for embedding in embeddings])
        max_length = self.max_length
        max_length = max(batch_max_length, max_length)
        input_dim = embeddings[0].shape[1]

        padded_embeddings = torch.zeros(batch_size, max_length, input_dim, device=embeddings[0].device)
        attention_mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=embeddings[0].device)

        for i, embedding in enumerate(embeddings):
            length = embedding.shape[0]
            padded_embeddings[i, :length, :] = embedding
            attention_mask[i, :length] = True

        # Step 2: Compute self-attention scores and outputs
        Q = self.query(padded_embeddings)
        K = self.key(padded_embeddings)
        V = self.value(padded_embeddings)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(input_dim, dtype=torch.float32))
        scores = scores.masked_fill(~attention_mask.unsqueeze(1), float('-inf'))  # Mask padding
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # Compute overall attention scores (average across heads)
        overall_attention_scores = attention_weights.mean(dim=1)


        if self.aggregation_method == 'sampling':
        # Attention-Based Sampling
            top_indices = torch.argsort(overall_attention_scores, dim=-1, descending=True)[:, :self.target_length]
            sampled_embeddings = torch.zeros(batch_size, self.target_length, input_dim, device=embeddings[0].device)
            for i in range(batch_size):
                sampled_embeddings[i] = attention_output[i, top_indices[i]]

            # Project the final output
            resized_embeddings = self.output_projection(sampled_embeddings)

        return resized_embeddings


