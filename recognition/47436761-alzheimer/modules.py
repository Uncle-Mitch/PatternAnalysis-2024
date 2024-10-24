import torch.nn as nn
import torch


class PatchEmbedding(nn.Module):
    """Class to perform patch embedding on the input images."""
    def __init__(self, in_channels=1, patch_size=16, embed_size=768):
        """
        Initializes the PatchEmbedding module.
        Args:
            in_channels (int, optional): Number of input channels. Default is 1.
            patch_size (int, optional): Size of each patch. Default is 16.
            embed_size (int, optional): Size of the embedding. Default is 768.
        """
        
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_size, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(start_dim=2)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))

    def forward(self, x):
        """
        Forward pass for the module.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width].
        Returns:
            torch.Tensor: Output tensor after applying projection, rearranging, and concatenating class tokens.
        """
        
        # Apply projection to get patches and flatten them
        x = self.projection(x)
        x = x.transpose(1, 2)  # Rearrange to [batch_size, num_patches, embed_size]
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        # Concatenate class tokens with the patch embeddings
        x = torch.cat((cls_tokens, x), dim=1)
        return x


class TransformerEncoder(nn.Module):
    """Class to implement a Transformer Encoder block from the Vision Transformer Architecture."""
    def __init__(self, d_model=768, num_heads=8, d_mlp=2048, dropout_rate=0.1):
        """
        Initializes the TransformerEncoder module.

        Args:
            d_model (int, optional): The dimension of the input embeddings. Default is 768.
            num_heads (int, optional): The number of attention heads. Default is 8.
            d_mlp (int, optional): The dimension of the feed-forward network. Default is 2048.
            dropout_rate (float, optional): The dropout rate to be applied in the feed-forward network. Default is 0.1.
        """
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_mlp, d_model),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        """
        Perform a forward pass through the network.

        This method applies a self-attention block followed by an MLP block, each with a residual connection.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the self-attention and MLP blocks.
        """
        # Self-Attention block with residual connection
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attention(x_norm, x_norm, x_norm)
        x = x + attn_output

        # MLP block with residual connection
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output
        return x


class AlzheimerModel(nn.Module):
    """
    A PyTorch model for Alzheimer's disease classification using a Vision Transformer architecture.
    """
    def __init__(self, in_channels, patch_size, embed_size, img_size, num_layers, num_heads, d_mlp, dropout_rate, num_classes=2):
        """
        Initializes the AlzheimerModel.

        Args:
            in_channels (int): Number of input channels.
            patch_size (int): Size of each patch.
            embed_size (int): Size of the embedding vector.
            img_size (int): Size of the input image.
            num_layers (int): Number of transformer encoder layers.
            num_heads (int): Number of attention heads in the transformer encoder.
            d_mlp (int): Dimension of the MLP in the transformer encoder.
            dropout_rate (float): Dropout rate for the transformer encoder.
            num_classes (int, optional): Number of output classes. Default is 2.
        """
        super(AlzheimerModel, self).__init__()

        num_patches = (img_size // patch_size) ** 2
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_size)
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_size))

        self.encoders = nn.ModuleList([
            TransformerEncoder(d_model=embed_size, num_heads=num_heads, d_mlp=d_mlp, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ])

        # MLP Head for classification
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, num_classes)
        )

    def forward(self, x):
        """
        Perform a forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor to the model.
        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        x = self.patch_embedding(x)
        x = x + self.positional_embedding

        # Pass through each Transformer Encoder layer
        for encoder in self.encoders:
            x = encoder(x)
        
        # Use the CLS token's output for classification
        output = self.mlp_head(x[:, 0])
        return output