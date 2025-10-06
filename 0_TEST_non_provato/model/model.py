import torch
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, GATConv, GlobalAttention, GraphNorm
from torch_geometric.nn import global_mean_pool as gmp, global_max_pool as gmaxp
import warnings
import torch.nn as nn
warnings.filterwarnings("ignore")


class GAT_with_GAP_norm_leakyRELU_06_GraphNorm(torch.nn.Module):
    def __init__(self, input_dim=768, embedding_size=64, num_heads=4, dropout_rate=0.6, output_size=1):
        super(GAT_with_GAP_norm_leakyRELU_06_GraphNorm, self).__init__()
        
        # GAT layers
        self.initial_conv = GATConv(input_dim, embedding_size, heads=num_heads, concat=True)  
        self.graph_norm = GraphNorm(embedding_size * num_heads)  

        self.conv1 = GATConv(embedding_size * num_heads, embedding_size, heads=num_heads, concat=True)  
        self.graph_norm1 = GraphNorm(embedding_size * num_heads)  
        
        # Global Attention Pooling 
        self.attention_pool = GlobalAttention(nn.Linear(embedding_size * num_heads, 1))  

        # Output layer
        self.out = nn.Linear(embedding_size * num_heads, output_size)  
        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x, edge_index, batch_index):
        x = x.float()
        
        # First GAT layer with GraphNorm
        hidden, attn_weights = self.initial_conv(x, edge_index, return_attention_weights=True)
        hidden = self.graph_norm(hidden)  # Apply GraphNorm
        hidden = self.leaky_relu(hidden)  # Apply Leaky ReLU after GraphNorm
        hidden = self.dropout(hidden)  # Apply Dropout
        
        # Second GAT layer with GraphNorm
        hidden, attn_weights1 = self.conv1(hidden, edge_index, return_attention_weights=True)
        hidden = self.graph_norm1(hidden)  # Apply GraphNorm
        hidden = self.leaky_relu(hidden)  # Apply Leaky ReLU after GraphNorm
        hidden = self.dropout(hidden)  # Apply Dropout

        # Global Attention Pooling
        hidden = self.attention_pool(hidden, batch_index)

        # Final output
        out = self.out(hidden)
        return out, hidden, attn_weights, attn_weights1

