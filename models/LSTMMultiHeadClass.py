import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMMultiHead(nn.Module):
    def __init__(self, input_dim=6, encoder_hidden=256, decoder_hidden=256, num_heads=8):
        super(LSTMMultiHead, self).__init__()
        
        self.encoder_hidden = encoder_hidden
        self.num_heads = num_heads
        assert encoder_hidden % num_heads == 0, "encoder_hidden must be divisible by num_heads"
        self.head_dim = encoder_hidden // num_heads
        
        # Encoders
        self.ego_lstm = nn.LSTM(input_size=input_dim, hidden_size=encoder_hidden, num_layers=2, dropout=0.1) # ego car
        self.nbr_lstm = nn.LSTM(input_size=input_dim, hidden_size=encoder_hidden, num_layers=2, dropout=0.1) # other agents
        
        # Spatial Attention projections
        self.query = nn.Linear(encoder_hidden, encoder_hidden)
        self.key = nn.Linear(encoder_hidden, encoder_hidden)
        self.value = nn.Linear(encoder_hidden, encoder_hidden)
        self.attn_out = nn.Linear(encoder_hidden, encoder_hidden)
        
        # Temporal Attention Layer using a 1D Convolutional Layer
        self.temporal_conv = nn.Conv1d(2*encoder_hidden, encoder_hidden, kernel_size=3, padding=1)
        
        # Decoder
        self.decoder = nn.LSTM(input_size=encoder_hidden, hidden_size=decoder_hidden, num_layers=2, dropout=0.1)
        self.output_layer = nn.Linear(decoder_hidden, 2)

    def forward(self, data):
        x = data.x  # Expected shape: (batch_size * num_agents, 50, 6)
        
        batch_x_agents = x.shape[0]
        num_agents = 20
        num_neighbors = num_agents - 1
        batch_size = batch_x_agents // num_agents

        # Reshape and encode
        x = x.reshape(batch_size, num_agents, 50, 6)
        ego_hist = x[:, 0].permute(1, 0, 2)                                     # (50, bs, 6)
        nbrs_hist = x[:, 1:].reshape(-1, 50, 6).permute(1, 0, 2)                # (50, bs*num_neighbors, 6)

        # Encode trajectories
        ego_out, _ = self.ego_lstm(ego_hist)                                    # (50, bs, encoder_hidden)
        nbrs_out, _ = self.nbr_lstm(nbrs_hist)                                  # (50, bs*num_neighbors, encoder_hidden)
        nbrs_out = nbrs_out.view(50, batch_size, num_neighbors, -1)             # (50, bs, num_neighbors, encoder_hidden)

        # Claude was used to help with tensor dimensions in multi-head attention
        # Prepare attention inputs
        queries = self.query(ego_out)                                           # (50, bs, encoder_hidden)
        keys = self.key(nbrs_out)                                               # (50, bs, num_neighbors, encoder_hidden)
        values = self.value(nbrs_out)                                           # (50, bs, num_neighbors, encoder_hidden)
        
        # Reshape for multi-head attention
        queries = queries.view(50, batch_size, self.num_heads, self.head_dim).transpose(1, 2)                       # (50, num_heads, bs, head_dim)
        keys = keys.view(50, batch_size, num_neighbors, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)       # (50, num_heads, bs, num_neighbors, head_dim)
        values = values.view(50, batch_size, num_neighbors, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)   # (50, num_heads, bs, num_neighbors, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(queries.unsqueeze(3), keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attn_weights, values)                           # (50, num_heads, bs, 1, head_dim)
        
        # Combine heads
        attended = attended.squeeze(3).transpose(1, 2).reshape(50, batch_size, -1)  # (50, bs, encoder_hidden)
        attended = self.attn_out(attended)

        # Combine with ego features
        combined = torch.cat([ego_out, attended], dim=-1)                       # (50, bs, 2*encoder_hidden)
        
        # Temporal Attention using Conv1D (Which time steps are the most important to the ego?)
        temporal_features = self.temporal_conv(combined.permute(1, 2, 0))       # (bs, encoder_hidden, 50)
        temporal_features = temporal_features.permute(2, 0, 1)                  # (50, bs, encoder_hidden)
        
        # Decode into predictions
        decoder_input = temporal_features[-1:].repeat(60, 1, 1)                 # (60, bs, encoder_hidden)
        hidden = None
        predictions = []
        
        # Generate 60 future timesteps
        for t in range(60):
            output, hidden = self.decoder(decoder_input[t:t+1], hidden)
            pred = self.output_layer(output.squeeze(0))
            predictions.append(pred)
        
        predictions = torch.stack(predictions).permute(1, 0, 2)                 # (bs, 60, 2)
        return predictions