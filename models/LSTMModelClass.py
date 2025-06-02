import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMAttention(nn.Module):
    def __init__(self, input_dim=6, encoder_hidden=128, decoder_hidden=128, output_dim=60 * 2):
        super(LSTMAttention, self).__init__()

        # Ego Encoder
        self.ego_lstm = nn.LSTM(input_size=6, hidden_size=encoder_hidden)
        # Other Agents Encoder
        self.nbr_lstm = nn.LSTM(input_size=6, hidden_size=encoder_hidden)

        # Spacial Attention Layer
        self.spatial_attn = nn.Sequential(
            nn.Linear(128,64), nn.ReLU(), nn.Linear(64,1)
        )
        # Temporal Attention Layer using a 1D Convolutional Layer
        self.temporal_attn = nn.Conv1d(256, 128, kernel_size=5, padding=1)

        #Decoder
        self.decoder = nn.LSTM(input_size=128, hidden_size=decoder_hidden)
        self.output_layer = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,2))


    def forward(self, data):
        # Claude was used to help with tensor dimensions in attention calculation layers
        
        x = data.x  # Expected shape: (batch_size * num_agents, 50, 6)
        
        batch_x_agents = x.shape[0]
        num_agents = 20
        num_neighbors = num_agents - 1
        batch_size = batch_x_agents // num_agents

        x = x.reshape(batch_size, num_agents, 50, 6)                     # (batch_size, num_agents, 50, 6)
        ego_hist = x[:, 0, :, :]                                         # (batch_size, 50, 6)
        nbrs_hist = x[:, 1:, :, :]                                       # (batch_size, num_neighbors, 50, 6)
        
        # Permute for LSTM input (seq_len, batch_size, features)
        ego_hist = ego_hist.permute(1, 0, 2)                             # (50, batch_size, 6)
        nbrs_hist = nbrs_hist.reshape(-1, 50, 6)
        nbrs_hist = nbrs_hist.permute(1, 0, 2)                           # (50, batch_size * num_neighbors, 6)

        # Encode Histories
        ego_out, _ = self.ego_lstm(ego_hist)                             # (50, batch_size, encoder_hidden=64)
        nbrs_out, _ = self.nbr_lstm(nbrs_hist)                           # (50, batch_size*num_neighbors, encoder_hidden=64)
        nbrs_out = nbrs_out.view(50, batch_size, num_neighbors, -1)      # (50, batch_size, num_neighbors, encoder_hidden=64)

        # Spatial Attention (Which agents are the most important to the ego?)
        nbrs_reshaped = nbrs_out.view(-1, 128)                            # (50*batch_size*num_neighbors, 64)
        attn_scores = self.spatial_attn(nbrs_reshaped)                   # (50*batch_size*num_neighbors, 1)
        attn_scores = attn_scores.view(50, batch_size, num_neighbors, 1) # (50, batch_size, num_neighbors, 1)
        attn_weights = F.softmax(attn_scores, dim=2)                     # (50, batch_size, num_neighbors, 1)
        attended_nbrs = nbrs_out * attn_weights                          # (50, batch_size, num_neighbors, 64)
        attended_nbrs = attended_nbrs.sum(dim=2)                         # (50, batch_size, 64)
        combined = torch.cat([ego_out, attended_nbrs], dim=-1)           # (50, batch_size, 128)

        # Temporal Attention using Conv1D (Which time steps are the most important to the ego?)
        combined_for_conv = combined.permute(1, 2, 0)                    # (batch_size, 128, 50)
        temporal_attn_out = self.temporal_attn(combined_for_conv)        # (batch_size, 64, 50)

        # Decode into predictions
        temporal_attn_out = temporal_attn_out.permute(2, 0, 1)           # (50, batch_size, 128)
        decoder_input = temporal_attn_out[-1:, :, :]                     # (1, batch_size, 128) -> Get last timestep from temporal attention
        hidden = None
        predictions = []
        current_input = decoder_input

        # Generate 60 future timesteps
        for t in range(60):
            output, hidden = self.decoder(current_input, hidden)
            pred = self.output_layer(output)
            predictions.append(pred)

            current_input = output

        predictions = torch.cat(predictions, dim=0)                      # (60, batch_size, 2)
        predictions = predictions.permute(1, 0, 2)                       # (batch_size, 60, 2)

        return predictions
