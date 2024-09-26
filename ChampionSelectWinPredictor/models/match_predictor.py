# models/match_predictor.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LoLMatchPredictor(nn.Module):
    def __init__(self, num_champions, embedding_dim, heuristic_dim):
        super(LoLMatchPredictor, self).__init__()
        self.embedding_dim = embedding_dim
        self.champion_embedding = nn.Embedding(num_champions, embedding_dim)
        # Calculate input dimension
        input_dim = (embedding_dim * 5) * 2 + heuristic_dim * 2
        # Define neural network layers
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, blue_team_indices, red_team_indices, blue_heuristics, red_heuristics):
        # Get embeddings for champions
        blue_embeds = self.champion_embedding(blue_team_indices)  # Shape: (batch_size, 5, embedding_dim)
        red_embeds = self.champion_embedding(red_team_indices)    # Shape: (batch_size, 5, embedding_dim)

        # Flatten the embeddings
        blue_embeds = blue_embeds.view(blue_embeds.size(0), -1)  # Shape: (batch_size, 5 * embedding_dim)
        red_embeds = red_embeds.view(red_embeds.size(0), -1)     # Shape: (batch_size, 5 * embedding_dim)

        # Concatenate embeddings and heuristics
        x = torch.cat([blue_embeds, red_embeds, blue_heuristics, red_heuristics], dim=1)

        # Pass through neural network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sigmoid(self.output(x))

        return x