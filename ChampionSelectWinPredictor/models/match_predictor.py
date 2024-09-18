# models/match_predictor.py

import torch
import torch.nn as nn

class LoLMatchPredictor(nn.Module):
    def __init__(self, num_champions, embedding_dim, attribute_dim,
                 num_classes, class_embedding_dim, damage_type_dim):
        super(LoLMatchPredictor, self).__init__()  # Ensure this is the first line

        # Embedding layers
        self.embedding = nn.Embedding(num_champions, embedding_dim)
        self.class_embedding = nn.Embedding(num_classes, class_embedding_dim)

        # Update input dimension
        input_dim = (embedding_dim + attribute_dim + class_embedding_dim + damage_type_dim) * 2

        # Define your fully connected layers
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, team_a_ids, team_b_ids,
                team_a_numerical, team_b_numerical,
                team_a_class_ids, team_b_class_ids,
                team_a_damage_one_hot, team_b_damage_one_hot):
        # Get champion embeddings
        emb_a = self.embedding(team_a_ids)  # [batch_size, 5, embedding_dim]
        emb_b = self.embedding(team_b_ids)  # [batch_size, 5, embedding_dim]

        # Get class embeddings
        class_emb_a = self.class_embedding(team_a_class_ids)  # [batch_size, 5, class_embedding_dim]
        class_emb_b = self.class_embedding(team_b_class_ids)  # [batch_size, 5, class_embedding_dim]

        # Aggregate embeddings and features
        team_a_emb = torch.mean(emb_a, dim=1)
        team_b_emb = torch.mean(emb_b, dim=1)
        team_a_class_emb = torch.mean(class_emb_a, dim=1)
        team_b_class_emb = torch.mean(class_emb_b, dim=1)
        team_a_numerical = torch.mean(team_a_numerical, dim=1)
        team_b_numerical = torch.mean(team_b_numerical, dim=1)
        team_a_damage = torch.mean(team_a_damage_one_hot, dim=1)
        team_b_damage = torch.mean(team_b_damage_one_hot, dim=1)

        # Concatenate all features
        team_a_features = torch.cat([team_a_emb, team_a_numerical, team_a_class_emb, team_a_damage], dim=1)
        team_b_features = torch.cat([team_b_emb, team_b_numerical, team_b_class_emb, team_b_damage], dim=1)

        x = torch.cat([team_a_features, team_b_features], dim=1)

        # Pass through fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))

        return x.squeeze()
