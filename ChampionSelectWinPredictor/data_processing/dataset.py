# dataset.py

import torch
from torch.utils.data import Dataset


class LoLMatchDataset(Dataset):
    def __init__(self, matches):
        self.matches = matches

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        (blue_team_ids, red_team_ids, blue_team_attrs, red_team_attrs, label) = self.matches[idx]

        def split_attrs(team_attrs):
            numerical_features = []
            class_ids = []
            damage_type_one_hots = []
            for attrs in team_attrs:
                # Extract numerical features (first 7 values)
                numerical_features.append(attrs[:7])
                # Extract class ID (8th value)
                class_ids.append(int(attrs[7]))
                # Extract damage type one-hot (last 3 values)
                damage_type_one_hots.append(attrs[8:11])
            return numerical_features, class_ids, damage_type_one_hots

        blue_numerical, blue_class_ids, blue_damage_one_hot = split_attrs(blue_team_attrs)
        red_numerical, red_class_ids, red_damage_one_hot = split_attrs(red_team_attrs)

        # Convert lists to tensors
        blue_team_ids = torch.tensor(blue_team_ids, dtype=torch.long)
        red_team_ids = torch.tensor(red_team_ids, dtype=torch.long)
        blue_numerical = torch.tensor(blue_numerical, dtype=torch.float)
        red_numerical = torch.tensor(red_numerical, dtype=torch.float)
        blue_class_ids = torch.tensor(blue_class_ids, dtype=torch.long)
        red_class_ids = torch.tensor(red_class_ids, dtype=torch.long)
        blue_damage_one_hot = torch.tensor(blue_damage_one_hot, dtype=torch.float)
        red_damage_one_hot = torch.tensor(red_damage_one_hot, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)

        return (blue_team_ids, red_team_ids,
                blue_numerical, red_numerical,
                blue_class_ids, red_class_ids,
                blue_damage_one_hot, red_damage_one_hot,
                label)
