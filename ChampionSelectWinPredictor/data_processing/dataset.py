from torch.utils.data import Dataset
import torch


class LoLMatchDataset(Dataset):
    def __init__(self, matches):
        self.matches = matches

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        match = self.matches[idx]
        blue_team_ids = torch.tensor(match['blue_team_ids'], dtype=torch.long)
        red_team_ids = torch.tensor(match['red_team_ids'], dtype=torch.long)

        # Adjust champion IDs to be indices starting at 0
        blue_team_indices = blue_team_ids - 1
        red_team_indices = red_team_ids - 1

        # Convert heuristics to tensors
        blue_heuristics = torch.tensor(list(match['blue_heuristics'].values()), dtype=torch.float)
        red_heuristics = torch.tensor(list(match['red_heuristics'].values()), dtype=torch.float)

        label = torch.tensor([match['label']], dtype=torch.float)

        return (blue_team_indices, red_team_indices, blue_heuristics, red_heuristics, label)