# match_data_processor.py

import pandas as pd
from ChampionSelectWinPredictor.utils import normalize_name
import re


class MatchDataProcessor:
    def __init__(self, csv_file, champion_attributes, translator):
        self.csv_file = csv_file
        self.champion_attributes = champion_attributes
        self.translator = translator
        self.matches = self.load_and_process_matches()
        self.unique_champions = self.get_unique_champions()

    def load_and_process_matches(self):
        data = pd.read_csv(self.csv_file)
        data.sort_values(by=['match_matchId', 'player_teamId'], inplace=True)
        grouped_matches = data.groupby('match_matchId')
        matches = []

        for match_id, match_data in grouped_matches:
            blue_team_data = match_data[match_data['player_teamId'] == 'blue']
            red_team_data = match_data[match_data['player_teamId'] == 'red']

            if len(blue_team_data) == 5 and len(red_team_data) == 5:
                blue_team_champs = blue_team_data['player_champName'].tolist()
                red_team_champs = red_team_data['player_champName'].tolist()
                blue_team_win = blue_team_data['player_win'].iloc[0]
                label = 1 if blue_team_win == 1 else 0
                matches.append((blue_team_champs, red_team_champs, label))

        return matches

    def get_unique_champions(self):
        champions = set()
        for blue_team, red_team, _ in self.matches:
            champions.update(blue_team)
            champions.update(red_team)
        return list(champions)

    def default_attributes(self):
        # Define default attributes if any attributes are missing
        # Adjust the length according to the number of features
        return {
            'Style': 0.0,
            'Difficulty': 0.0,
            'Damage': 0.0,
            'Sturdiness': 0.0,
            'Crowd-Control': 0.0,
            'Mobility': 0.0,
            'Functionality': 0.0,
            'ClassID': 0,
            'DamageTypeOneHot': [0.0, 0.0, 0.0]
        }

    def get_champion_attrs(self, champ):
        # Normalize champion name
        normalized_champ_name = normalize_name(champ)
        champ_id = self.translator.get_champion_id(normalized_champ_name)
        if champ_id is None:
            print(f"Champion ID not found for champion '{champ}'. Using default attributes.")
            attrs = self.default_attributes()
        else:
            champ_id_str = str(champ_id)
            attrs = self.champion_attributes.get(champ_id_str, self.default_attributes())
            if attrs == self.default_attributes():
                print(f"Attributes not found for champion ID {champ_id_str}, using default attributes.")

        # Extract normalized numerical features
        numerical_features = ['Style', 'Difficulty', 'Damage', 'Sturdiness', 'Crowd-Control', 'Mobility',
                              'Functionality']
        numerical_values = [float(attrs.get(feature, 0.0)) for feature in numerical_features]

        # Get ClassID
        class_id = int(attrs.get('ClassID', 0))

        # Get DamageTypeOneHot
        damage_type_one_hot = attrs.get('DamageTypeOneHot', [0.0, 0.0, 0.0])
        damage_type_one_hot = [float(x) for x in damage_type_one_hot]

        # Combine all features into a single list
        all_attrs = numerical_values + [class_id] + damage_type_one_hot

        return all_attrs

    # match_data_processor.py
    def convert_matches_to_ids(self, champion_to_id):
        matches_ids = []
        for blue_team, red_team, label in self.matches:
            blue_team_ids = [champion_to_id[normalize_name(champ)] for champ in blue_team]
            red_team_ids = [champion_to_id[normalize_name(champ)] for champ in red_team]

            # Get attributes for each champion
            blue_team_attrs = [self.get_champion_attrs(champ) for champ in blue_team]
            red_team_attrs = [self.get_champion_attrs(champ) for champ in red_team]

            matches_ids.append((blue_team_ids, red_team_ids, blue_team_attrs, red_team_attrs, label))
        return matches_ids

