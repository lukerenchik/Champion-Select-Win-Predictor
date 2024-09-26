import pandas as pd

def normalize_name(name):
    return name.strip().lower().replace("'", "").replace(" ", "")


class MatchDataProcessor:
    def __init__(self, csv_file, champion_attributes, translator):
        self.csv_file = csv_file
        self.champion_attributes = champion_attributes
        self.translator = translator
        self.matches = self.load_matches()
        self.unique_champion_ids = self.get_unique_champion_ids()

    def load_matches(self):
        # Load match data from CSV
        df = pd.read_csv(self.csv_file)

        # Ensure necessary columns are present
        required_columns = [
            'match_matchId', 'player_teamId', 'player_champName', 'player_win'
        ]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in CSV file.")

        # Convert 'player_win' to integer (1 or 0)
        df['player_win'] = df['player_win'].astype(str).map({'True': 1, 'False': 0})
        # Drop rows with NaN in 'player_win' after mapping
        df = df.dropna(subset=['player_win'])
        df['player_win'] = df['player_win'].astype(int)

        # Group by 'match_matchId' to assemble matches
        grouped = df.groupby('match_matchId')
        matches = []

        for match_id, group in grouped:
            blue_team = group[group['player_teamId'] == 'blue']
            red_team = group[group['player_teamId'] == 'red']

            blue_team_champions = blue_team['player_champName'].tolist()
            red_team_champions = red_team['player_champName'].tolist()

            # Ensure each team has 5 champions
            if len(blue_team_champions) != 5 or len(red_team_champions) != 5:
                # Skip matches that don't have 5 players on each team
                continue

            # Get champion IDs using the translator
            blue_team_ids = [self.translator.get_champion_id(champ) for champ in blue_team_champions]
            red_team_ids = [self.translator.get_champion_id(champ) for champ in red_team_champions]

            # Check for any None values (champions not found)
            if None in blue_team_ids or None in red_team_ids:
                # Skip matches with unknown champions
                continue

            # Determine the winner
            blue_win = blue_team['player_win'].iloc[0]
            label = blue_win  # 1 if blue team wins, 0 if red team wins

            match = {
                'blue_team_ids': blue_team_ids,  # Champion IDs
                'red_team_ids': red_team_ids,  # Champion IDs
                'label': label
            }
            matches.append(match)

        return matches

    def get_unique_champion_ids(self):
        champion_ids = set()
        for match in self.matches:
            champion_ids.update(match['blue_team_ids'])
            champion_ids.update(match['red_team_ids'])
        return list(champion_ids)

    def compute_team_heuristics(self, team_champion_ids):
        # Initialize counts
        counts = {
            'isHypercarry': 0,
            'isEnchanter': 0,
            'isWarden': 0,
            'isPhysicalDmg': 0,
            'isMagicalDmg': 0,
            'isDiver': 0,
            'isSplitpusher': 0,
            'isPicker': 0,
            'isSieger': 0,
            'isEngager': 0,
            'hasZoneControl': 0,
            'frontlineMelter': 0,
            'bigNuke': 0,
            'isAntiDive': 0,
            'isSnowballer': 0,
            'isSnowballEnabler': 0,
            'isFrontliner': 0,
            'antiInvis': 0,
            'isAntiSnowballer': 0,
            'hasInvis': 0,
            # Add other attributes if needed
        }

        # Sum attributes over team champions
        for champ_id in team_champion_ids:
            champ_attrs = self.champion_attributes[str(champ_id)]
            for attr in counts.keys():
                counts[attr] += int(champ_attrs[attr])  # Ensure attributes are integers

        # Compute heuristics based on counts
        heuristics = {}

        # Higher-Level Heuristics Implementation
        heuristics['hasProtectComp'] = int(counts['isHypercarry'] >= 1 and counts['isEnchanter'] >= 1)
        heuristics['hasStrongProtectComp'] = int(
            (counts['isHypercarry'] >= 1 and counts['isEnchanter'] >= 1 and counts['isWarden'] >= 1) or
            (counts['isHypercarry'] >= 1 and counts['isEnchanter'] >= 2)
        )
        heuristics['hasCompetingHypercarries'] = int(counts['isHypercarry'] > 1)
        heuristics['hasUnbalancedDamageProfile'] = int(counts['isPhysicalDmg'] == 0 or counts['isMagicalDmg'] == 0)
        heuristics['hasStrongDamageProfile'] = int(counts['isPhysicalDmg'] >= 2 and counts['isMagicalDmg'] >= 2)
        heuristics['hasDiveComp'] = int(counts['isDiver'] >= 2)
        heuristics['hasSuperDiveComp'] = int(counts['isDiver'] >= 3)
        heuristics['hasFourOneSplit'] = int(counts['isSplitpusher'] == 1)
        heuristics['hasOneThreeOneSplit'] = int(counts['isSplitpusher'] == 2)
        heuristics['hasTooManySidelaners'] = int(counts['isSplitpusher'] >= 3)
        heuristics['hasAbilityToCatch'] = int(counts['isPicker'] == 1)
        heuristics['hasCatchComp'] = int(counts['isPicker'] == 2)
        heuristics['hasTooMuchCatch'] = int(counts['isPicker'] >= 3)
        heuristics['hasSiegeSetup'] = int(counts['isSieger'] == 1)
        heuristics['hasStrongSiege'] = int(counts['isSieger'] == 2)
        heuristics['hasTooMuchSiege'] = int(counts['isSieger'] >= 3)
        heuristics['hasEngageOption'] = int(counts['isEngager'] == 1)
        heuristics['hasStrongEngage'] = int(counts['isEngager'] == 2)
        heuristics['hasTooMuchEngage'] = int(counts['isEngager'] >= 3)
        heuristics['canZoneEngagers'] = int(counts['hasZoneControl'] >= 1)
        heuristics['tanksAreWorse'] = int(counts['frontlineMelter'] >= 1)
        heuristics['hasWombo'] = int(counts['bigNuke'] >= 1 and counts['isEngager'] >= 1)
        heuristics['hasBigWombo'] = int(counts['bigNuke'] >= 2 and counts['isEngager'] >= 2)
        heuristics['hasConsistentWombo'] = int(counts['bigNuke'] >= 1 and counts['isEngager'] >= 2)
        heuristics['canStopDive'] = int(counts['isAntiDive'] >= 1)
        heuristics['hasCounterDive'] = int(counts['isAntiDive'] >= 2)
        heuristics['snowballersNotEnabled'] = int(counts['isSnowballer'] >= 1 and counts['isSnowballEnabler'] == 0)
        heuristics['snowballersEnabled'] = int(counts['isSnowballer'] >= 1 and counts['isSnowballEnabler'] == 1)
        heuristics['snowballersDream'] = int(counts['isSnowballer'] >= 1 and counts['isSnowballEnabler'] >= 2)
        heuristics['tooManySnowballers'] = int(counts['isSnowballer'] >= 3)
        heuristics['hasBalancedComp'] = int(counts['isFrontliner'] >= 1 and counts['isHypercarry'] >= 1 and counts['isEnchanter'] >= 1)
        heuristics['hasInvisibility'] = int(counts['hasInvis'] >= 1)
        heuristics['hasAntiInvisibility'] = int(counts['antiInvis'] >= 1)
        heuristics['hasAntiSnowball'] = int(counts['isAntiSnowballer'] >= 1)

        # Add any additional heuristics here

        return heuristics

    def convert_matches_to_ids_with_heuristics(self):
        match_data = []
        for match in self.matches:
            blue_team_ids = match['blue_team_ids']
            red_team_ids = match['red_team_ids']
            label = match['label']

            # Compute heuristics
            blue_heuristics = self.compute_team_heuristics(blue_team_ids)
            red_heuristics = self.compute_team_heuristics(red_team_ids)

            match_data.append({
                'blue_team_ids': blue_team_ids,
                'red_team_ids': red_team_ids,
                'blue_heuristics': blue_heuristics,
                'red_heuristics': red_heuristics,
                'label': label
            })
        return match_data



