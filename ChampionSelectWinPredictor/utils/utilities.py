import re

def normalize_name(champ_name):
    # Your existing normalize_name function...
    champ_name = champ_name.lower()
    champ_name = re.sub(r"[^a-z0-9]+", "", champ_name)
    return champ_name