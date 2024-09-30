from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
from bs4 import BeautifulSoup

# Initialize Driver
driver = webdriver.Chrome()  # Or the driver for your chosen browser
driver.get('https://www.metasrc.com/lol/14.18/stats?ranks=diamond,master,grandmaster,challenger')
time.sleep(10)
#Wait for content to load
wait = WebDriverWait(driver, 10)

print(driver.page_source)

html_content = driver.page_source

soup = BeautifulSoup(html_content, 'html.parser')

rows = soup.find_all('tr', {'class': '_sbzxul'})

data = []

for row in rows:
    cells = row.find_all('td')
    if cells:
        # Extract champion name
        champ_cell = cells[0]
        champ_name = champ_cell.find('span', {'hidden': 'hidden'}).text.strip()
        role = cells[1].text.strip()
        tier = cells[2].text.strip()
        score = cells[3].text.strip()
        delta = cells[4].text.strip()
        win_rate = cells[5].text.strip()
        pick_rate = cells[6].text.strip()
        ban_rate = cells[7].text.strip()
        main_rate = cells[8].text.strip()
        kda = cells[9].text.strip()

        data.append({
            'Champion': champ_name,
            'Role': role,
            'Tier': tier,
            'Score': score,
            'Delta': delta,
            'Win Rate': win_rate,
            'Pick Rate': pick_rate,
            'Ban Rate': ban_rate,
            'Main Rate': main_rate,
            'KDA': kda
        })

# Convert to DataFrame
df = pd.DataFrame(data)

df.to_csv('champion_winrate_stats.csv', index=False)

print(df)