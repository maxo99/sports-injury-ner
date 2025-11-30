import pandas as pd
import requests
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def get_injuries_espn():
    INJURIES_URL = "https://www.espn.com/nfl/injuries"

    response = requests.get(INJURIES_URL, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    # Find all of the teams table scrollers div
    scrollers = soup.find_all("div", class_="Table__Scroller")

    data = []
    for scroller in scrollers:
        table = scroller.find("table")
        tbody = table.find("tbody")
        rows = tbody.find_all("tr")

        for row in rows:
            cells = row.find_all("td")
            row_data = [cell.get_text(strip=True) for cell in cells]
            if row_data:
                data.append(row_data)

    # Create DataFrame and save to CSV
    columns = ["Name", "Position", "Date", "Status", "Comment"]
    df = pd.DataFrame(data, columns=columns)

    print(f"Extracted {len(df)} rows.")
    print(df.head())

    df.to_csv("injuries_espn.csv", index=False)
    print("Saved to injuries_espn.csv")


def get_injuries_nfl():
    INJURIES_URL = "https://www.nfl.com/injuries/"
    response = requests.get(INJURIES_URL, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    tables = soup.find_all("table")

    data = []
    for table in tables:
        tbody = table.find("tbody")
        if not tbody:
            continue

        rows = tbody.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            row_data = [cell.get_text(strip=True) for cell in cells]
            if row_data:
                data.append(row_data)

    # Create DataFrame and save to CSV
    columns = ["Player", "Position", "Injuries", "Practice Status", "Game Status"]
    df = pd.DataFrame(data, columns=columns)

    print(f"Extracted {len(df)} rows from NFL.com")
    print(df.head())

    df.to_csv("injuries_nfl.csv", index=False)
    print("Saved to injuries_nfl.csv")


if __name__ == "__main__":
    print("Getting ESPN injuries...")
    get_injuries_espn()
    print("\nGetting NFL injuries...")
    get_injuries_nfl()
