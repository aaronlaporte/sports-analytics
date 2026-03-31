"""Force ingest Statcast data for a specific date."""
import sys
import sqlite3
from pybaseball import statcast

date = sys.argv[1] if len(sys.argv) > 1 else "2026-03-29"
print(f"Pulling Statcast for {date}...")

df = statcast(start_dt=date, end_dt=date)
print(f"Got {len(df)} rows")

if len(df) == 0:
    print("No data available")
    sys.exit(1)

cols = [
    "game_pk", "game_date", "batter", "pitcher", "events", "description",
    "bb_type", "hit_distance_sc", "launch_speed", "launch_angle",
    "estimated_ba_using_speedangle", "stand", "p_throws", "home_team",
    "away_team", "inning_topbot", "at_bat_number", "pitch_number",
]
available = [c for c in cols if c in df.columns]
df2 = df[available].copy()

db = sqlite3.connect("data/mlb.db")
# Delete existing data for this date to avoid duplicates
db.execute("DELETE FROM statcast_staging WHERE game_date = ?", (date,))
df2.to_sql("statcast_staging", db, if_exists="append", index=False)
db.commit()
db.close()
print(f"Ingested {len(df2)} rows for {date}")
