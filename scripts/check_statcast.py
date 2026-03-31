"""Quick check of Statcast data availability."""
import sys
from pybaseball import statcast

date = sys.argv[1] if len(sys.argv) > 1 else "2026-03-29"
print(f"Pulling Statcast for {date}...")
df = statcast(start_dt=date, end_dt=date)
print(f"Rows returned: {len(df)}")
if len(df) > 0:
    print(f"Unique batters: {df['batter'].nunique()}")
    print(f"Unique games: {df['game_pk'].nunique()}")
    print(df[["game_date", "batter", "events"]].dropna(subset=["events"]).head(10))
else:
    print("No data returned from pybaseball")
