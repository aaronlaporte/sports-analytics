"""Resolve missing player names in player_lookup table."""
import sqlite3
from pybaseball import playerid_reverse_lookup

db = sqlite3.connect("data/mlb.db")
cur = db.cursor()

# Find batter IDs with no name
cur.execute("""
    SELECT DISTINCT h.batter_id FROM hr_features h
    LEFT JOIN player_lookup p ON h.batter_id = p.mlbam_id
    WHERE p.player_name IS NULL
    UNION
    SELECT DISTINCT b.batter_id FROM batter_stats b
    LEFT JOIN player_lookup p ON b.batter_id = p.mlbam_id
    WHERE p.player_name IS NULL
""")
missing = [r[0] for r in cur.fetchall()]
print(f"Missing names: {len(missing)}")

for i in range(0, len(missing), 500):
    chunk = missing[i : i + 500]
    try:
        result = playerid_reverse_lookup(chunk, key_type="mlbam")
        for _, row in result.iterrows():
            mlbam = int(row["key_mlbam"])
            first = str(row.get("name_first", "")).strip()
            last = str(row.get("name_last", "")).strip()
            name = f"{first} {last}".strip()
            if name and name != " ":
                cur.execute(
                    "INSERT OR REPLACE INTO player_lookup VALUES (?, ?)",
                    (mlbam, name),
                )
        print(f"  Chunk {i // 500 + 1}: resolved {len(result)}")
    except Exception as e:
        print(f"  Chunk {i // 500 + 1} error: {e}")

db.commit()
cur.execute("SELECT COUNT(*) FROM player_lookup")
print(f"Total names now: {cur.fetchone()[0]}")
db.close()
print("Done.")
