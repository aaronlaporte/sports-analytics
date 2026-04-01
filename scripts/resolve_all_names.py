"""
scripts/resolve_all_names.py -- Resolve ALL player names via MLB Stats API.

pybaseball's playerid_reverse_lookup misses minor leaguers. The MLB Stats API
(statsapi.mlb.com) covers everyone with an MLBAM ID.

Also populates current_team for each player.

Usage:
    python scripts/resolve_all_names.py
"""

import json
import sqlite3
import time
import urllib.request

DB_PATH = "data/mlb.db"
API_BASE = "https://statsapi.mlb.com/api/v1/people"


def resolve_names():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Ensure player_lookup has a current_team column
    try:
        conn.execute("ALTER TABLE player_lookup ADD COLUMN current_team TEXT")
        conn.commit()
        print("Added current_team column to player_lookup.")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Get ALL unique player IDs from statcast (batters + pitchers)
    batter_ids = {r[0] for r in conn.execute(
        "SELECT DISTINCT batter FROM statcast_staging"
    ).fetchall()}
    pitcher_ids = {r[0] for r in conn.execute(
        "SELECT DISTINCT pitcher FROM statcast_staging"
    ).fetchall()}
    all_ids = batter_ids | pitcher_ids

    # Get existing resolved names
    existing = {}
    for r in conn.execute("SELECT mlbam_id, player_name, current_team FROM player_lookup"):
        existing[r["mlbam_id"]] = (r["player_name"], r["current_team"])

    # Find IDs that need resolution (no name or no team)
    needs_name = [pid for pid in all_ids if pid not in existing or existing[pid][0] is None]
    needs_team = [pid for pid in all_ids if pid in existing and existing[pid][0] is not None and existing[pid][1] is None]

    to_resolve = set(needs_name) | set(needs_team)
    print(f"Total unique player IDs: {len(all_ids)}")
    print(f"Already resolved with name+team: {len(all_ids) - len(to_resolve)}")
    print(f"Need resolution: {len(to_resolve)} ({len(needs_name)} missing name, {len(needs_team)} missing team)")

    if not to_resolve:
        print("All players resolved.")
        conn.close()
        return

    # Batch via MLB API (supports comma-separated IDs, up to ~100 per request)
    resolved = 0
    failed = 0
    batch_size = 100
    id_list = sorted(to_resolve)

    for i in range(0, len(id_list), batch_size):
        batch = id_list[i:i + batch_size]
        ids_str = ",".join(str(pid) for pid in batch)
        url = f"{API_BASE}?personIds={ids_str}&hydrate=currentTeam"

        try:
            req = urllib.request.Request(url)
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read())

            for person in data.get("people", []):
                pid = person["id"]
                name = person.get("fullName", "").lower()
                team_obj = person.get("currentTeam", {})
                team_abbr = team_obj.get("abbreviation") or team_obj.get("name")

                if name:
                    conn.execute(
                        "INSERT OR REPLACE INTO player_lookup (mlbam_id, player_name, current_team) VALUES (?, ?, ?)",
                        (pid, name, team_abbr),
                    )
                    resolved += 1

        except Exception as e:
            print(f"  Batch {i//batch_size + 1} failed: {e}")
            failed += len(batch)

        if (i // batch_size + 1) % 10 == 0:
            conn.commit()
            print(f"  Progress: {i + len(batch)}/{len(id_list)} ({resolved} resolved)")

        time.sleep(0.2)  # Be nice to the API

    conn.commit()

    # Final stats
    total_lookup = conn.execute("SELECT COUNT(*) FROM player_lookup WHERE player_name IS NOT NULL").fetchone()[0]
    still_missing = conn.execute("""
        SELECT COUNT(DISTINCT b.batter)
        FROM statcast_staging b
        LEFT JOIN player_lookup p ON b.batter = p.mlbam_id
        WHERE p.player_name IS NULL
    """).fetchone()[0]

    print(f"\nResolution complete:")
    print(f"  Resolved this run: {resolved}")
    print(f"  Failed: {failed}")
    print(f"  Total in player_lookup: {total_lookup}")
    print(f"  Still missing (batters): {still_missing}")
    conn.close()


if __name__ == "__main__":
    resolve_names()
