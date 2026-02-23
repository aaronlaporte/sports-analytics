# SPORTS ANALYTICS PLATFORM — IMPLEMENTATION PLAN
Generated: 2026-02-23

## GOAL
Build a consolidated sports analytics platform with:
- Data pipelines that pull stats → SQLite → Tableau (personal use)
- Zip integration scripts in `zip-tools/sports/` for live on-demand queries via Telegram
- Three sports: MLB (priority, April), NBA (live now), NFL (September)

## CONTEXT
- API Keys: stored in `keys/` (gitignored) + 1Password
  - SportsDataIO: stored in keys/sdio_key.txt
  - The Odds API: stored in keys/odds_key.txt
- Existing repos migrated/consolidated from:
  - mlb-betting-analytics → mlb/
  - nfl_analytics → nfl/
- New: nba/ (built from scratch)
- SportsDataIO free tier covers: full stats, schedules, injuries for NFL/MLB/NBA
  - NOT available: projections, historical odds (401 locked)
- The Odds API free tier (500 req/month):
  - Player props ONLY work via /v4/sports/{sport}/events/{id}/odds (event-level endpoint)
  - Top-level /odds/ endpoint does NOT support player prop markets
  - NBA markets confirmed: player_points, player_rebounds, player_assists, player_threes
  - MLB markets confirmed: batter_hits, pitcher_strikeouts
  - Pull on-demand only (not cron) to stay within 500/month

---

## REPO STRUCTURE

```
sports-analytics/
├── shared/
│   ├── sdio_client.py          # SDIO API wrapper (ported from nfl_analytics)
│   ├── odds_client.py          # The Odds API wrapper (new)
│   ├── db.py                   # SQLite helpers + schema definitions
│   └── config.py               # Paths, seasons, key loading
├── mlb/
│   ├── ingest/
│   │   ├── pull_statcast.py    # pybaseball Statcast (ported + SQLite output)
│   │   ├── pull_schedules.py   # SDIO: daily schedule + lineup confirmation
│   │   └── pull_injuries.py    # SDIO: injury reports
│   ├── features/
│   │   └── build_features.py   # Ported + enhanced stats_builders.py → SQLite
│   ├── model/
│   │   └── train_model.py      # Beta-binomial + optional ML layer
│   ├── predict/
│   │   ├── daily_picks.py      # Top 10 hit picks w/ auto EV (no manual template)
│   │   └── daily_bets_plan.py  # 4-bet daily plan
│   ├── odds/
│   │   └── pull_live_odds.py   # Odds API: batter_hits, pitcher_strikeouts per game
│   └── run_pipeline.py         # Orchestrator: ingest → features → predict → odds
├── nba/
│   ├── ingest/
│   │   ├── pull_player_stats.py # SDIO: player game stats (80+ fields per player)
│   │   ├── pull_schedules.py    # SDIO: game schedule, home/away
│   │   └── pull_injuries.py     # SDIO: injury status + notes
│   ├── features/
│   │   └── build_features.py    # Rolling windows, advanced metrics, opp rank
│   ├── model/
│   │   └── train_model.py       # 4 calibrated models: pts, reb, ast, 3PM
│   ├── predict/
│   │   └── daily_picks.py       # Top picks per prop category w/ EV
│   ├── odds/
│   │   └── pull_live_odds.py    # Odds API: player_points/rebounds/assists/threes
│   └── run_pipeline.py
├── nfl/
│   ├── ingest/                  # Ported from nfl_analytics (enhanced)
│   │   ├── pull_schedule.py
│   │   ├── pull_players.py
│   │   ├── pull_player_game_stats.py
│   │   ├── pull_team_game_stats.py
│   │   └── pull_injuries.py
│   ├── features/
│   │   └── build_training_table.py  # Enhanced: all 32 teams, SQLite output
│   ├── model/
│   │   └── train_model.py      # Enhanced: backtest added, all teams, calibrated
│   ├── predict/
│   │   └── predict_week.py     # All 32 teams (was limited to 4)
│   ├── odds/
│   │   └── pull_live_odds.py   # Odds API: anytime_td, receiving_yards (in-season only)
│   └── run_pipeline.py
├── data/
│   ├── mlb.db                  # SQLite — all MLB data + model outputs
│   ├── nba.db                  # SQLite — all NBA data + model outputs
│   └── nfl.db                  # SQLite — all NFL data + model outputs
├── keys/
│   ├── sdio_key.txt            # gitignored — SportsDataIO API key
│   └── odds_key.txt            # gitignored — The Odds API key
├── .gitignore
├── requirements.txt
└── README.md
```

---

## PHASE 1: MLB PIPELINE ENHANCEMENT
**Priority — season starts April, pipeline must be production-ready**

### What stays (working well in mlb-betting-analytics)
- pybaseball Statcast pull with incremental caching — keep as primary stats source
- Beta-binomial shrinkage hit probability model — interpretable, proven
- Backtesting (backtest_hit_prob.py, backtest_real_odds.py)
- Daily picks + 2-leg parlay combo generation logic
- Weekly PDF report generation

### What changes

**1. Migrate CSV → SQLite (mlb.db)**

| Table | Replaces | Description |
|-------|----------|-------------|
| `statcast_raw` | data/archive/statcast_raw.csv | Raw pitch-by-pitch, deduplicated |
| `batter_stats` | data/batter_stats.csv | Daily aggregated batter metrics |
| `pitcher_stats` | data/pitcher_stats.csv | Daily pitcher metrics |
| `matchup_stats` | data/matchup_stats.csv | Batter vs pitcher history |
| `model_params` | tracking/hit_prob_model.json | Trained model parameters |
| `daily_picks` | tracking/daily_picks_*.csv | Predictions + EV, date-keyed |
| `live_odds` | (new) | Lines + odds from The Odds API |
| `bets` | tracking/bet_tracking.xlsx | Personal betting log, append-only |

**2. Add confirmed lineup data via SDIO**
- Endpoint: `/v3/mlb/scores/json/GamesByDate/{date}` → check `LineupConfirmed`
- Endpoint: `/v3/mlb/scores/json/Lineups/{gameId}` → get confirmed batting order
- Filter picks to confirmed lineup players only (eliminates scratched players)
- Attach confirmed starting pitcher to each pick (replaces proxy)
- Pull pitcher handedness from SDIO player data for hand-split accuracy

**3. Auto-pull live odds (replaces manual odds template)**
- `pull_live_odds.py`:
  1. Fetch today's MLB games from Odds API (`/v4/sports/baseball_mlb/odds/`)
  2. For each game event ID, fetch player props (`/v4/sports/baseball_mlb/events/{id}/odds`)
  3. Markets: `batter_hits` (over 0.5 hits = 1+ hits line), `pitcher_strikeouts`
  4. Write lines + American odds to `live_odds` table
- `daily_picks.py` auto-joins picks with live odds → auto-calculates EV
- EV formula: `(model_prob * decimal_odds) - 1`
- Flag picks with EV > 0.05 as value bets
- Eliminates `make_odds_template.py`, `log_odds.py`, and manual copy-paste entirely

**4. Pitcher confirmed matchup**
- SDIO player roster: pull pitcher throwing hand (L/R)
- Replace the "last faced pitcher" proxy with actual confirmed starter
- Recompute hand splits against confirmed pitcher handedness

---

## PHASE 2: NBA PIPELINE
**Build now — active season, live odds available to test against**

### Data confirmed available from SDIO probe
- 317 players per game night with 80+ fields
- Advanced metrics: TrueShootingPct, UsageRatePercentage, PlayerEfficiencyRating, AssistsPct
- Context: HomeOrAway, OpponentRank, OpponentPositionRank
- Fantasy salaries: DraftKingsSalary, FanDuelSalary, YahooSalary (proxy for role/usage)
- Injury: InjuryStatus, InjuryBodyPart, InjuryNotes, LineupStatus

### SQLite Schema (nba.db)

| Table | Description |
|-------|-------------|
| `player_game_stats` | Raw SDIO data, one row per player per game |
| `schedules` | Game schedule with home/away, date |
| `features` | Engineered rolling features, built daily |
| `daily_picks` | Model predictions + live odds + EV |
| `live_odds` | The Odds API player prop lines |

### Feature Engineering (build_features.py)

Rolling windows (3, 5, 10, 20 games) for each stat:
- Points, Rebounds, Assists, ThreePointersMade, Steals, Blocks, Turnovers, Minutes
- FieldGoalsAttempted, FreeThrowsAttempted

Advanced rolling:
- UsageRatePercentage, TrueShootingPercentage, PlayerEfficiencyRating

Contextual features:
- `is_home` (HomeOrAway == 'HOME')
- `opponent_rank` (OpponentRank — lower = tougher defense)
- `opponent_position_rank` (position-specific defensive ranking)
- `days_rest` (days since last game from schedule)
- `injury_status` (0=healthy, 1=questionable, 2=doubtful)
- `salary_dk`, `salary_fd` (proxy for projected usage/role)
- `season_avg_pts`, `season_avg_reb`, `season_avg_ast` (season priors)

### Models (train_model.py)

Four separate calibrated HistGradientBoosting classifiers:
- `pts_model`: P(player scores OVER their points line tonight)
- `reb_model`: P(player grabs OVER their rebounds line tonight)
- `ast_model`: P(player dishes OVER their assists line tonight)
- `threes_model`: P(player hits OVER their 3PM line tonight)

Training strategy:
- Use 2024-25 season data for initial training
- As 2025-26 season accumulates, retrain on combined data
- Chronological split (no future leakage)
- Calibrated with CalibratedClassifierCV for reliable probabilities

### Live Odds (pull_live_odds.py)

```
1. GET /v4/sports/basketball_nba/odds/?markets=h2h → get game list + event IDs
2. For each game tonight:
   GET /v4/sports/basketball_nba/events/{id}/odds?markets=player_points,player_rebounds,player_assists,player_threes
3. Parse: player name, market, Over/Under, line (point), American odds
4. Write to live_odds table
5. Join with daily_picks → compute EV per bet
```

API cost: ~4 games/night = 4 requests. If pulled 15 nights/month = 60 requests.

### Daily Output (daily_picks.py)

Top 5 picks per category (points, rebounds, assists, 3s) with:
- Player name, team, opponent
- Model probability
- Book line + American odds (best available from FanDuel/DraftKings)
- EV (auto-calculated)
- Value flag (EV > 0.05)

Store in `nba.db → daily_picks`, also export CSV for Tableau.

---

## PHASE 3: NFL PIPELINE ENHANCEMENT
**Lower priority — build/refine during offseason for September**

### Key improvements over nfl_analytics

1. **Remove 4-team filter** — predict all 32 teams (was LAR, SEA, NE, DEN only)
2. **SQLite migration** — replace raw JSONs + CSVs with `nfl.db`
3. **Add backtest module** — missing from V1, needed to validate model
4. **Retrain with all-team data** — broader training set improves generalization
5. **In-season**: The Odds API for `player_anytime_touchdown`, `player_reception_yards`
6. **Touch share** — pull snap count / target share from SDIO if available on free tier
7. **Short-week adjustment** — encode days rest (Thu night, MNF → Sunday turnaround)

### NFL is offseason until September
- No live props to pull right now — build and validate pipeline on 2025 season data
- The Odds API NFL endpoints will activate when 2026 season schedule drops (~May)

---

## ZIP INTEGRATION (zip-tools/sports/)

### SKILL.md
OpenClaw skill registered at `/opt/openclaw/skills/sports/SKILL.md`

Zip uses this skill when asked:
- "Who should I bet on tonight in the NBA?"
- "What are the best MLB prop bets today?"
- "Give me my fantasy waiver targets this week"
- "What's the EV on Tatum over 25.5 points?"

### analyze.py

```bash
# Tonight's NBA picks with live odds and EV
python analyze.py --nba-tonight

# Today's MLB picks (pulls live odds if lineup confirmed)
python analyze.py --mlb-today

# NFL picks for current week (in-season only)
python analyze.py --nfl-week

# Player-specific analysis
python analyze.py --player "Jayson Tatum" --sport nba

# Full summary all active sports
python analyze.py --full
```

Behavior:
- Queries SQLite DBs for model predictions + recent form
- Calls Odds API for live lines only when explicitly requested (preserves quota)
- Returns plain-text summary for Zip to relay via Telegram

### Odds API Budget Management
- 500 requests/month on free tier
- Strategy: pull only on-demand via Zip, not on any cron schedule
- ~60 NBA prop requests/month + ~60 MLB prop requests/month = 120 total
- Leaves 380 buffer for testing, edge cases, NFL when in-season

---

## TABLEAU CONNECTIVITY

All three SQLite DBs connect via SQLite ODBC driver (same as financials setup).
Key Tableau data sources:
- `player_game_stats` — historical game-by-game performance
- `daily_picks` — today's model predictions
- `live_odds` — current sportsbook lines
- `bets` — personal betting log with outcomes + ROI over time

---

## BUILD ORDER (next session)

1. Init `sports-analytics` repo: structure, .gitignore, requirements.txt, keys/
2. `shared/`: sdio_client.py, odds_client.py, db.py, config.py
3. MLB Phase 1: SQLite schema + port existing pipeline scripts (ingest + features)
4. MLB Phase 2: Add SDIO lineup confirmation
5. MLB Phase 3: Odds API integration + auto-EV (eliminates manual template)
6. NBA Phase 1: Ingest + SQLite schema, start pulling 2024-25 season data
7. NBA Phase 2: Feature engineering + model training
8. NBA Phase 3: Odds API integration + daily_picks
9. NFL: Migrate + enhance (September priority)
10. Zip integration: SKILL.md + analyze.py + trigger_zip.sh → deploy to VM

---

## NEW SESSION PROMPT

Paste this into a new Claude Code session to pick up where this plan left off:

```
@/Users/aaronlaporte/Documents/GitHub/sports-analytics/PLAN.md

You are building a consolidated sports analytics platform for Aaron.
Read PLAN.md completely before starting anything.

CRITICAL CONTEXT:
- SportsDataIO API key: in keys/sdio_key.txt (create the file)
- The Odds API key: in keys/odds_key.txt (create the file)
- Aaron uses Yahoo, Sleeper, ESPN for fantasy sports
- Priority: MLB (April season start) → NBA (live season) → NFL (September)
- Existing repos to port from:
    ~/Documents/GitHub/mlb-betting-analytics
    ~/Documents/GitHub/nfl_analytics
- New repo already created at: ~/Documents/GitHub/sports-analytics

API BEHAVIOR (confirmed by prior testing — do not re-probe):
- SDIO: /v3/nba/stats/json/PlayerGameStatsByDate/{YYYY-MON-DD} = 317 players/game, 80+ fields
- SDIO: /v3/mlb/scores/json/GamesByDate/{date} = schedule + LineupConfirmed flag
- SDIO: Projections endpoint = 401 unauthorized (not on free plan)
- The Odds API: Player props ONLY via /v4/sports/{sport}/events/{id}/odds
  - Top-level /odds/ endpoint does NOT support player_points etc.
  - NBA markets: player_points, player_rebounds, player_assists, player_threes ✓
  - MLB markets: batter_hits, pitcher_strikeouts ✓
  - 500 req/month — pull on-demand only, not cron
- NFL is offseason — no live props until September 2026

START: Build shared/ utilities and MLB pipeline first.
Ask Aaron questions as you work. Do not assume.
```
