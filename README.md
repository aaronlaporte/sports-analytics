# sports-analytics

Multi-sport player prop analytics platform. Covers MLB (hit probability), NBA (player props), and NFL (anytime TD). Data stored in SQLite, visualized in Tableau, surfaced via Zip (OpenClaw AI).

## Architecture

```
sports-analytics/
├── shared/           # Config, API clients, DB schema
│   ├── config.py         SDIO + Odds API keys, DB paths, constants
│   ├── sdio_client.py    SportsDataIO REST client (MLB/NBA/NFL)
│   ├── odds_client.py    The Odds API client (player props)
│   └── db.py             SQLite schema + connection helpers
├── mlb/
│   ├── ingest/           Statcast (pybaseball), schedules, injuries
│   ├── features/         Batter rolling stats → batter_stats table
│   ├── model/            Beta-binomial hit probability model
│   ├── odds/             Pull live batter_hits/pitcher_strikeouts props
│   ├── predict/          Daily picks + 4-bet plan (hits + HR)
│   └── run_pipeline.py   Full MLB pipeline orchestrator
├── nba/
│   ├── ingest/           Player box scores + schedules (SDIO)
│   ├── features/         Rolling 3/5/10/20 game windows + EWM
│   ├── model/            4 calibrated HGB models (pts/reb/ast/3s)
│   ├── odds/             Pull live player prop lines
│   ├── predict/          Score tonight's players, join odds, compute EV
│   └── run_pipeline.py   Full NBA pipeline orchestrator
├── nfl/
│   ├── ingest/           Schedule, players, game stats, team stats, injuries
│   ├── features/         TD training table (all 32 teams)
│   ├── model/            Anytime TD (LogReg+stacker) + 2+TD (Poisson) + rec (Ridge)
│   ├── odds/             Pull live anytime_td/reception_yards props
│   ├── predict/          Predict upcoming week for all 32 teams
│   └── run_pipeline.py   Full NFL pipeline orchestrator
├── data/             mlb.db, nba.db, nfl.db (gitignored)
└── keys/             sdio_key.txt, odds_key.txt (gitignored)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize databases
python shared/db.py

# NBA (in-season now)
python nba/run_pipeline.py

# MLB (April–October)
python mlb/run_pipeline.py

# NFL (September–February)
python nfl/run_pipeline.py
```

## API Keys

Place in `keys/` directory (gitignored) or set as env vars:

| File | Env Var | Service |
|------|---------|---------|
| `keys/sdio_key.txt` | `SDIO_API_KEY` | SportsDataIO (stats/schedules/injuries) |
| `keys/odds_key.txt` | `ODDS_API_KEY` | The Odds API (player props) |

**The Odds API budget**: 500 requests/month free tier. Pull odds on-demand only (not cron).
- ~60 NBA games/month × 1 req = 60
- ~60 MLB games/month × 1 req = 60
- ~16 NFL games/week × 4 weeks/month = 64
- ~184 total / month; plenty of buffer.

## Models

### MLB — Beta-Binomial Hit Probability
Shrinks batter season hit rate with recent-50-PA rate and pitcher-hand splits.
Grid search over prior_pa and blend weights, minimizes Brier score on chrono holdout.

### NBA — Calibrated Gradient Boosting
4 separate `CalibratedClassifierCV(HistGradientBoostingClassifier)` models:
points / rebounds / assists / 3-pointers made.
Features: rolling 3/5/10/20 windows, EWM(5), season averages, days rest, home/away,
opponent rank, injury flags, DK salary proxy.

### NFL — Logistic Regression + Poisson Stacker
- Anytime TD: `CalibratedClassifierCV(LogisticRegression)` global + per-position models stacked
- 2+ TD: `PoissonRegressor` → P(λ ≥ 2) via Poisson CDF
- Receptions: `Ridge` regression
Recency-weighted training (half-life = 45 days).

## Zip Integration

`zip-tools/sports/` contains:
- `SKILL.md` — OpenClaw skill definition (triggers: "nba picks", "mlb today", etc.)
- `analyze.py` — CLI frontend: `--nba-tonight`, `--mlb-today`, `--nfl-week`, `--player NAME`
- `trigger_zip.sh` — Generate report and wake Zip via Telegram

Deploy to VM: `scp -i ~/.ssh/codex_vm -r sports/ codex@10.8.0.1:/tmp/ && ssh ... sudo cp -r /tmp/sports/ /opt/zip-tools/`

## Tableau

Connect to SQLite DBs via SQLite ODBC driver: http://www.ch-werner.de/sqliteodbc/

Local sync (from VM): `finance/sync_db.sh` pattern — adapt for each sport DB.

## Season Status (as of 2026-02-23)

| Sport | Status | Notes |
|-------|--------|-------|
| NBA | ✅ Active | 2024-25 regular season |
| MLB | ⏳ Preseason | Regular season starts ~March 27 |
| NFL | 🏆 Offseason | Super Bowl Feb 9, 2026. Next season Sep 2026 |
