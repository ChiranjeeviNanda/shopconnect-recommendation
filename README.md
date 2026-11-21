# ShopConnect Meal-Plan Recommendation Engine

This is a PostgreSQL-backed meal recommendation project for households. It generates a synthetic 90-day meal-planning dataset, ingests it into PostgreSQL, ranks compatible recipes for a household and meal slot, and optionally generates short Gemini explanations for the resulting ranked list.

> **Architecture Note:** The ranking engine is a deterministic, content-and-history-based recommender rather than a trained machine-learning model. Gemini is used only to explain an already-ranked result; it does not select or reorder recommendations.

---

## Repository Layout

```text
shopconnect-recommendation/
├── data/
│   ├── raw/                  # Input recipe catalog (recipe_list.csv)
│   └── processed/            # Generated CSVs (households, recipes, plans, interactions, orders)
├── db/
│   ├── schema.sql            # PostgreSQL table definitions
│   └── views.sql             # Analytical SQL views
├── logs/                     # Pipeline execution logs
├── notebooks/
│   └── sql_analysis.ipynb    # SQL and data exploration notebook
├── src/
│   ├── simulation/           # Synthetic behavior, scoring, and order generation modules
│   ├── run_pipeline.py       # End-to-end setup workflow
│   ├── analyze_recipes.py    # Raw catalog audit and cleanup
│   ├── generate_synthetic_data.py
│   ├── household_generator.py
│   ├── ingest_data.py        # Database schema load and CSV ingestion
│   ├── setup_views.py        # Database view creation
│   ├── recommendation_engine.py
│   ├── ai_explanations.py    # Optional Gemini explanation generator
│   ├── config.py             # Environment and database settings
│   └── logging_config.py     # Logging setup and formatting
├── tests/
│   ├── evaluate_engine.py    # Offline chronological benchmark
│   └── test_*.py             # Unit and integration tests
├── .env                      # Local configuration (optional)
├── requirements.txt
├── PRODUCTION_OPERATIONS.md  # Proposed production architecture
└── README.md
```

---

## Setup

### 1. Prerequisites

Ensure the following are installed and running before starting:

* **Python:** 3.10 or newer
* **PostgreSQL:** Active server instance with a target database (defaults to `shopconnect`)

### 2. Environment & Dependencies

Create and activate a virtual environment, then install the required packages:

**Windows (PowerShell):**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** If PowerShell blocks script execution, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in your current terminal and retry activating the environment.

**macOS / Linux (Bash/Zsh):**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Database Configuration

Create the PostgreSQL database if it does not already exist:

```bash
createdb shopconnect
```

By default, the application connects using `postgresql+psycopg2://postgres:postgres@localhost:5432/shopconnect`. To override these defaults, set the corresponding environment variables in your terminal or a project-root `.env` file:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DB_USER` | `postgres` | PostgreSQL username |
| `DB_PASSWORD` | `postgres` | PostgreSQL password |
| `DB_HOST` | `localhost` | Database host address |
| `DB_PORT` | `5432` | Database port |
| `DB_NAME` | `shopconnect` | Target database name |

**Windows (PowerShell):**

```powershell
$env:DB_USER = "postgres"
$env:DB_PASSWORD = "postgres"
$env:DB_HOST = "localhost"
$env:DB_PORT = "5432"
$env:DB_NAME = "shopconnect"
```

**macOS / Linux (Bash/Zsh):**

```bash
export DB_USER="postgres"
export DB_PASSWORD="postgres"
export DB_HOST="localhost"
export DB_PORT="5432"
export DB_NAME="shopconnect"
```

### 4. Gemini API Configuration *(Optional)*

A Gemini API key is **not** required to run the pipeline or generate rankings. If omitted, `src/ai_explanations.py` automatically falls back to grounded, rule-based explanations.

To enable AI-generated explanations, add `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) to a `.env` file in the project root:

```env
GEMINI_API_KEY=your-api-key-here
```

### 5. Run the Data Pipeline

Run the orchestration script from the **repository root** so relative paths (`data/`, `db/`, `logs/`) resolve properly:

```bash
python src/run_pipeline.py
```

This executes four sequential steps (halting immediately if any step fails):

1. **`src/analyze_recipes.py`** — Audits and analyzes the raw recipe catalog.
2. **`src/generate_synthetic_data.py`** — Generates deterministic synthetic datasets (seed `42`) in `data/processed/`.
3. **`src/ingest_data.py`** — Applies `db/schema.sql`, truncates existing project tables, and loads the processed CSVs.
4. **`src/setup_views.py`** — Creates or refreshes analytical views defined in `db/views.sql`.

#### Customizing Pipeline Execution

You can skip specific stages using CLI flags (run `python src/run_pipeline.py --help` for full details):

* `--skip-analysis`
* `--skip-generation`
* `--skip-ingestion`
* `--skip-views`

*Example: Reload the database and views using existing CSVs without regenerating synthetic data:*

```bash
python src/run_pipeline.py --skip-analysis --skip-generation
```

> *Alternatively, any of the four scripts listed above can be executed standalone in order.*

### 6. Verify Setup with Tests

Once data ingestion completes, run the test suite to verify database constraints and recommendation logic:

```bash
python -m pytest -q
```

To include live Gemini API integration test (requires a valid API key):

```bash
python -m pytest -q --live-gemini
```

---

## Synthetic Data Rules

The pipeline ingests the 1,000-row raw recipe catalog (`data/raw/recipe_list.csv`), cleans and filters it into 984 valid recipes, and generates deterministic synthetic datasets (seed `42`) under `data/processed/`.

### Households (`households.csv` & `food_dislikes.csv`)

* **Scale & IDs:** Generates 300 households (`H001`–`H300`) with a fixed creation date of `2026-01-01`.
* **Demographics:** Samples household size uniformly from `1` to `5` members.
* **Regional distribution:** 55% South India, 20% North India, 20% West India, and 5% East India.
* **Dietary split:** 60% Vegetarian, 25% Non-Vegetarian, 10% Vegan, and 5% Jain.
* **Preferences & goals:** Assigns spice tolerance (`mild`, `medium`, `spicy`) and health goals (`everyday`, `high-protein`, `weight-loss`).
* **Ingredient dislikes:** Roughly 40% of households receive 1–2 dislikes sampled from `mushroom`, `brinjal`, `capsicum`, `paneer`, `seafood`, `gourd`, and `jackfruit`.

### Meal-plan simulation (`meal_plans.csv` & `meal_plan_items.csv`)

* **Window & cadence:** Simulates 90 days starting `2026-09-01`, generating `breakfast`, `lunch`, and `dinner` slots in 7-day blocks across active household days.
* **Cold-start cohort:** Designates ~18% of households as cold-start users, staggering their join dates between day 35 and day 86 and capping each at 4 total interactions.
* **Engagement & affinity:** Assigns each household a randomized engagement rate between `0.55` and `1.00`. Candidate sampling boosts recipes matching the household's home region (`1.35x`), health goal (`1.15x`), and spice level (`1.10x`).
* **Repeat cooldowns:** Applies stronger or moderate cooldown penalties to recipes planned within the last 3, 7, or 14 days to enforce menu variety.

### Interaction behavior (`interactions.csv` & `orders.csv`)

Each planned meal item generates one of five interaction outcomes:

* **Positive signals:** `cooked`, `accepted`
* **Negative / friction signals:** `swapped`, `skipped`, `not_cooked`

Swap and skip probabilities vary by meal role and household engagement, with repeat dishes carrying a higher swap likelihood. When a swap triggers, the simulator searches for a replacement matching the exact meal role first—falling back to compatible slot roles—while strictly enforcing diet and dislike rules. If no valid substitute exists, the interaction defaults to `accepted`.

> **Supplied Snapshot Totals:** 300 households · 984 active recipes · 172 dislike entries · 3,456 meal plans · 72,090 plan items · 67,374 interactions · 2,460 orders.

### Popularity score (`recipes.popularity_score`)

`popularity_score` is a `0`–`10` normalized metric derived from historical interaction weights:

| Interaction Type | Weight |
| :--- | ---: |
| `cooked` | `+2.0` |
| `swapped` (replacement target `swapped_to_recipe_code`) | `+1.5` |
| `accepted` | `+1.0` |
| `swapped` (original rejected recipe) | `-0.5` |
| `not_cooked` | `-0.5` |
| `skipped` | `-1.0` |

> **Data Leakage Caveat:** The popularity baselines recompute scores using only interactions and meal-plan exposures before the evaluation cutoff, with additive rate smoothing (constant = `5`). The personalized engine uses the `popularity_score` stored in `recipes.csv`; in the supplied generated data this score was calculated from the full simulation window, including the test period. Its offline results therefore have potential future-data leakage and are not directly comparable as a leakage-free temporal benchmark.

---

## Key insights

### 1. Main dishes and anchors carry the volume, with similar cook rates

Main dishes have 24,212 interactions and a 54.1% cook rate. Anchors have 20,919 interactions and a 55.8% cook rate. These two roles make up the core of meal-plan activity.

### 2. Snacks are the clearest weak spot

Snacks have the lowest cook rate (47.4%) and the highest swap (17.5%) and skip (15.5%) rates. Snack selection is a clear improvement area compared to condiments or beverages.

### 3. Supporting roles perform consistently well

Condiments (58.9%) and beverages (58.5%) have the highest cook rates, with low swap and skip rates. Households tend to keep smaller meal items once added to a plan.

![All Action Shares of Total Interactions](assets/action_shares_by_meal_role.png)

### 4. Swap pairs show close substitutes and clear winners

Top swap pairs are mostly close variants within the same role, such as Kakdi Salad to Kheera Salad (18 swaps) and Kakdi Raita to Kheera Raita (16 swaps). Green Gram Sundal and Mint Rice have the highest net swap balance (+17 each), while Spinach Mash and Arbi Fry have the lowest (-13 each).

![Top Swap Pairs and Net Swap Balance](assets/swap_pairs_and_balance.png)

### 5. Weight-loss plans convert best

Weight-loss plans reach an 83.70% plan-to-order conversion rate, compared with 70.45% for high-protein plans and 58.98% for everyday plans. This was surprising because everyday plans felt like the default routine choice, yet weight-loss plans show much stronger order follow-through. Template goal is a strong signal of grocery order intent.

![Grocery Order Conversion Rate by Template Goal](assets/conversion_rate_by_goal.png)

### 6. Repeats are common, but repeat intervals vary widely

Across 290 households, the median average repeat interval is 29.8 days, and the mean is 27.1 days. Because values span from 4.0 days to 38.7 days, a single fixed repeat window does not fit every household.

![Distribution of Average Household Repeat-Dish Intervals](assets/repeat_dish_intervals.png)

---

## Model Choice and Ranking Logic

### Why a deterministic rule-based engine?

The recommendation engine uses a deterministic, content-and-history-based scoring engine rather than a trained machine-learning model. On a compact synthetic dataset with strict dietary constraints, an explicit scoring model requires zero training overhead, avoids black-box failure modes, and keeps every recommendation and constraint filter fully auditable.

### Candidate filtering & scoring pipeline

Given a `(household, date, slot, meal_role)` query, the engine executes six steps:

1. **Profile lookup:** Loads household attributes, health goals, and ingredient dislikes.
2. **Repeat-window preference:** Excludes recipes planned in the past **7 days** to match the weekly meal-plan cadence. If this leaves no candidates, it restores the candidates, so repeats can occur when alternatives are unavailable.
3. **Catalog hygiene:** Filters to active, published, deduplicated catalog entries.
4. **Candidate filtering:** Applies diet and dislike filters. It prefers compatible meal slots and exact requested roles, but falls back to broader slot or role candidates when no compatible recipes are available.
5. **Cold-start detection:** Flags households with `< 5` prior interactions as cold-start.
6. **Multi-signal scoring:** Ranks surviving candidates using weighted profile, history, rotation, and popularity multipliers:

| Signal Category | Positive Boosts | Penalties |
| :--- | :--- | :--- |
| **Profile & Role** | Exact meal role match, home region match, spice tolerance match, high-protein or quick-prep (`weight-loss`) alignment | — |
| **Household History** | Prior `accepted`/`cooked` recipes; recent affinity for matching dish families and regions | Prior `skipped`, `not_cooked`, or negative recipe history |
| **Recency & Rotation** | Time-decayed boost for warm households (stronger within 10 days, tapering over 30 days) | Cooldown penalty for recently eaten dishes |
| **Global Popularity** | Applied as a bounded tie-breaking multiplier: `1 + (popularity_score / 10) * 0.05` | — |

### Evaluation baselines

Offline benchmarks compare the personalized engine against two deterministic baselines using the same diet/dislike filters and slot/role fallbacks. All systems exclude recent repeats when alternatives exist, but allow them if repeat exclusion empties the candidate pool:

* **Global popularity:** Ranks valid candidates strictly by pre-cutoff `popularity_score`.
* **Region + popularity:** Applies a `1.15x` boost to `popularity_score` when a recipe matches the household's home region.

### Gemini explanations (`src/ai_explanations.py`)

When `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is configured, the engine calls `gemini-3.8-flash` to generate concise, human-readable explanations for the top-ranked dishes. If no API key is set or a request fails, it falls back to deterministic, template-based explanations.

---

## Evaluation

### Methodology & Metrics

Evaluation uses a chronological split defaulting to a 75-day training window (`test_start = 2026-11-15`). The popularity baselines compute scores strictly from pre-cutoff history; the personalized engine currently uses the stored full-window popularity score, as noted above. Ground-truth targets are defined as:

* **`accepted` / `cooked` events:** The planned recipe itself.
* **`swapped` events:** The replacement recipe (`swapped_to_recipe_code`).

Performance is measured across the top $K=5$ recommendations using three metrics:

* **`Hits@5`:** Count of test events where the ground-truth recipe appears in the top 5.
* **`Recall@5`:** `Hits@5 / Total Events` (equivalent to `HitRate@5` for single-target events).
* **`NDCG@5`:** Position-aware ranking quality, computed as $\frac{1}{\log_2(\text{rank} + 2)}$ (0-indexed) when the target is in the top 5, and `0` otherwise.

Run the benchmark after database ingestion:

```bash
python -m tests.evaluate_engine
```

Optional CLI flags: `--test-start YYYY-MM-DD`, `--train-days N`, `--limit N`, `--k N`, and `--explain` (run `python -m tests.evaluate_engine --help` for full details).

### Recorded benchmark results (9,731 test events)

Evaluated from `2026-11-15` onward (seed `42`):

| Model | Hits@5 | Recall@5 | NDCG@5 | Recall Lift vs. Global |
| :--- | ---: | ---: | ---: | ---: |
| Global popularity | 977 / 9,731 | 0.1004 | 0.0621 | — |
| Region + popularity | 1,036 / 9,731 | 0.1065 | 0.0671 | +6.08% |
| **Personalized engine** | **1,013 / 9,731** | **0.1041** | **0.0656** | **+3.68%** (+5.58% NDCG@5) |

#### Cohort breakdown

| Cohort | Events ($N$) | Model | Hits@5 | Recall@5 | NDCG@5 |
| :--- | ---: | :--- | ---: | ---: | ---: |
| **Swap-only** | 1,330 | Global popularity | 150 / 1,330 | 0.1128 | 0.0699 |
| | | Region + popularity | 156 / 1,330 | 0.1173 | 0.0774 |
| | | **Personalized engine** | **166 / 1,330** | **0.1248** | **0.0809** |
| **Accepted / cooked** | 8,401 | Global popularity | 827 / 8,401 | 0.0984 | 0.0609 |
| | | Region + popularity | 880 / 8,401 | 0.1047 | 0.0654 |
| | | **Personalized engine** | **847 / 8,401** | **0.1008** | **0.0632** |
| **Cold-start** (`< 5` prior) | 53 | Global popularity | 1 / 53 | 0.0189 | 0.0094 |
| | | Region + popularity | 1 / 53 | 0.0189 | 0.0094 |
| | | **Personalized engine** | **3 / 53** | **0.0566** | **0.0286** |
| **Warm** (`>= 5` prior) | 9,678 | Global popularity | 976 / 9,678 | 0.1008 | 0.0624 |
| | | Region + popularity | 1,035 / 9,678 | 0.1069 | 0.0674 |
| | | **Personalized engine** | **1,010 / 9,678** | **0.1044** | **0.0658** |

### Observations from the recorded results

* **Swap events:** The personalized engine's recorded Recall@5 is `0.1248`, compared with `0.1173` for region + popularity. Because of the personalized popularity leakage described above, this should be treated as an observation rather than a reliable model comparison.
* **Cold-start cohort:** The personalized engine records 3 hits among 53 events, compared with 1 for each baseline. This cohort is small and not enough to establish cold-start performance.
* **Synthetic behavior:** The simulation favors home-region recipes, so the region baseline's overall result may reflect the generator's own assumptions rather than general customer preferences.

---

## Further Reading

* See [the SQL analysis notebook](./notebooks/sql_analysis.ipynb) for detailed analysis of generated data using the database views.
* See [Production Operations](./PRODUCTION_OPERATIONS.md) for a proposed production approach.

---

## Known Limitations

* **Synthetic data and uncalibrated weights:** Profiles, interactions, swaps, and orders come from fixed rules and probabilities, not real users. Ranking multipliers and the `< 5` interaction cold-start threshold use fixed values, not weights learned from real data.
* **Popularity score leakage:** The personalized engine reads `popularity_score` from `recipes.csv` (computed over the full simulation), while baselines use pre-cutoff data only. This makes the personalized benchmark non-leakage-free.
* **Narrow offline evaluation:** Each test event has one target, ignores `skipped` and `not_cooked` outcomes, includes correlated household events, and has a small cold-start sample (`53` events) with no confidence intervals or multi-seed runs.
* **Soft exclusions and approximate filtering:** Slot, meal-role, and 7-day repeat filters fall back to broader pools or repeats when no matches remain, so exclusions are not hard guarantees. Diet checks rely on catalog tags, and dislike checks use substring matching on names, ingredient JSON, and dish families, which can miss aliases or cause false matches.
* **Catalog quality dependence:** Only active, published, deduplicated recipes stay in the catalog, and missing region or diet values get defaults. Bad metadata can drop or misrank recipes.