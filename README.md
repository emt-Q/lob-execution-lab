# LOB Execution Lab — Limit vs Take, Latency & Adverse Selection

A research-grade execution backtest on Binance public L2 data. The project
reconstructs a tick-by-tick limit order book, simulates **FIFO queue
positions with partial fills, cancellations, latency and quote replacement**,
predicts short-horizon mid-price moves from order-flow signals, and compares
four execution strategies across a latency ladder.

> The point of this project is **not** a high Sharpe ratio. It is to answer:
>
> 1. **Under what conditions is posting a limit order better than taking?**
> 2. **How much prediction accuracy is required to offset queueing and
>    latency?**
> 3. **Why does an order-flow signal that looks like alpha die once fill
>    selectivity (the winner's curse) is accounted for?**

## Headline results

At zero latency, the selective/signal strategies already beat immediate
taking: **adaptive_signal** has a 20-tick markout of **+0.20 ticks** and
**−0.009 bps** implementation shortfall, vs **−1.25 ticks / +0.001 bps** for
market take. But the passive edge is fragile — by 100 ms the join-best 20-tick
markout deteriorates from −1.39 to **−2.91 ticks**, *worse than taking*.
Numbers are on the deterministic synthetic stream (see Findings for how they
were produced); live Binance data is a drop-in replacement.

## Quick start

```bash
pip install -r requirements.txt

# full pipeline on deterministic synthetic data (~few minutes):
python run_all.py

# ...or record live Binance L2 data first (30 min), then run on it:
python data/binance_client.py --minutes 30
python run_all --raw data/raw/btcusdt_live.jsonl

# tests
python -m pytest tests/ -q
```

No API key is needed: depth snapshots, diff depth and trades are all public.
If `api.binance.com` geo-blocks you (HTTP 451, e.g. US IPs), the default
endpoint is Binance's public data mirror `data-api.binance.vision` /
`data-stream.binance.vision`, which serves identical data. Override with
`--endpoint {vision,binance.us,binance.com}`.

## What is built

| Requirement | Where |
|---|---|
| Reconstruct tick-by-tick book from Binance public L2 | `core/orderbook.py`, `data/binance_client.py` |
| FIFO queue position, partial fills, cancel, latency, quote replacement | `execution/queue_model.py`, `execution/backtester.py` |
| OFI, queue imbalance, microprice, trade sign, short-run volatility | `features/features.py` |
| Predict 1/5/20-tick mid-price movement | `model/predictor.py` |
| Four execution strategies | `execution/strategies.py` |
| Latency ladder 0/10/50/100 ms | `analysis/experiments.py` |
| Fill prob, time-to-fill, 1/5/20 markout, implementation shortfall, adverse selection, inventory, net PnL after fees, latency sensitivity | `analysis/metrics.py`, `results/summary.csv` |

## Methodology

### 1. Data and book reconstruction

We record three public streams for `BTCUSDT`:

* REST depth snapshot (`/api/v3/depth?limit=1000`) with `lastUpdateId`;
* diff depth stream (`@depth@100ms`): each event carries `U` (first update
  id), `u` (final update id) and changed price levels — a level with qty 0 is
  deleted, otherwise the aggregate qty is replaced;
* trade stream (`@trade`): price, qty and `m` (buyer-is-maker), which gives
  the aggressor sign.

Local reconstruction follows Binance's standard recipe: buffer diffs, fetch
the snapshot, discard diffs with `u <= lastUpdateId`, verify the first diff
satisfies `U <= lastUpdateId+1 <= u`, then enforce `U == previous_u + 1`
forever after (any gap is recorded by `BookReconstructor.gaps`). This yields
an exchange-time ground-truth book.

An offline, deterministic **synthetic LOB generator** (`data/synthetic.py`)
emits the exact same event schema. It is built on three latent variables: a
**fair price** driven by the permanent impact of signed trades plus regime
drift; a persistent **aggressor-pressure** process (AR(1) with sign-flipping
shocks) that makes buy/sell market orders cluster; and a **quoted mid** that
tracks fair with a lag (partial adjustment). The recorded book is always a
valid tight grid around the quoted mid; trades remove top-of-book size,
limit adds/cancels perturb quantities, and only levels leaving the grid are
cancelled. The lag between fair and quotes is what generates both the
short-horizon predictability and the adverse selection of resting orders.
It makes the entire pipeline reproducible without network access; all
results below are produced on it, and live data is a drop-in replacement.

### 2. FIFO queue model

A price level is an ordered list of **blocks** —
`[market qty][our order][market qty]...` — which is exactly price-time
priority:

* **queue position**: aggregate volume ahead of each of our orders;
* **partial fills**: an aggressive trade shrinks blocks front-to-back, so our
  order fills only after every block ahead of it is consumed;
* **cancellations**: aggregate depth decreases not explained by trade prints
  are cancellations; they are attributed **pro-rata across market blocks** —
  the unbiased queue-model assumption (we cannot observe who cancelled;
  "all cancels ahead" is optimistic, "all behind" is pessimistic);
* **late joins**: aggregate qty increases append at the back (FIFO), so they
  never improve our position;
* **quote replacement**: cancelling and re-posting at a new price appends a
  fresh block — queue priority is forfeited.

### 3. Latency semantics

With one-way latency `L` (symmetric for market data and orders):

```
exchange:  event occurs at T
trader:    sees it at T + L          (market-data latency)
           decides on book containing only events with time <= T
order:     reaches exchange at T + 2L (order latency)
fills:     occur at exchange time
```

Decisions therefore use information delayed by `L`, and the order's queue
position is fixed from the book at its *arrival* time. The engine is a single
exchange-time event loop with due order operations interleaved with market
events; market orders and marketable limits sweep the book at arrival (not at
decision) time.

### 4. Signals and prediction

Per market event we compute:

* **queue imbalance** `QI = (Qb - Qa)/(Qb + Qa)` at top of book;
* **microprice** `(Pb·Qa + Pa·Qb)/(Qb + Qa)`, used as its deviation from
  mid in ticks;
* **order-flow imbalance (OFI)**, Cont–Kukanov–Stoikov (2013), window sum
  normalized by mean top-of-book depth;
* **trade sign flow** — aggressor-signed traded volume (and count) over a
  window;
* **short-run realized volatility** of mid returns, in tick units;
* spread in ticks.

Labels are the signed future mid move at horizons **h = 1, 5, 20 events**
(continuous, in ticks) and a direction class (down / flat / up, with a small
deadband). Models are fit **walk-forward** (train on the first half, test on
the second): multinomial logistic regression for direction plus ridge
regression for expected signed magnitude `E[Δmid]`. No future information
crosses a decision point.

### 5. Strategies

1. **market_take** — cross immediately; taker fee; no queue risk, full spread
   paid.
2. **join_best** — post at best bid/ask; maintain price priority (replace if
   the best on our side moves through us); market-take the residual on child
   timeout or parent deadline.
3. **improve_one_tick** — post one tick better. When the spread is one tick
   the order is marketable and degrades to taking; when the spread is wider it
   becomes the new best with **zero volume ahead**, jumping the queue as a
   maker.
4. **adaptive_signal** — combines the predicted move (ticks) with an
   inventory schedule: strong adverse predicted move or being behind schedule
   ⇒ take; mild signal ⇒ improve; favorable signal ⇒ join; resting quotes are
   pulled if the signal turns against them.

Parent orders (0.5 BTC, 200-event deadline, 0.1 BTC children) are started at
paired random times in the test half and run for every strategy × latency,
buy and sell sides alternating. Execution quantities are clamped at exchange
arrival so decision races can never overfill a parent.

### 6. Metrics

* **fill probability** — fraction of resting children that (partially) filled;
* **time-to-fill** — decision to first maker fill;
* **1/5/20-tick markouts** — signed mid move after a fill (positive =
  favorable); **adverse selection = −markout**;
* **implementation shortfall** — signed fill VWAP vs the parent's arrival
  mid, in bps;
* **inventory** — mean/max deviation from the parent's schedule;
* **net PnL after fees** — negative shortfall net of maker/taker fees;
* **latency sensitivity** — every metric on the 0/10/50/100 ms ladder.

## Findings

All numbers below come from the deterministic synthetic stream
(`SYNTH_SEED=7`, 90k depth events / 33k trades); 13 unit/integration tests
pass and every strategy × latency run uses the *same* 200-parent schedule, so
comparisons are paired. Markouts are side-adjusted (positive = favorable);
implementation shortfall (IS) is fill VWAP vs the parent's arrival mid.

### Zero-latency scoreboard

| strategy | resting fill prob | median TTF (ms) | mk1 | mk5 | mk20 | IS (bps) | net PnL (bps) |
|---|---|---|---|---|---|---|---|
| market_take      | — | — | −0.74 | −0.88 | −1.25 | +0.001 | −0.001 |
| join_best        | 0.049 | 502 | −0.36 | −0.67 | −1.39 | −0.003 | +0.003 |
| improve_one_tick | 0.032 | 0   | −0.52 | −0.52 | −0.29 | −0.008 | +0.008 |
| adaptive_signal  | 0.053 | 477 | −0.46 | −0.33 | **+0.20** | **−0.009** | **+0.009** |

The signal itself is modest and realistic: walk-forward expected-move vs
realized correlations are **0.31 / 0.41 / 0.37** at h = 1/5/20, with
directional accuracy **0.56 / 0.52 / 0.47**.

### Q1. When is posting better than taking?

Posting earns the half-spread but only fills in selected states, so it wins
conditionally, not on average:

* **Join best** beats taking at the 1-tick horizon (mk1 −0.36 vs −0.74) and on
  IS (−0.003 vs +0.001 bps) — when it fills, it fills cheaply. Its advantage
  is largest when **queue imbalance is high** (posting advantage +0.012 bps in
  the top QI third vs **−0.004 bps in the low-QI third**): a thick book on our
  side means the level is unlikely to be swept against us.
* Naive joining **loses in low-QI and low-OFI states** — the same resting
  order that is cheap to fill is the one most likely to be adversely selected.
* The **adaptive** strategy's advantage over taking grows with realized
  volatility (+0.002 bps in low RV vs **+0.018 bps in high RV**): a signal is
  worth more when prices move, because it decides *when not to post*.
* **Improve-one-tick** only posts as a maker when the spread is ≥ 2 ticks
  (a 1-tick spread cannot be improved inside; the order is then marketable and
  degrades to taking). In those wider, calmer states it jumps an empty queue
  and delivers the best h20 passive markout (−0.29).

**Bottom line:** post when the book is balanced/thick on your side, volatility
is moderate, and the signal does not point against you; take otherwise.

### Q2. How much accuracy offsets queueing and latency?

From the signal-strength sweep of the adaptive strategy:

* At **0–10 ms**, even the uninformative baseline (≈0.52 directional accuracy)
  has non-positive IS — queueing alone does not cost you if you are fast and
  follow an inventory schedule.
* At **100 ms**, IS crosses from favorable to costly between 0.57 and 0.61
  directional accuracy at the 5-tick horizon — i.e. you need roughly
  **≈60% directional accuracy to overcome 100 ms** of round-trip delay.
* The marginal value of accuracy is steep near the latency edge but flat once
  you are fast: going from 0.52 → 0.68 accuracy improves 0 ms IS only modestly,
  while the same move at 100 ms is the difference between profit and loss.
  (At `signal_mix=1` the classifier becomes overconfident and accuracy falls
  back — an overfit signal is worse than a calibrated weaker one.)

**Bottom line:** prediction accuracy and latency are substitutes on a steep
curve — shave latency first; buy prediction only to clear the breakeven at the
latency you cannot avoid.

### Q3. Why does apparent alpha die after fill selectivity?

`results/alpha_decay.csv` measures the same signal two ways (expected
favorable 5-tick edge, in ticks):

| signal bucket (expected favor) | apparent edge *if traded at mid* | fill-conditioned edge | # fills |
|---|---|---|---|
| Q1 weak/against | 0.11 | 0.39 | 102 |
| Q2 | 0.28 | 0.60 | 26 |
| Q3 | 0.51 | 0.34 | 28 |
| Q4 | 0.75 | 0.46 | 27 |
| Q5 strong favor | **1.20** | **0.45** | 20 |

* If you could trade the favored side **at the mid**, the edge is real and
  well-calibrated, rising monotonically to 1.2 ticks.
* You cannot. A resting order is a **free option** that counterparties exercise
  against you: fills concentrate in the *weak/against* bucket (102 fills) where
  aggressors hit you, while the *strong-favor* bucket — where price is about to
  move your way — barely fills at all (20), because price runs away from your
  resting order.
* Conditioning on a fill therefore inverts the population: the monotonic
  0.11 → 1.20 alpha collapses to a **flat ≈0.34–0.60 ticks with no relation to
  signal strength**. After the ~1 bps fee and the residual adverse drift, that
  apparent edge is not capturable as a passive maker.

**Bottom line:** "the signal predicts the mid" and "the signal predicts the mid
*conditional on my resting order being filled*" are different statements. The
first is alpha at the mid; the second is what a maker actually earns, and the
selection mechanism (winner's curse) removes exactly the states that carried
the alpha.

### Latency sensitivity

20-tick markout across the ladder:

| strategy | 0 ms | 10 ms | 50 ms | 100 ms |
|---|---|---|---|---|
| market_take | −1.25 | −1.29 | −1.44 | −1.47 |
| join_best | −1.39 | −1.42 | −2.00 | **−2.91** |
| improve_one_tick | −0.29 | −0.44 | −0.95 | −2.03 |
| adaptive_signal | +0.20 | +0.09 | −0.45 | −1.08 |

Taker costs are nearly latency-invariant (the half-spread is paid up front);
passive costs degrade roughly monotonically because a delayed maker posts at a
stale price and is picked off before it can re-quote. The adaptive strategy
degrades but stays at or near the best at every latency — the value of acting
on a signal is precisely in not resting through adverse states.

## Repository layout

```
config.py                 instrument, fees, endpoints, horizons, backtest cfg
data/binance_client.py    live snapshot + WS recorder (public data)
data/synthetic.py         deterministic LOB event generator, same schema
data/loader.py            JSONL -> typed, time-ordered events
core/orderbook.py         L2 book + Binance reconstruction validator
features/features.py      OFI, QI, microprice, trade flow, vol, labels
model/predictor.py        walk-forward logistic + ridge models
execution/queue_model.py  FIFO blocks, partial fills, pro-rata cancels
execution/strategies.py   the four strategies
execution/backtester.py   event-driven engine with latency semantics
analysis/metrics.py       markouts, shortfall, adverse selection, PnL
analysis/experiments.py   sweeps, condition/alpha analyses, figures
scripts/                  numbered one-step pipeline scripts
results/                  metrics CSVs + per-fill/per-parent ledgers
reports/figures/          six diagnostic figures
tests/                    unit + integration tests
```

## Assumptions and caveats

* **Frozen-book backtest**: our orders do not alter the recorded market;
  queue inference uses aggregate levels plus the stated pro-rata cancellation
  rule. A production matcher needs order-level (L3) data.
* Latency is modeled as a fixed symmetric one-way delay; real latency is
  jittery and venue/co-location dependent.
* Fees use Binance spot VIP-0 rates (maker = taker = 1 bps default); change
  them in `config.py`. Slippage from walking multiple levels is modeled from
  the reconstructed depth.
* Signals are deliberately simple linear models; the goal is execution
  mechanics and their interaction with selectivity, not alpha maximization.
