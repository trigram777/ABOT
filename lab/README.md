# `lab` — a general options backtesting engine

The old `engine/` package knew what an iron condor was: it had a `Condor`, a
`Strangle`, and six named pieces, and every question had to be phrased in those
terms. That was right when there was one trade. It is wrong for a research
programme that sweeps naked longs, verticals up to three strikes wide, ITM short
spreads, butterflies, and long structures that get COVERED or SHORTED into
spreads halfway through a session.

So this package has **four verbs and one noun**: `submit`, `close`,
`close_all`, `settle`, and a `Position`. A condor is four legs in one call. A
COVER is one more call. The engine is never told what either is called.

`engine/` is left untouched so the closed nine-stage programme stays
reproducible.

## Layers

| module | owns |
|---|---|
| `session.py` | one 0DTE expiry as dense `[minute, strike]` arrays, the SPX spot path, the settlement close, and strike selection by price / delta / offset |
| `fills.py` | what a price is: the mid↔cross bracket, tick concession |
| `broker.py` | positions, atomic orders, commissions, settlement, ledger |
| `indicators.py` | the dual Bollinger metric set, computed once for the whole decade |
| `exits.py` | W, L, COVER and SHORT — the exit conventions |
| `runner.py` | the method protocol and the session runner; one row per trade |
| `select.py` | strike selection — naked longs, straddles, credit verticals, condors |
| `examples.py` | worked example methods, as signal generators |
| `sweep.py` | many cells, one pass over the data |
| `metrics.py` | Sharpe, max drawdown, and tail dependence |
| `validate.py` | walk-forward splits and the Deflated Sharpe Ratio |
| `search.py` | the unified Bayesian sweep, over Optuna TPE |
| `scan.py` | which indicators have conditional signal, before spending trials |
| `study.py` | the streamlined workflow end to end, as a command |
| `regimes.py` | which days were ordinary and which were history — the reporting split |
| `gates.py` | indicator conditions for entries and exits |
| `demo.py` | four structures on one real session, none known to the engine |
| `demo_abot.py` | the indicator layer wired to the broker, one session or all 1,894 |

## Measured facts this encodes

**A minute's quote is the book at the START of that minute.** Measured on
liquid near-the-money strikes across 2022 / 2024 / 2026 sessions:

| | 2022-06-15 | 2024-01-03 | 2026-06-15 |
|---|---|---|---|
| mean \|quote_mid[m] − bar_open[m]\| | $0.52 | $0.07 | $0.07 |
| mean \|quote_mid[m] − bar_close[m]\| | $1.31 | $0.31 | $0.41 |

The quote tracks its own bar's **open** (and the previous bar's close) about ten
times better than its own close. So a rule deciding at `m:00` that reads
`bid[m]` and fills against it is reading the book as it stood when the decision
was made. **No shift is applied, and none is needed.** The same convention names
`spot_open[m]` as the decision price and `spot_close[m]` explicitly as
lookahead, so using the latter is a choice rather than an accident.

**09:30:00 is not a book.** The option file reports no underlying at all on
essentially every session — it is the pre-rotation snapshot. `book_reported[m]`
is the gate, and it is deliberately *not* derived from the spot series, because
IBKR does print a 09:30 bar and therefore cannot tell you the option book was
not there yet. Earliest honest entry is **09:31**.

**Commission is charged per LEG, and expiry is free.** Both from the broker's
own prints (COSTS.md), reproduced exactly by `leg_commission` and pinned by
tests against all six observed figures:

```
per leg:  max($1.00, $0.65 × qty)  +  qty × third-party
third-party:  XSP $0.22   SPXW $0.54 (premium < $1) / $0.63 (≥ $1)
```

SPXW at one contract costs **$1.63/ct**; at two or more, **$1.28/ct** and flat
thereafter. One contract is the worst size to trade. Leg count is chosen once at
design time and paid on every trade — the demo's 40-wide condor collects $0.95
of gross credit on 2024-01-03 and pays $6.16 in fees for it.

## The three invariants

1. **Cash is signed, money in positive.** Session P&L is `sum(every cash flow)
   + settlement`, with no basis arithmetic in the total. Basis is tracked only
   because W and L triggers are quoted as multiples of the entry price.
2. **An order is atomic.** A four-leg package either goes on or it does not.
   Three filled legs and a refused fourth is a position no broker would have
   given, and the naked leg it leaves would dominate every statistic it entered.
   `close_all` refuses as a whole too: a rule that cannot exit one leg has not
   exited.
3. **A partial total is never shown.** `mark()` returns `None` if any held leg
   has no price. A partial sum on a short structure reads as a smaller loss than
   reality.

## The fill bracket

One knob, `edge ∈ [0, 1]`: `price = mid + side · edge · (ask−bid)/2`. `MID` and
`CROSS` are the two ends and **every result is reported against both**, because
in the previous programme that assumption was worth $23–45 a trade — more than
any parameter it measured in nine stages. A strategy that survives only at the
mid is an execution claim, not a strategy claim.

Two rules keep `edge` honest:

- **Tick concession applies only to prices we construct.** A mid is snapped to
  the $0.05/$0.10 grid, always against ourselves, because an off-grid order is
  rejected outright (IBKR Error 110). A price we merely *take* is already one
  someone is showing — snapping a $3.05 ask up to $3.10 would charge us for a
  rule that governs orders, not quotes.
- **Conceding never concedes past the book.** A $2.95/$3.05 quote at half-edge
  snaps to $3.10, worse than simply crossing. Clamped, so `edge` stays monotone.

A fill needs a **two-sided** quote. A one-sided contract can in reality be
lifted, but allowing it would make the two brackets disagree about *which trades
happen* as well as at what price, and the bracket is only interpretable when
both ends see the same opportunity set. `allow_one_sided_buy` exists to answer
that later; it is off, and every refusal is recorded rather than skipped.

## Quote hygiene, applied once at load

A zero or absent bid is not a price, and NaN is worse than zero because `NaN > 0`
and `NaN <= 0` are *both* False — only the negated comparison catches it. A
crossed book (`ask < bid`) is bad data, not an arbitrage. All three collapse to
NaN, and every reader treats NaN as "no price" and refuses rather than inventing
one. The failure this prevents is a 0.00/8.00 quote whose $4.00 midpoint is a
credit no counterparty ever offered.

## Strike selection

`by_price` reports the **miss** in dollars and never silently substitutes:
whether a $0.50 bucket that could only be filled at $0.80 belongs in a study is
a question for the analysis, not the engine, and a rule that substituted quietly
would make the two indistinguishable afterwards. `target_met` is `None` unless a
tolerance was supplied — the miss is reported, not judged.

`step(right, strike, n)` walks `n` strikes further **out of the money**, with
the direction taken from the right. A COVER ("sell the next OTM strike against
it") is `step(+1)` and a SHORT ("sell a closer-to-ATM strike") is `step(-1)` on
either side, with no `if right == CALL` at the call site.

## Bars

`session.bars(tf)` for `tf ∈ {1, 5, 15, 30, 60}` returns **one continuous RTH
series** 2017→2026, not per-session frames. A 20-period band at 60m needs three
sessions of history and cannot exist inside one day, so sessions are a column,
not a boundary.

Note the 60m series opens each day with a **30-minute stub** (09:30–10:00) before
the hourly grid resumes — an artefact of RTH starting on the half hour, and
something a bar-indexed rule has to expect.

## Running it

```bash
pytest                    # 293 pass, 12 skip; ~3 s once the sample data exists
pytest lab/test_broker.py -q
```

The first run builds a synthetic sample dataset (see the top-level README).
The 12 skips assert facts about the production dataset — its size, a session
named by date, a base rate of the real market — and run when `LAB_DATA_ROOT`
points at one.

## Running sweeps — two rules learned the hard way

**Use `spawn`, never `fork`.** Polars starts a Rayon thread pool on first use,
and `fork()` after that leaves the child holding a pool whose worker threads do
not exist. The symptom is not an error: it is N worker processes sitting in
`futex_do_wait` at a load average of 0.1, forever, producing no output. Any
parent that has touched Polars — reading the calendar is enough — must set the
start method before building the pool:

```python
import multiprocessing as mp
mp.set_start_method("spawn", force=True)      # BEFORE ProcessPoolExecutor
```

**Load each session once and run every parameter cell against it.** A warm load
is ~66 ms (131 ms averaged including each worker's one-time spot-file build), so
the whole calendar loads in **12 seconds on 24 workers**. Re-loading per cell
would make the load dominate a 100-cell sweep by two orders of magnitude.

## Dataset health, measured across all 1,894 eligible sessions

Every session loads and trades: **1,894 ok, 0 errors, 0 order rejections**, all
391 minutes long, settlement present on every one.

| | |
|---|---|
| strikes per session | 16 (2017 min) / 46 (median) / 222 (max) |
| first live option minute | 09:31 on 1,890 sessions; 09:30 on 3 (2019-04-24, 06-17, 07-01); **09:41 on 2019-06-14** |
| a $2.00 call findable at 10:00 | 1,893 / 1,894, median miss $0.20, 23 sessions miss by > $1 |
| opening VIX missing | 1 session (2021-05-14) |

**The corridor does not always reach $0.50.** At 10:00 the cheapest listed call
costs more than $0.50 on 8 sessions, and the misses are concentrated on
high-volatility days where the VIX-scaled corridor is wide in *points* but stops
short in *price* — 2022-02-22's outermost call still costs $13.40 at 10:00. The
dataset rebuild extending strikes to $0.50 at all times removes this; until it
lands, `by_price(...).miss` is how a study sees it.

Never assume the first tradeable minute is 09:31 — one session in the sample
does not open its book until **09:41**. Gate on `book_reported`, not on a
constant.

## Indicators

`indicators.build(timeframe, BandConfig())` returns one wide frame — one row per
bar, the whole 2017→2026 continuous series — carrying every first- and
second-order metric in the specification. It is **~40 ms for the decade at 30m**
and cached per (timeframe, config), so a sweep that holds the bands fixed pays
for them once rather than once per session. `with_third_order()` adds the
specification's third order as a plain lag of the whole metric set.

### The lookahead rule, and what it means precisely

*"Bollinger metrics must only use the opening Bollinger print per bar, not where
it drifted to by the bar's close."*

Implemented as: **every band at bar `t` is computed from bars `t−n … t−1`** —
fully closed bars only — and the metrics compare bar `t`'s OPEN against those
bands. A band therefore has exactly one value per bar and cannot drift within
it, which is the property the rule asks for. The bar's open is the *only* value
from bar `t` that any metric touches, because it is the only one known when a
decision at `t` is taken.

The alternative reading — band includes the current bar, with the open standing
in for the running price, the way a live chart paints it — is also
lookahead-safe and differs only in whether the newest price participates in its
own band. It is **not** what is implemented. Setting `source="open"` gets within
one bar of it.

**Two lookahead tests, because one is not enough.** Truncating the series and
demanding bit-identical surviving rows catches windows that peek *forward*. It
does **not** catch a band that reads its own bar — deleting the future does not
change bar `t`'s own close. So there is a second test that perturbs one bar's
high/low/close while leaving its open alone and demands nothing on that bar
moves. Both were mutation-checked: dropping the band `.shift(1)`, dropping the
green/red-average shift, and flipping `prev_range` to `shift(-1)` each fail 3, 3
and 5 tests respectively.

### The metric set

| order | metrics |
|---|---|
| first | `prev_range`, `prev_green`, `zone`, `s_slope`, `f_slope`, `s_bandwidth`, `f_bandwidth`, `s_pctb`, `f_pctb` |
| second | `green_red_avg`, `relation`, `slope_pair`, `cross_low/mid/high`, `bandwidth_ratio`, `gap_low/mid/high`, `pctb_spread` |
| third | `prev_*` of all of the above (`with_third_order`) |

`zone` is the specification's discrete open-position — `BL / UL / L / M / H / UH / BH` —
assigned by **counting** how many of the six band lines the open sits above,
rather than by a chain of pairwise comparisons. A fast low line can sit above a
slow mid in a sharp move, and an `if`-chain would fall through to nothing;
counting keeps the mapping total. Note `M` is the gap *between* the two mid
lines, so it is empty when they coincide — correct, not a bug.

`s_pctb` / `f_pctb` are **centred**: 0 at the mid, ±1 at the bands. Identical to
`2·(%b − 0.5)` for symmetric bands and defined when width is zero.

**`CONTRACTION` is an addition to the specification's three slope types.** Flat,
Expansion (diverging) and Trend (same direction) are named there; converging is
not, which would leave a squeeze — the most-watched Bollinger regime there is —
folded into `FLAT` and indistinguishable from a genuinely quiet band. Flagged
rather than absorbed silently.

`flat_eps` is a fraction of the band's **own bandwidth**, not a number of
points, because SPX ran 2,250 in 2017 and 7,500 in 2026 and an absolute
threshold would mean something different at each end of the sample.

### Sweep axes left open

`source ∈ {close, open, hlc3}` × `ma ∈ {sma, ema}`, per the specification's two question
marks, plus `k`, `fast`, `slow`, `slope_lookback`, `flat_eps` and
`green_red_period`. All hashable on `BandConfig`, all cache keys. Changing `ma`
alone moves 5–15% of zone assignments, so the axes are not cosmetic.

### Bar-to-minute mapping

`for_session(day, tf, cfg)` adds a `minute` column — minutes since 09:30, which
is exactly the row index into a `Session`'s option arrays. A rule that fires on
a 15m bar therefore knows without further lookup which chain minute it may trade
at. (Watch for `dt.hour()` being **Int8**: `(hour − 9) * 60` overflows from
12:00 on and silently yields negative minutes. Cast before the arithmetic.)

## Exits: W, L, COVER, SHORT

A **method** answers only "what would you open, and when". Everything after
that — W, L, covering, shorting, settling — is `runner.py`'s job and is
identical for all of them. Five very different methods differ in what they open
and in nothing else, which is what makes comparing them honest.

### Two fill models, because they are two different order types

**A W exit fills at its own limit price.** It was resting: the moment someone
bids $4.00, the order sitting at $4.00 trades at $4.00. If the minute grid first
shows a bid of $8.00, the limit filled inside that minute at $4.00 — booking
$8.00 would collect a windfall a resting order never received.

**An L exit fills at the market.** A stop is not a resting limit; it is a
decision to be out, and it concedes the spread like any crossing order.

Getting that backwards would flatter every stop in the programme.

### The trigger reads the book the fill model trades on

`trigger_edge` defaults to `None`, meaning *follow the fill model*. This is the
only self-consistent pairing: a W is a resting limit and needs a **bid** at its
level, so triggering off a midpoint while filling at the bid produces a signal
that then cannot be filled. An explicit value overrides it — that is how the
smoothing question gets asked — and a W so triggered may find no bid and be
**refused**, which is recorded rather than silently filled.

### Other rules the tests pin

- **The adverse trigger wins a tie.** A minute has no internal order; assuming
  the favourable one filled first would flatter every result.
- **One trigger, ever.** Every action either flattens the trade or converts it
  into a spread held to expiry, so a trade's exit is a single first-crossing —
  one numpy `argmax`, not a per-minute loop over every open trade.
- **COVER is `step(+n)`, SHORT is `step(−n)`**, on either right. `Session.step`
  owns the direction, so neither call site contains `if right == CALL`.
- **Only a naked long can be converted**, and a cover off the end of the chain
  is refused rather than invented.
- **Debit and credit structures read W and L oppositely.** `w = 0.5` on a credit
  structure means "buy it back for half the credit"; `l = 2.0` means "for twice
  it". **The stated ranges (W 1.5–5, L 0.25–0.65) are the debit-side ones** —
  a credit structure needs its own on the other side of 1.0.
- **Trade P&L always sums to the account.** Trades own their cash so the rows
  are attributable; the broker owns the account. `SessionRun.reconciles` is
  asserted across every policy and both brackets, not assumed.

## Metrics: why Sharpe is not enough here

Risk numbers are computed on the **daily** P&L series, not per trade — a method
opening thirteen trades a day has thirteen correlated bets on one move, and a
per-trade Sharpe would count them as thirteen independent observations.

`Score` also always reports **tail dependence**: `total_ex_top1`,
`total_ex_top5`, `top1_share` and `days_to_half`. Not on request — always.
Deflated Sharpe corrects for how many configurations were tried; it does
nothing about an expectancy that rests on one observation, and that turned out
to be the first thing the data did.

## Regimes: separating ordinary days from history

A naked long held to expiry is paid by the tail, and the tail is a handful of
sessions. Naked puts have the same problem pointed the other way — they will
look magnificent because 2020 is in the sample. So **every result involving a
naked long held to expiry is reported split by regime**. Not to delete the
extreme days: they are history, and a method that cannot survive one is not a
method. Only so that "this works" and "this caught one tariff announcement"
stop looking identical on a summary line.

### Two axes, because they catch different days

**`move_ratio`** — realised move ÷ the market's own priced move (the opening ATM
straddle). *The option market was wrong by this factor.* Catches 2025-10-10: a
191-point move against a $22.90 straddle, **8.4×**, on a day that was only
−2.8%.

**`abs_return`** — the raw size of the day, **open to settlement** (a 0DTE
position lives inside the session, so an overnight gap is not part of what it
could have captured). *This was a historic move regardless of what it cost.*
Catches 2025-04-09 at **+9.99%**, where the straddle was already $157.80 so the
mispricing was a moderate 3.1×.

The two agree on only **6 of the 19** days each puts in its own top 1%. Either
alone would miss most of the other's, so a day is EXTREME if it is extreme on
**either**.

Thresholds are **fixed and interpretable**, not sample percentiles —
`move_ratio ≥ 3.5` and `|return| ≥ 3.0%`. "The priced move was wrong by 3.5×"
means the same thing next year; a p99 cut silently re-labels history every time
the dataset grows.

Result: **1,742 normal / 109 elevated / 33 extreme** sessions, in 27 episodes.

### Episodes, not days

Extremes arrive in clusters — 2025-04-03 → 04-09 is one event, not four.
`episode` groups EXTREME days separated by fewer than `episode_gap` (5)
sessions, and `days_from_extreme` lets a report exclude the *neighbourhood* of
an event: `split(trades, exclude_near=3)`.

### The rule/report boundary

Half this table is computed from the realised day and is **reporting only** —
feeding `move_ratio` to an entry rule would be perfect foresight of the exact
thing being predicted. The other half is known at the open and is safe:
the **opening straddle** (the market's priced move), **VIX**, and **`prior_vol`**
(20-day trailing realised vol from prior sessions).

`SAFE_FOR_RULES` and `REPORTING_ONLY` name which is which, they are asserted
disjoint and exhaustive, and `rule_safe()` returns only the former — the safe
path made the easy one.

`prior_vol` is computed over **every** trading session from the master index,
not the option-session subset: before 2022 SPXW listed 0DTE only Mon/Wed/Fri,
and differencing the subset would treat a Friday-to-Monday move as one day and
inflate the early years.

## Conditional and dynamic exits

### W and L can decay with time held

`w = 4.5, w_end = 3.0, w_half_life = 45` starts the close limit at 4.5× the
entry premium and pulls it toward 3×, halving the remaining distance every 45
minutes. A static level is the same object with `w_end` unset, so **the constant
case is a point in the search space rather than a separate code path** — which
is what lets an optimiser move between them.

Decay is measured in **minutes held**, not minutes of the day: two trades opened
an hour apart are the same trade at different times, and anchoring the schedule
to the clock would hand the later one a level it never asked for.

`l_end` works the same way, in either direction — `l_end > l` tightens the stop
as expiry approaches, `l_end < l` gives a trade more room the longer it lives.

Because the trigger is now an array rather than a scalar, nothing about the
evaluation changed: it is still one `argmax` per trade over a pre-built
per-minute level series.

A fired W books **the level it had decayed to**, not its starting ask — that is
the price it was resting at when it filled.

### One exit beyond the specification's four, off by default

`exit_gate` closes on the first **bar open** after entry at which an indicator
condition passes. Bar open and not any minute: the metrics are valid there and
nowhere else inside the bar, and the specification puts entry/exit assessment on the
bars while reserving the 1-minute option data for limit and stop triggers.

It crosses the spread when it fires, like a stop — a decision to be out, not a
resting order.

There is deliberately **no unconditional time exit**. Nothing is ever flattened
merely because a clock struck, and nothing needs to be: SPX is cash-settled, so
an untouched position expires for free. A time-of-day condition belongs in a
gate, alongside whatever else makes it a decision.

### Precedence when several fire in the same minute

**L, then the gate, then W.** Adverse first, favourable last,
for the same reason ties already went to L: a minute has no internal order, and
resolving it in the trade's favour would flatter every result. `l_action="none"`
removes only the L — the other exits still stand.

### A refused exit is recorded

`Trade.exit_refused` is set when a trigger fired but the book could not fill it.
"Held on purpose" and "tried to get out and could not" are different outcomes
and must not share a row.

## Gates

A `Gate` is one predicate on one indicator column; a `GateSet` is `all` or `any`
of them. Everything is frozen, hashable and flat — **a gate holding a lambda
would be unhashable, uncacheable, and impossible to write into a results table**,
which is the whole reason an optimiser can search this.

- **An empty `GateSet` passes everything.** Switching an indicator off is the
  same object with one fewer gate, so the ungated baseline is a point in the
  search space.
- **A missing indicator never passes.** NaN fails every numpy comparison
  including the negated ones, so `outside` would quietly pass it unless handled.
- **`None`, not NaN, for an unset bound.** `NaN != NaN`, so two identical gates
  built with NaN bounds compare unequal while hashing the same — silently
  defeating every cache key and dedup in a sweep.
- **Reporting-only regime columns are refused by name.** `Gate.validate` raises
  on anything in `regimes.REPORTING_ONLY` rather than trusting the caller to
  remember that gating on `move_ratio` is foresight of the thing being predicted.

The method owns *which* timeframe and band config the gate reads (`features()`);
the policy owns *which condition* closes a position. The runner joins them.

### Mechanical check on the decay

A naive hourly long, L=0.4 throughout, 16 cells × 2 brackets in **18 s**. Per trade,
mid / cross:

| policy | mid | cross | W fire rate (mid) |
|---|---|---|---|
| static w2 | −$36.80 | −$47.10 | 0.278 |
| static w3 | −$33.32 | −$44.56 | 0.167 |
| **decay 4.5→3 @45m** | −$33.09 | −$43.24 | 0.141 |
| static w4.5 | −$31.56 | −$40.66 | 0.105 |
| **decay 6→3 @45m** | −$30.34 | −$41.23 | 0.128 |

The schedule behaves as specified: a decaying W fires **less often than a static
W at its asymptote and more often than one at its start**, and lands between
them on P&L. That is the mechanism working, not a result about any method — none of
these cells is gated, time-bucketed, or swept over strike price.

## The option's own chart

The specification asks for entries and exits on **both** the index chart and the traded
option's own price action, and for **mixing them on one order** — "price action
on the SPX triggers an entry, but the exit is based on Bollingers from the
option's price action". `indicators.option_features(sess, contract, spec)`
computes the full metric set on one contract's chart.

**The metric computation is shared.** `indicators.metrics(bar_frame, cfg)` is
called by both `build()` (SPX) and `option_features()` (option). Two
implementations of `zone` or `%b` would let the vocabularies drift, and a gate
named `s_pctb` would then mean different things depending on where it pointed.

### Session-bounded, and it cannot be otherwise

A 0DTE contract exists for one day, and yesterday's 4750C is a different
contract at a different distance from spot — so unlike the SPX series there is
no continuous history to warm a band from. A 20-period slow band is ready
**exactly 20 bars in**: minute 20 at 1m, minute 100 at 5m. `ChartSpec.validate`
therefore **refuses 30m and 60m by name**, with the reason, rather than
returning a column of nulls.

That is late for an opening trade and fine for power hour, which is where these
timeframes are wanted.

`source="mid"` uses the quote midpoint — present every minute the book is
two-sided. `source="trade"` uses the printed OHLC — higher fidelity, gappy at
far strikes.

Cached on (session identity, contract, spec); `Session` is `eq=False` so it
hashes by identity, which is exactly what a per-session cache key wants.
**18 µs cached, 1.6 ms cold.**

### Routing: `Gate.chart`

`spot` (default) or `option`. A mixed `GateSet` is split with `for_chart()`:

- The **spot** half is the same for every trade in the session, so it is built
  once per session.
- The **option** half is per contract and is built when the trade exists.
- An **entry** gate on the option chart is applied by the *runner*, after the
  candidate strike is known — until then there is no option to have a chart, so
  it cannot live inside the method.
- `mode="any"` **cannot span both charts** and raises: the two frames have
  different rows, so the disjunction is undefined. `all` splits cleanly and is
  what a mixed rule means.

## Timeframe cost

One ungated naked long, all 1,894 sessions, per (cell × bracket):

| bars | trades per cell | seconds per cell |
|---|---|---|
| 60m | 11,228 | **0.23** |
| 5m | 120,654 | **5.8** |
| 1m | 596,152 | **11.3** |

**A 1m cell costs ~50× a 60m cell.** That is the number to budget an Optuna run
against: a 500-trial study at 60m is two minutes, the same study at 1m is over
an hour. Gating cuts the trade count and most of the cost with it.

## The search layer

```bash
python -m lab.study --method ZONE --trials 200 --fold 0
```

Three stages in the order that keeps them honest: search on **training days
only**, evaluate the winner **once** on held-out test days, then **deflate**
against every trial run.

### Why TPE and not a GA

The space is **conditional** — an indicator's threshold exists only if that
indicator is on, and a COVER's width only matters if the W action is a COVER. A
GA mutates dead genes; TPE's define-by-run expresses it as ordinary Python
control flow, which is exactly what the `suggest_*` functions are.
`TPESampler(multivariate=True, group=True)` models each conditional group
separately rather than marginalising over trials where the parameter was absent.

The usual argument for Bayesian optimisation — expensive evaluations — does
**not** apply here. A trial is a couple of seconds. The binding constraint is
overfitting, which is why three rules are wired in rather than left to
discipline:

1. **The objective is the CROSS bracket.** Nothing whose edge depends on
   midpoint fills can win a trial. The mid Sharpe and the gap are recorded on
   every trial, so the size of the execution dependence is always visible.
2. **Every trial's Sharpe is kept.** The DSR needs the spread across trials and
   an honest count of them; Optuna's storage is what makes that count durable.
3. **A random-search arm is first-class.** Cheap, and without it there is no way
   to tell whether TPE found signal or reached the noise ceiling faster.

A configuration trading fewer than `MIN_TRADES` (200) scores as a failure — a
Sharpe from four lucky sessions is not a result.

### Thresholds are searched as quantiles

`s_bandwidth` lives near 0.01, `s_pctb` on [−1, 1], `prev_range` in index
points. A gate's threshold is suggested as a **quantile** and mapped through the
column's own empirical distribution: scale-free, uniform across columns, and
stable as the index goes from 2,250 to 7,500.

**The quantiles come from the training window only.** Deriving them from all
history would leak the test period's distribution into the definition of every
threshold — subtle enough to survive review, large enough to matter.

Three column kinds, three encodings:

- **numeric** → `ge` / `le` / `between` on quantiles
- **ordered categorical** (`zone`) → a **contiguous band**, because zones run
  BL < UL < L < M < H < UH < BH and a band searches far better than seven free
  booleans
- **set categorical** (slopes, relations) → a subset via a bitmask

The asymptote of a decaying W is searched as a **fraction of its start**, so the
two cannot cross and a decaying W always relaxes rather than tightening.

### Walk-forward, not one split

`walk_forward` yields successive (train, test) windows, so a configuration has
to survive refitting through 2018, 2020, 2022 and 2025 rather than through
whichever regime landed in the second half.

**Splits are by date and contiguous.** Random fold assignment would put
2025-04-08 in train and 2025-04-09 in test — a rule fitted on one tariff
headline validated on the next day of the same event.

**A purge gap (25 sessions) separates them.** Nothing here holds a position
overnight, so leakage through open trades is impossible — but the indicator
series *is* continuous across sessions, and a 20-period 60m band on the first
test day is computed from bars inside the training window.

### The Deflated Sharpe Ratio

Given **N** configurations tried, how surprising is the best one's Sharpe? The
benchmark rises with N *and* with the spread of Sharpes across trials — a more
diverse search means more chances to get lucky, so the same edge stops clearing
the bar. Skew and excess kurtosis are corrected for, and both matter here: a
0DTE long's returns are violently right-skewed and fat-tailed, and the textbook
Sharpe standard error assumes neither.

**The spread of trial Sharpes is estimated robustly.** The classical estimator
is the plain sample variance, which assumes trials are draws from one
well-behaved distribution of candidates. A real search is not that: it produces
a cluster of plausible configurations plus a handful of catastrophes. The first
study run here had trial Sharpes from −0.5 down to −10, and the sample variance
those implied set the selection benchmark at **6.44** — a bar nothing of any
kind clears, which makes the DSR uninformative rather than strict. The scale now
comes from the interquartile range (÷1.349). **The trial COUNT is untouched**:
every configuration tried still counts toward N, because that is the
multiple-testing term and trimming it is exactly the dishonesty the statistic
exists to catch. Only the dispersion is made robust.

**DSR does not rescue an expectancy resting on one observation.** It prices
multiple testing, not a degenerate payoff. It sits *alongside* the
tail-dependence figures and the regime split, never instead of them.

### Pool reuse

`sweep.run` keeps one process pool across calls. Building a fresh pool per trial
pays 24 × (spawn + import Polars + build the spot index) every time — several
seconds of overhead against about a second of work. Reuse also keeps each
worker's session and indicator caches warm, which is the larger win from the
second trial onward. **6.7 s → 2.4 s per trial.**

## Measured redundancy in the metric set

Two metrics are **exact functions** of others. Reconstructed at
**100.000000%** over 31,346 bars at 30m, and asserted by tests at 5m, 30m and
60m:

```
zone     = f(s_pctb, f_pctb)                 discretised at -1 / 0 / +1
relation = f(sign(gap_low), sign(gap_high))  gap_mid is not involved
```

`s_pctb > 1` **is** `price > SH` by construction, so the six band comparisons a
zone counts are already carried by the two centred %b values. The gaps are
signed, so their signs already say which family sits inside which — no change
was needed there.

**The reverse does not hold, and the gap is where these methods live.** Within
zone `BL` the observed `|s_pctb|` runs from 1.00 to **7.11** (p50 1.39, p99
3.81). A bar seven half-widths below the mid and one barely below SL are the
same zone, so `s_pctb <= -3` is a genuinely different — and far rarer —
condition than `zone == BL`.

**So keep both, for search efficiency rather than information.** `zone in
{BL, UL}` is one ordered-band parameter; the same condition through %b asks an
optimiser to discover two continuous thresholds landing on exactly −1. The
categoricals are a *prior on where the interesting cuts are*, and a good one,
because those cut points are how the trade is described. The cost is real: two
views of one feature double the ways to express a rule, which is more room to
fit noise.

`search.Space.metric_view` makes that a controlled experiment rather than an
accident:

| view | columns | what it is |
|---|---|---|
| `both` | 20 | the default |
| `categorical` | 9 | the coarse trader vocabulary, smallest space |
| `continuous` | 11 | strictly richer, unbounded, harder to search |

`indicators.DERIVED_FROM` records the dependencies so nothing has to re-derive
them, and a test asserts the two views partition cleanly with every derived
column on one side and its sources on the other.

## Staging: what to run, and in what order

Sweeping everything jointly from the start is what the first study did at 50
trials, and **random search beat TPE** — the signature of a space too large for
the budget. But pure staging is worse in a different way: it assumes
**separability**, and there is direct evidence against it (a decaying W is
known to work in live trading, while every W is destructive on the *ungated*
baseline — that gap is an interaction between entry and exit).

So stages narrow **ranges and priors**, never freeze values.

1. **Ungated baseline + conditional scan** (`scan.py`). Nearly free, consumes no
   trials. This is what the specification already prescribes — *"emergent
   statistical buckets that will suggest useful indicators for entries."*
2. **Cheap exhaustive grids for SHAPE** (`sweep.wl_grid`). A plateau versus a
   spike is the overfitting diagnostic, and TPE cannot give you the surface.
3. **Joint Bayesian on the narrowed space** (`search.py`), with the random arm,
   at 1000+ trials.
4. **Walk-forward + DSR** (`validate.py`) with the honest cumulative count.

The entry window belongs in step 3 as an **axis**, not as a preliminary split:
bucketing first multiplies the trial count by the number of buckets and starves
each sub-study of trades. `scan.by_hour` gives the bucketed *report* for free.

## The conditional scan

One ungated run, then per-trade P&L sliced by every indicator's own deciles.
Joined on **(`date`, `bar_minute`)** — the signal's own bar, not the fill
minute, because the liveness gate can push a fill a minute or two later and
joining on the fill would silently drop every late-opening session.

**It is not a significance test.** Twenty columns times ten buckets is two
hundred comparisons on one dataset. Its purpose is *ordering* — which columns
are worth putting in the search space — and anything it suggests still has to
win a trial in a study whose count includes it.

### Why `monotone_p` exists

A column whose P&L rises steadily across its deciles is a far better gate
candidate than one whose middle bucket happens to be highest: the first is a
relationship, the second a coincidence with ten chances to happen. But rho
alone invites over-reading. Measured over 60 synthetic draws of 2,000 trades
with **no relationship at all**:

| noise \|rho\| | median | p90 | max |
|---|---|---|---|
| 10 buckets | **0.21** | **0.52** | **0.73** |

This is not hypothetical. The first real scan run here ranked `s_pctb` at
rho = −0.60 — which the p-value puts at **0.067**, not distinguishable from
noise, and exactly the reading that would otherwise have entered the search
space as a finding.

