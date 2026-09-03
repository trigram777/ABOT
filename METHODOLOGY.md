# A 0DTE Options Research Programme
## Methodology, engineering record, and the mistakes that shaped both

---

## What this is

This is the condensed record of a research programme that builds and trades
intraday options strategies on a cash-settled index — from raw vendor tick data,
through a purpose-built backtesting engine, to a live decision engine placing
orders at a broker.

It is not a strategy write-up. It is a record of **how the work was done**: what
the engine had to be able to express, which measurements turned out to be
artefacts of the measuring apparatus, which promising results died under a proper
null, and which bugs were only findable in production. The interesting content is
the errors. Nearly every rule below exists because something looked right for a
while and wasn't.

The underlying research log runs to about 16,700 lines and is maintained as a
living document — every result, every reversal, and every withdrawn claim stays
in it, struck through rather than deleted. This is the readable extract.

### The redaction rule

This is public, so the tradeable content is gone. The rule applied throughout:

| removed | kept |
|---|---|
| Every constant — thresholds, multipliers, widths, offsets, deltas | Every *finding about method* |
| The entry schedule and all clock times | Facts about market microstructure |
| The strike-selection corridor | Error magnitudes, expressed as ratios |
| All absolute P&L, Sharpe and drawdown figures | Engineering invariants and bug post-mortems |

Absolute performance figures are removed for two reasons, not one. The obvious
one is that they are the edge. The less obvious one is that an unauditable
profit claim in a portfolio document is worth *negative* credibility — the reader
cannot check it, so it reads as either boasting or naivety. Where a magnitude is
load-bearing to a methodological point, it appears as a ratio or a percentage,
which carries the whole argument and none of the strategy.

Nothing in this document is a solicitation, a recommendation, or a claim of
profitability.

---

## 1. The problem, and why it is harder than it looks

Same-day-expiry index options are an unusually punishing research target, in ways
that have nothing to do with predicting direction:

**The instrument dies daily.** Every contract has one session of price history.
Any indicator with a lookback longer than a few minutes is unavailable for the
first part of the session and is never available on a longer timeframe at all.
There is no cross-session continuity to lean on.

**The decisions are path-dependent.** Trailing exits, stops that reprice, and
re-entries conditioned on earlier fills all mean a position's outcome depends on
the *order* in which prices arrived, not on any summary of the day. This removes
most of the standard backtesting shortcuts: you cannot score a trade from
open/close, and you cannot vectorise over trades independently.

**The costs are not a rounding error.** Commissions are charged per leg, on both
ends. On cheap contracts they can reach a double-digit percentage of the premium.
A study that prices them at the end rather than the beginning does not get a
slightly optimistic answer — it can get the wrong *sign*. (This is not
hypothetical: the predecessor programme to this one ran nine stages of
optimisation before charging fees, and charging them turned the refined result
into a loser.)

**The dataset is large and the questions are combinatorial.** The working panel
is 1,894 sessions (2017 → 2026), stored as dense `[minute, strike]` arrays.
Parameter sweeps routinely run 16,000 to 1,900,000 cells.

**And the ground truth is at a resolution you probably don't have.** More on this
in §4.1, because it turned out to be the single largest effect in the entire
programme — larger than every strategy parameter combined.

---

## 2. The engine

A purpose-built backtester, ~5,800 lines of engine plus ~4,500 lines of tests.
The design constraint that shaped everything: **the engine knows only how to buy
and sell options at strikes.** Every structure the programme studies is a
composition of four verbs — `submit`, `close`, `close_all`, `settle`. No strategy
logic lives in the engine. This is what makes a new study a day of work instead
of a week, and it is what made it possible to publish the engine without
publishing anything proprietary.

| module | owns |
|---|---|
| `session.py` | one expiry as dense `[minute, strike]` arrays; spot at 1/5/15/30/60m; strike selection by price, delta or offset |
| `paths.py` | **the resolution axis** — bar / intra-minute / reconstructed-from-ticks — and the tick grid |
| `fills.py` | the mid↔cross fill bracket and tick concession |
| `broker.py` | atomic orders, per-leg commissions, free settlement, full ledger |
| `indicators.py` | a dual-band metric set, computable on the index chart *or* on a contract's own price path |
| `gates.py` / `exits.py` / `methods.py` / `runner.py` | the strategy apparatus |
| `sweep.py` | many cells, one pass over the data; a day is the unit of work |
| `metrics.py` | Sharpe, drawdown, and **tail dependence, always** |
| `validate.py` | walk-forward folds with a purge gap, Deflated Sharpe |
| `search.py` | Bayesian search over a conditional space |
| `scan.py` | conditional P&L by decile from a single ungated run |

Throughput, per cell across all 1,894 sessions: 0.23 s at hourly bars, 11.3 s at
one-minute bars. Session load is ~66 ms warm.

### Four facts the engine encodes, each measured rather than assumed

**A minute's quote is the book at the *start* of that minute.** Measured: the mean
absolute difference between a minute's quote and that minute's *open* is $0.07;
against its own *close* it is $0.31 — a factor of four. So a rule that decides at
`m:00` and fills against the quote at `m` is honest, and no shift is applied. The
close-of-minute index value is *named* `spot_close` specifically so that using it
is a visible choice rather than an accident.

**Never hardcode the first tradeable minute.** The option book does not exist at
the opening bell on essentially every session. It is true on 1,890 of 1,894
sessions that trading is possible one minute in — and wrong on four. One session
has no book until eleven minutes after the open. Rules gate on a liveness
predicate, never on a constant.

**Bands read closed bars only.** Every indicator at bar `t` is computed from bars
`t−n … t−1` and compared against bar `t`'s *open*. There are **two** independent
lookahead tests, because they catch different bugs: truncating the series and
demanding the surviving rows are identical catches a window that peeks forward,
but it does *not* catch a band that reads its own bar — deleting the future
doesn't change bar `t`'s own close. So a second test perturbs one bar's
high/low/close while leaving its open alone and demands nothing on that bar
moves. Both tests were mutation-checked, i.e. deliberately broken to confirm they
fail.

**Process pools use `spawn`, never `fork`.** The dataframe library starts a
thread pool on first use, and a forked child inherits a pool whose threads do not
exist. **The symptom is silent** — N workers parked in `futex_do_wait` at load
0.1, forever, with no error and no timeout. The same silent symptom has a second
cause worth knowing: an OOM-killed worker. `Pool.imap_unordered` cannot report a
dead worker; `ProcessPoolExecutor` raises. Prefer the one that raises, and assert
the output count before merging.

---

## 3. The disciplines

The programme maintains a numbered list of standing rules. Each was added the day
something broke. These are the ones that generalise beyond this domain — they are
reproduced close to their original wording because the specificity is the point.

**Normalise by capital at risk.** Per-trade dollars are confounded by position
size — a $10 option risks twenty times what a $0.50 one does. The raw and
normalised readings of the same grid pointed in *opposite directions*. Report
return on risk, and report the fee share beside it, because at the cheap end
commission decides the sign on its own.

**A hit rate is not an expectancy.** A 65.3% "ends up in the desired state"
statistic became a *negative* result the moment price was attached to it.

**An entry is scored on what it is TRYING to do.** Where an opening trade exists
to bring about a *state* rather than to profit on its own, the state is the
outcome column and its P&L is not the objective. Scoring such an entry on its own
P&L measures something the method is not attempting. Two withdrawn sections of
the research log are what that mistake looks like.

**Any signal measured against a price-selected strike must be controlled for that
strike's distance from spot.** A fixed-dollar option is a few points out of the
money on a quiet day and many points out on a wild one — so *every* volatility
proxy correlates with the geometry and looks predictive for a reason that has
nothing to do with prediction. Sighted three times. One candidate signal lost
~90% of its effect to the control; a family of others lost 74–83%. The control is
one line — recompute the statistic within bands of the offset — and where a
signal survives it, say by how much.

**A control that changes the sample size cannot be applied to a statistic that
depends on it.** Residualise; do not subset.

**Buckets must be comparable before they are compared.** Reporting by time of day
silently compares different populations of days, because later entries do not
always qualify. Report the count per bucket, and where counts differ materially,
report a balanced panel beside the raw one. Related: report *every* bucket, never
just the winner — seven time buckets is a seven-way argmax.

**A gate that declines days must be judged as a classifier, not only as a P&L
filter.** Report recall, precision, and the ordinary days discarded, against the
base rate. A P&L improvement alone cannot distinguish "this finds the bad days"
from "this trades less."

**Ask what a parameter actually varies before searching it.** One threshold in
this programme read like a payment threshold and was nothing of the kind — the
selection walk stopped at the last qualifying strike, so the quantity it appeared
to control was *pinned* and what actually moved was distance from spot. A range
chosen for the wrong quantity searches the wrong axis.

**A candidate found at one parameter setting has been found at one setting.**
Re-run it across the axis it was found on before it enters the search space. One
candidate *flipped sign*, at a cost of sixty seconds of compute.

**A test window spent on parameter selection is no longer a test window.** It can
confirm a setting fixed before it was looked at; it cannot choose between
settings and then report the winner as out-of-sample. Compute shape diagnostics
on training data wherever the answer will influence a choice.

**Freeze and write down a specification before the work that tests it.** A rule
that is still moving cannot be falsified by the next result.

**Tail dependence is always reported.** Deflated Sharpe prices multiple testing;
it does nothing about an expectancy resting on one observation. And *which* tail
matters depends on the sign of the structure: a debit method is diagnosed by
deleting its best days, a credit method by deleting its worst. Reporting only the
former on a short structure measures nothing.

**Nothing fitted is adopted until it has been spent out of bag** — see §4.2,
which is the most useful thing in this document.

---

## 4. Six findings that changed the programme

### 4.1 The resolution of a backtest is set by how often the live engine *looks*, not by the granularity of the data

This is the largest single effect measured anywhere in the programme, and it was
invisible for months.

The setup: a path-dependent exit — a trailing stop — evaluated over reconstructed
intraday prices. The obvious question is "what data resolution do I need?" The
right question turned out to be different.

Taking **one reconstructed price path** and changing only how often the decision
loop reads it:

- read once a minute → reproduces the coarse-bar backtest almost exactly
- read every five seconds, as the live engine actually reads it → **the reported
  daily P&L falls by roughly two thirds, and the Sharpe by roughly 40%**

The entire discrepancy between the optimistic backtest and the honest one is the
**decision cadence**. Not quotes-versus-prints. Not the smoothing filter. Not
tick data versus bars. The polling interval of the live engine is a backtest
parameter, and it had never been treated as one.

The mechanism is simple once seen: a trailing stop evaluated once a minute cannot
give back a peak that occurred *inside* that minute. It is measuring against a
market that does not exist. Measured on this dataset, a single minute's own
traded range is 10–15% of a typical contract's price, and a **majority of
individual minutes span the entire give-back of a typical trailing stop** — rising
above 95% in the most volatile part of the session, and worsening every year of
the decade.

Two corollaries that cost real money to learn:

- **Looking less often is not a free way back to the optimistic number.** A trail
  stepped on a slow poll is a *watched* trigger with no resting order, so it
  cannot fill at its level. Changing *when* an order is sent changes *what kind*
  of order it is — resting versus watched — and those fill at different prices. A
  delay may never be compared against an immediate rule without holding the fill
  model constant.
- **An offset finer than the tick grid is not an offset at all.** Every level
  snaps to the grid, so the parameter surface goes flat where the ticks run out
  instead of running away into a fictional optimum.

The general lesson: **the artefact is the decision loop.** Before asking what
resolution your data needs, ask what resolution your *decisions* run at, and make
that an explicit axis of every result.

### 4.2 A per-cell argmax has never once survived being spent out of sample

The programme's most productive rule, and the cheapest to implement (about twenty
lines):

> Fit the thing on a resample of the sessions, score it on the sessions left out,
> and compare it there against the simplest alternative it claims to beat — usually
> a single uniform value. **The null is not zero**: an argmax over a grid gains
> something on pure noise, and that gain is what the out-of-bag distribution
> measures. Report the mean gain *and* the share of draws on which the fitted
> value actually wins.

It has overturned a live reading three times and retroactively explained four
earlier results. In **five** separate attempts to fit a parameter per-bucket
rather than globally, the fitted vector has **never** beaten a single uniform
constant out of bag. One representative case: a per-bucket fit gained +0.16 of
Sharpe in sample and lost **−0.237 out of bag, winning on 30% of draws** — i.e.
worse than a coin flip against doing nothing.

Three things make this rule sharper than a plain train/test split:

**Report the argmax reproduction rate.** Across resamples, how often does a cell
land on the same winner? A cell that reproduces 50% of the time is *a coin flip
wearing a decimal point*, and reporting its fitted value to two decimals is a
category error.

**Put the error bar on the DIFFERENCE, not on either column.** Statistics
computed over the same sessions are paired. The unpaired bracket is both wider
and the wrong question.

**The fitting criterion does not have to be P&L for this to bite.** One constant
was fitted four different ways — on P&L, on a geometric median, on an agreement
rate, and on a price — and **all four lost out of bag.** The two that a live
system could most conveniently optimise, because they never touch the outcome at
all, lost hardest (winning 12% and 2% of draws). *A criterion that never touches
P&L is not thereby safe from selection.* This is the single most
counter-intuitive result the programme produced.

The rule has exactly one survivor, and it is instructive: a global constant on a
monotone axis, which won 100% of out-of-bag draws at 91% reproduction. It
survived because it is **not a per-cell fit at all** — there was no
argmax-over-noise to deflate. **And it was still not adopted**, because reading
the axis at matched capital revealed it to be a position-sizing knob wearing the
costume of a selection parameter. Surviving the rule is *necessary, not
sufficient*: it certifies that a number is not noise, not that it is the right
number.

### 4.3 There was nothing to select among entry times, and the null construction is what proved it

A natural hypothesis: some entry times are better than others, so learn which.

The measurement that killed it is worth copying. The best entry bar of the
morning repeats as the best bar on **17.6%** of subsequent sessions, against
**16.7%** expected by chance. And the decisive check: **shuffling each bar's
column independently produced a *higher* oracle value than the real panel.** If
your idealised, look-ahead-perfect selector performs *better* on scrambled data
than on real data, there is no structure to select on, and the apparent structure
is the selector's own variance.

A companion result on the indicator side: across every candidate signal, four
windows and three timeframes, three of four windows produced nothing above a
day-permutation 99th percentile — and a *random* daily pick matched the tuned
ruleset.

Both results are negative, both took real work, and both saved far more work than
they cost. **Sizing the prize before chasing it** is now a standard first step:
compute what a perfect oracle would earn, and if the oracle is small, stop.

### 4.4 A responding order priced at the triggering bar is lookahead — worth 10× the effect being measured

Subtle, and it survived several rounds of review.

A minute's quote is the book at the *start* of that minute. A stop triggers on
the intra-minute *low*. So when a trigger on contract A causes an order in
contract B, pricing B at B's start-of-minute quote uses a price that existed
*before* the event that caused the order. Closing a position is immune — same
contract, own resting level — but every conversion, hedge, roll or re-entry
placed in response to a trigger is exposed.

Measured on one such idea: priced at the triggering bar it showed **$62.73 per
entry**; priced at the next available book, as it would actually fill, **$5.80**.
An order-of-magnitude difference, and the idea was correctly rejected only after
the fix.

The general form: **lookahead does not only live in indicators.** It lives in the
plumbing between a trigger and its consequence, which is exactly where nobody
looks because it feels like execution code rather than research code.

### 4.5 A hedge that loses money on every axis, for a reason that is structural

One structural component was carried for a long time on the intuition that it was
insurance. Measured across nine rule variants and a wide range of parameters, its
cost was stable and always negative.

The reason it *must* lose is economic, not statistical: the component is the same
trade as the main position, reversed. It buys the very richness the method exists
to sell. Averaged over enough sessions it can only return the spread, and it does
not insure the risk that actually hurts — a bad fill on a fast reversal — because
that risk materialises *before* it pays.

It is retained anyway, for the one thing it does do: it converts an undefined
risk into a defined one, which is what makes return-on-capital computable at all,
and which is what makes "more of this trade here" versus "one of this and one of
that" a well-posed question.

The transferable point: **an insurance component should be priced against the
specific risk it is claimed to insure, not against total P&L.** And when the
answer is "it costs money and we keep it anyway," that should be stated as a
financing decision rather than smuggled in as a performance improvement.

### 4.6 The fill model was wrong, and the domain expert was right

The engine originally reported every result across a fill bracket, from midpoint
to full spread crossing, and treated the pessimistic end as the conservative
number. The trader's objection, roughly: *these are among the most liquid options
listed; a multi-leg order has its own book, far tighter than the sum of its legs;
they fill and close at the midpoint, observed daily; commission is the real drag
and you are modelling it correctly already.*

This was accepted in full, and the reasoning matters more than the conclusion:
**pricing a spread at bid-minus-ask is not a conservative version of that trade —
it is a different trade, at a price nobody was asked to pay.** It also silently
*selects different strikes*, because the qualification step is what chooses them.
A wrong model is not made safe by being pessimistic.

The specific institutional detail — that a combination order has its own book —
is not something a backtest can discover. It came from someone who watches the
fills. That is worth saying plainly in a document like this: the most important
correction to the cost model came from domain knowledge, not from data.

The bracket was retained for reproducing pre-existing results, and one frozen
specification was deliberately *not* re-based, because it had been pre-registered
against the old model and moving its basis after the fact would have destroyed
the pre-registration.

---

## 5. From backtest to live: two implementations, one rule

The backtest and the live engine are **two independent implementations of the
same rule**, and that is deliberate.

- The backtest walks numpy arrays and can see the whole session at once.
- The live engine carries its own state one price at a time, and can never see
  ahead.

A differential harness runs both over the same real sessions and compares them
**per fill**. Recent run: 3,788 legs, 3,686 identical exits, and **zero
disagreements not explained by the tick grid** — the acceptance criterion is not a
percentage threshold but a requirement that every residual difference have an
identifiable off-grid cause.

That differential is the only thing that makes a backtested number *evidence
about the deployed code*. Without it, the backtest describes a program that does
not exist.

Two hard-won details:

**The differential derives its specification from the live constants** rather than
restating them. An earlier version hardcoded them, and they drifted — so the
harness built to detect divergence was itself diverging.

**Trail objects expose a single `step(price) -> bool`, not separate
test-then-update calls.** A caller who tests before updating gets a subtly
different rule that still looks correct in a log. Making the wrong usage
*unexpressible* was cheaper than documenting it.

### The limit of the technique, discovered three times

**A state reachable only through a broker rejection is invisible to the
differential.** The backtest's execution path always fills. So any branch that
exists to handle "the broker said no" is never executed on the backtest side, the
two implementations can diverge freely there, and the harness cannot see it.

This was sighted three separate times before it was named — a component left with
no exit armed; an order-pricing path that fell back to stale data instead of
admitting it had no book; a converter applied twice across a seam because both
halves were tested against fakes shaped like themselves. All three were live-only
failures in code with green test suites.

The generalisation: **differential testing verifies the paths both sides can
reach.** Enumerate the paths only one side can reach, and test those directly.

---

## 6. Seven production bugs worth reading

Selected for transferable lessons, not severity.

### `getattr(obj, 'attr', None)` turned a missing interface into silent absence of data

A price lookup read a quote book through `getattr(port, "book", None)`. The real
adapter did not hold a book — it *pushed* quotes. So in production the lookup
returned `None` for every contract, every trailing stop stepped on `NaN`, **no
water mark ever moved, and no stop could ever fire** — silently, for a full
session.

The test suite stayed green because the test fake had a `.book` attribute the
real port did not.

Two lessons, and the second is the bigger one. A `getattr` default converts a
missing *interface* into a plausible *absence of data*, which is much harder to
detect than a crash. And **a fake shaped differently from the real adapter makes
the entire suite meaningless** — that is now checked by driving real objects
against each other across the seam, with fakes that *refuse* anything of the
wrong shape.

### One spurious print permanently tightened a stop

Trailing levels read a **smoothed** price (a rolling median with stale samples
dropped); orders are priced against the **raw** book. Reversing either is a bug in
a different direction.

A ratchet keeps the most extreme price it has ever seen, so a single spurious
print tightens the level for the rest of the session and never washes out.
Measured on a live session: one bad tick dropped a water mark from 2.82 to 1.269
**permanently**. Conversely, a ladder started from a smoothed price begins at a
rung the market is not actually on.

**Smoothing is not a global property of a system. It is a property of a
question.** Valuing a position needed a *third* price again — an ask with no bid
is worth something, and `None` there means only that the bid was not positive.

### A docstring is not an implementation, and a test that does the work itself proves nothing

A method's docstring said the recovery path "gets its exit armed at once," and
named the alternative as "the one state this method must never sit in." It never
armed it. The position then fell through the same branch every tick and ran to
settlement in silence.

The test looked thorough and proved nothing, because **it performed the arming
itself before asserting** — demonstrating that the exit fires once armed, and
never that anything armed it.

Fixed as an *invariant* at the top of the state machine rather than as a line in
the branch, because the arming needs a price and the failing path does not have
one.

### A subscription callback log is not a position table

The broker API's position query returns an **append log**: every position event
that arrives while any request is open is appended to a shared list, and the
underlying subscription is never cancelled. Summing it reported the account
holding **three times** the contracts it actually held.

The fix is to collapse on `(account, contract)` keeping the **last** row — sizes
are absolute, not deltas — and to drop zero rows **after** the collapse, because a
position closed during the window is otherwise resurrected by an earlier row.

Worse, the client library keys the in-flight request on a literal string, so two
overlapping calls orphan one future and resolve the other with a half-filled
list. Reads are now serialised.

**Reconciliation reports; it never corrects.** Auto-correcting a disagreement
about what is real means placing an order to resolve a bookkeeping question.

### A timeout is not an answer, and it censored the measurement it existed to collect

A fill-report wait gave up after fifteen seconds, wrote a `pending` row with a
blank price, and moved on. Two orders in one session filled at *exactly* twenty-one
seconds. Both were among the largest price slips of the day, one in each
direction.

This is the part worth internalising: **censoring the latency distribution
censors the slippage distribution**, because a fill that takes longer is one the
market had longer to move for. The most important number in the entire
programme's cost model was being systematically trimmed of exactly its tail — and
**a distribution cannot be learned from its own tail** when the tail is what gets
dropped. Until it was fixed, twenty-one seconds and infinity were the same
observation.

It was silent because a separate consistency check *agreed*: the position was
marked as exiting the moment the order was emitted, so the reconciler stopped
expecting it whether or not it had filled.

Now: no deadline while the broker still reports the order open (the correct
response there is to wait), a deadline only when the broker cannot be asked, and
a wait-duration column on **every** order row.

### `QKeySequence('Q')` and `QKeySequence('q')` are the same sequence

Both were registered. That makes the binding *ambiguous*, and the framework then
emits `activatedAmbiguously` rather than `activated` — so **the quit key never
worked, from the day the window was written, and nothing reported it.**

Now: one binding per key, from a single table, with the ambiguity signal wired to
a visible log line so a double-binding can never be silent again.

The deeper finding was in the test suite. **Every window test constructed the
object via `__new__`**, bypassing `__init__` — where all the keyboard wiring
lived. The wiring had never been executed by a test at all. Several tests now
build a real window offscreen and press real keys.

### Every clock the program prints is one timezone, and that timezone has one definition

Every rule in the system is expressed as minutes after the market open, so the
timezone is a **fact about the rule**, not a display preference. Several
components were printing in the *system* zone: transcripts, log filenames, and —
critically — the default contract expiry. Between late evening and midnight local
time, the exchange date is already tomorrow, so the session state restored into
the wrong day's file and the default expiry named a different contract.

The chart was the subtle one. A timestamp normaliser dropped the timezone while
keeping the wall-clock reading, and the charting library re-rendered that as UTC
— so the axis silently displayed whatever timezone the broker terminal happened
to be set to.

The test suite now runs identically under three system timezones, with tests that
feed a UTC instant and assert the exchange-local rendering, so a
system-zone-dependent formatter fails everywhere at once rather than nowhere.

---

## 7. How it is tested

Roughly **1,000 test functions** across two suites — one for the research engine,
one for the live application. The engine suite runs in about eight seconds, which
is the number that actually matters: a suite slower than that stops being run.

What the tests are for, in rough order of value:

1. **The differential** (§5) — two implementations of one rule, compared per fill
   over real sessions.
2. **Two independent lookahead guards**, both mutation-checked.
3. **Real-dataset shape assertions** — the engine's assumptions about the data are
   asserted against the actual panel, not against fixtures.
4. **Cost-model pinning** — the commission model reproduces real broker
   confirmations *to the cent*, across eight multi-lot prints on two instruments.
   That test is what allows every backtest to charge fees with confidence.
5. **Cross-seam integration tests with hostile fakes** — fakes that reject
   anything of the wrong shape, added after the `getattr` bug above.
6. **Regression reproductions** — a harness that reproduces an old published
   result exactly, so that when a number moves it can be attributed to the *rule*
   and not to the harness. This is one of the highest-value tests in the
   programme and one of the least common in the wild.

One test is documented as **wall-clock dependent** and will fail on a loaded
machine — it polls a state machine against a fast synthetic clock. It is
annotated as such rather than quietly marked flaky, and the durable fix (sampling
state from the feed's own tick instead of polling) is recorded as owed work.

---

## 8. What is still open

A portfolio document that only lists successes is not a research record. The
honest state:

**The dominant unknown cannot be measured from historical data at all.** The
single largest uncertainty in the whole cost model is the price slippage on a
market order sent under stress. Tick data cannot answer it — only the system's own
live order log can, and that log currently holds a sample of about two dozen
orders across two quiet sessions. Every remaining parameter decision is
downstream of a quantity with n ≈ 23, and the correct response is more live
sessions, not more backtesting. This is stated at the top of the research log so
that nobody, including me, mistakes a well-measured secondary result for progress
on the primary unknown.

**A calm population cannot adjudicate a tail question.** A high-resolution data
pull covering 148 recent sessions ranks two candidate configurations as a
statistical coin flip. Over the full decade they are not close. The recent
population simply does not contain the events that separate them: its worst-case
capital commitment is ~2.3× its median, where the decade shows 9.6–16×. **The
higher-resolution measurement is not automatically the deciding one** — it decides
what it contains.

**One result is favourable, unadopted, and blocked on a data gap.** An
alternative structural parameterisation beats the deployed one substantially and
survives out-of-bag testing as an a-priori-named candidate. It is not adopted
because the high-resolution data was indexed against the *deployed* selector, so
only ~70% of the candidate's entries are covered — and the uncovered ones are
scored as unavailable rather than as missing. **A coverage hole is not a
result.** The fix is another data pull, which is queued and, this time,
acceptance-tested before it runs: the first version of that index silently missed
9.1% of the very legs it existed to cover.

**Several known defects are documented rather than fixed**, with the reasoning
recorded — including one non-obvious one about whether a resting stop order holds
queue position at all on this product class, which is verifiable only live.

---

## 9. Colophon

**Language:** Python 3.13 throughout.

**Data:** polars for the on-disk panel (~29 GB of vendor tick and quote data),
numpy for the in-memory `[minute, strike]` arrays. The division is deliberate:
polars for anything scan-shaped over the store, numpy for the inner loop. One
memory lesson worth recording — a naive `scan_parquet(...).unique().collect()`
over the full store peaked at **9.2 GB** and killed workers; the streaming engine
answers the same query in 0.7 s at 735 MB.

**Testing:** pytest. ~1,000 test functions across two suites.

**Search:** Optuna (TPE), with a **random-search control arm at equal trial
count** — without it there is no way to distinguish "the optimiser found signal"
from "the optimiser reached the noise ceiling faster." Thresholds are expressed as
*training-window quantiles* rather than absolute values, so a fold cannot inherit
a level from data it never saw. Every trial's score is retained for an honest
multiple-testing count.

**Statistics:** scipy where a standard test exists; hand-rolled permutation nulls
where it does not, because the standard tests assume independence that daily
market data does not have. Day-permutation nulls are used in preference to
parametric p-values throughout.

**Broker integration:** `ib_async` against Interactive Brokers, isolated behind
four adapter modules. Nothing else in the codebase knows the broker exists — which
is what makes the entire decision engine testable with no broker, no network and
no GUI, and is why the live rule engine imports nothing outside the standard
library.

**Interfaces:** PySide6 (Qt) for the trading desktop application, with
`lightweight-charts` for price rendering; Textual for the earlier terminal
interface, retained as a fallback. The presentation layer is a **pure function**
from state to display spans, carrying `(text, role)` pairs rather than rendered
strings — so colour is decided where the *state* is, never by pattern-matching on
formatted text, and two different front-ends can differ in layout but not in
fact.

**Architecture:** ports and adapters, enforced rather than aspirational. The
practical test is that the entire live decision engine — schedule, state machine,
trailing logic, order intent generation — imports only the Python standard
library. It has no dependency on the broker, the GUI, numpy, or the data layer.
That constraint was adopted for testability and paid off twice more: it is what
made the differential harness possible, and it is what makes the engine
publishable without publishing anything proprietary.

---

## Note on collaboration

*This program was carried out in sustained collaboration with an
AI coding assistant, with all research design, domain judgement, specification
and adjudication of results directed by the author.*

---

*This document describes a personal research programme. It is not investment
advice, not a solicitation, and contains no claim of profitability.*
