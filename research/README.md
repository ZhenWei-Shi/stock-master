# Offline daily factor lab

This is a local research tool. It does not import the scheduler, connect to a
broker, submit orders, or modify existing forward ledgers. Use Python 3.11+ with
`numpy`, `pandas`, and (only for acquisition) `yfinance` installed. Tests use pytest.

## Registered first batch

`factor_lab_v1.toml` fixes the universe, four candidate formulas, signal timing,
time partitions, comparison groups, cost scenarios and random seed before the
first run. Keep the registration file and its hash with every output. A changed
formula, universe, threshold or hypothesis count is a new experiment version.

- 12-minus-1-month momentum: adjusted close at t-21 / t-252 minus one.
- Five-session reversal: negative adjusted close return over five sessions.
- Signed relative volume: daily return sign times (volume / prior-20 mean - 1).
  Suppress this factor around splits because the volume field is not adjusted.
- Low volatility: negative annualized sample standard deviation of 20 daily
  adjusted-close returns.

Signals use close-t information. Entry is the next session adjusted open and exit
is five sessions after entry. Missing sessions are not compressed or forward
filled. A five-session label must finish inside the selected evaluation period.
Ranking uses signal-time availability only. If future outcomes are missing, the
entire date comparison is excluded and counted, rather than replacing selected
members using future knowledge of which prices are available.

Top quintile is compared with the same eligible universe equalweight and one
equal-size, deterministic random group on the same dates. Fees are flat per
cohort scenarios (10bp roundtrip; 20bp stress), not measured fills or a full
turnover-aware execution simulation. Top-minus-bottom is a gross diagnostic,
not an implemented short strategy. Overlapping cohort returns do not constitute
a daily self-financing portfolio or tradable NAV.

Date-level Newey-West statistics and a four-hypothesis Holm correction address
some dependence and multiple testing. Their normal approximation, arbitrary
fixed lag, cross-factor dependence, selection bias and nonstationarity remain
limitations. A significant statistic alone cannot qualify a trading strategy.

## Run explicitly

```bash
python -m src.factor_lab fetch --cache data/factor-cache-v1 --output data/cache-quality-v1.json
python -m src.factor_lab evaluate --cache data/factor-cache-v1 --output data/factor-results-v1.json
python -m src.factor_lab h4 --h4-input /private/read-only-audit-snapshot.json --output data/h4-supplement-v1.json
python -m pytest tests/test_factor*.py tests/test_h4_validation.py
```

Output paths are immutable: rerunning must use a new output path. Final JSON is
staged in a unique same-directory file, fsynced, and published with an atomic
hard link that refuses an existing target, even if another run finishes after
the startup check. This requires a filesystem supporting hard links; unsupported
publication fails without a partial result or overwrite fallback. Request/status
metadata remains explicitly replaceable and uses separate unique temporary files.
A complete,
matching cache is verified and reused without downloading. Acquisition staging
is bound to the requested universe and dates and verifies SHA256 before reuse.
A failed ticker blocks publication of a complete cache; it is not silently
removed from the universe. Cache manifest publication is the final atomic step.
An exclusive writer lock blocks concurrent publication into the same directory;
after a crash inspect the previous process before removing a stale lock.
Keep all prices, provenance, local evidence, and results under ignored `data/`.

The CLI runs development and validation only and refuses to unlock the reserved
historical period. It crops the input before calculating features and labels.
These are historical, retrospectively designed experiments: current survivor
selection, revised data, model pretraining and the analyst's historical knowledge
remain sources of hindsight. Even an untouched historical test is not new live
evidence. This batch contains no point-in-time index membership or delisted
stocks, and reports exploratory findings only.

## H4 supplemental evidence

The new H4 reader computes a separately versioned comparison on an immutable
copy of v2 forward records. It preserves the original strategy's registered
mean/t gates and all files. Revision `supplemental_v2026-10-22-r2` requires >=200
evaluated signal records over >=30 evaluated dates, positive net mean under an
additional 10bp stress and positive date-level excess over controls with HAC t>2.
An evaluated date has completed records in both groups and no registered pending
record in either group; partial dates are deferred in full. Newly pending dates
do not invalidate older complete samples. Ledger totals, evaluated counts and
mutually exclusive exclusion reasons are reported separately. Explicit statuses
are limited to completed/pending/open; unknown statuses fail instead of silently
dropping possible outcomes. Records without status retain the original ret/None
completion protocol. The same evaluated subset supplies all gates and statistics.
This is new
supporting research, not a retrospective replacement of the old acceptance rule.
Unmatched, unfinished and old-pool records are reported separately.

## Recovery

Stop the local process to halt research. No operational service needs rolling
back. Failed acquisition can resume validated staging; a corrupt cache is rejected
and must be inspected or acquired into a new directory. To retire an experiment,
keep its registration and results and use a new version rather than overwriting
the historical evidence. Price-data use must satisfy provider terms; no data is
redistributed with the repository.
