# AGENTS.md — Rules for implementing agents

## Core Principles

1. **No LLMs at runtime.** "Agents" are deterministic code + classical ML models
   communicating through typed messages on a Blackboard. Never call an LLM.

2. **Paper trading only.** Never place real orders, never store or request API keys
   with trading permissions. Market-data endpoints must be public/read-only.

3. **No lookahead leakage.** The single most important invariant.
   - No agent may read a bar before its `close_ts`.
   - All downstream logic uses `confirm_idx` / `confirm_ts` of pivots.
   - Once a pivot is emitted it never changes or disappears (no repaint).
   - Higher-timeframe bars are available only after their `close_ts`.
   - Features declare an `available_at` rule.
   - Labels (future-looking) live in `research/labels.py` only — never imported
     by feature or agent code.
   - Fitted parameters come from the training slice only.

4. **Same code live and in replay.** Live trading and backtest call the same
   agent entry points with the same event types.

5. **Determinism.** Every run takes a `seed` and `config_hash`. Same inputs +
   config → byte-identical outputs.

6. **Pure functions + explicit state.** No global state. Anything stateful must
   expose `snapshot()` / `restore()` for replayability.

7. **Time convention.** All timestamps are UTC integer milliseconds. Bars carry
   both `open_ts` and `close_ts`. A bar is "known" only at `close_ts`.

8. **Ambiguity resolution.** When a requirement is ambiguous, choose the simplest
   option, document in `docs/DECISIONS.md`, and continue.

9. **No parameter tuning to make backtests look good.** Parameters tuned must be
   logged as trials in `runs/trials.jsonl`.

10. **Honest reporting.** If wave features don't add value, say so. Never hide
    negative results.
