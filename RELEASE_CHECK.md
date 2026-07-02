# Release check — Latent Wedge v0.1

| Field | Value |
|-------|-------|
| Repo | USinani/world-model-latent-audit |
| Branch | `public-v0.1-release-prep` |
| HEAD | `1773446` (local; not pushed) |
| Date | 2026-07-02 |

## Pass/fail table

| Item | Check | Status |
|------|-------|--------|
| 1 | critique issue template | PASS (on branch; not on `main` yet) |
| 2 | stale 0.08/0.15/0.73 grep | PASS (audit only — see hits table) |
| 3 | commit pointers (`583974e`, `a3c0221`, `d54b9a6`) | PASS |
| 4 | relative links | PASS |
| 5 | write-up link in README | **PENDING deploy** — line added locally; flip to PASS after https://sinani.ai/latent-audit returns HTTP 200 |
| 6 | test suite | PASS |

## Item 2 — Stale headline grep (report only)

**Canonical headline chain (do not change):** `0.992 / 0.716 / −0.175`

**Stale trio `0.08 / 0.15 / 0.73` as headline R² chain:** not found in scoped docs.

Grep scope: `README.md`, `FINDINGS_latent_v0.md`, `docs/`, `LOG.md`.

| file:line | match | review note |
|-----------|-------|-------------|
| `README.md:37` | `0.08s` | frame interval (`frame_stride=8`), not headline R² |
| `README.md:180` | `−0.084` | capacity-matched gap, not stale trio |
| `README.md:210` | `0.08s` | frame interval |
| `docs/verification_wedge_claim_packet_v0.md:114` | `0.08s` | frame interval |
| `FINDINGS_latent_v0.md:35` | `−0.152` | per-seed canonical delta (contains `0.15` substring) |
| `FINDINGS_latent_v0.md:69` | `0.08s` | frame interval |
| `FINDINGS_latent_v0.md:111` | `−0.152` | per-seed canonical delta |
| `LOG.md:98` | `\| 0.08 \|` | historical M-verify linear R² for `z` predictor — not headline chain |
| `LOG.md:158,176,221,294` | `0.08 s` | frame interval / window duration |
| `LOG.md:268` | `−0.152` | per-seed canonical delta |
| `LOG.md:273` | `−0.084` | capacity-matched gap |

**`0.73`:** zero hits in scoped files.

### Reconciliation (no doc edits needed)

- Stale headline trio `0.08 / 0.15 / 0.73` was **never** in the committed record as a headline chain.
- `LOG.md:98`'s `0.08` is a historical M-verify **linear-probe** R² — not the headline chain.
- `0.73` appears **nowhere** in scoped files.
- Canonical chain `0.992 / 0.716 / −0.175` stands; old brief numbers were garbled memory of intermediate diagnostics.
- Nothing public needs scrubbing beyond never quoting the old pair again.

**Note:** `metrics.json` stores `delta_gate.mean ≈ −0.175` but not explicit `0.992` / `0.716` fields — those come from Branch C addendum analysis documented in FINDINGS (commits `a3c0221`, `d54b9a6`).

## Item 3 — Commit pointers

| SHA | subject |
|-----|---------|
| `583974e` | LW-11 terminal (48px): VOID-FIDELITY |
| `a3c0221` | Raw-observation consequence ceiling (Branch C) |
| `d54b9a6` | Branch C final verification: matched-capacity AND matched-probe |

All three resolve in repo git history. Source: `docs/verification_wedge_claim_packet_v0.md` §8.

## Item 4 — Relative links

Markdown links checked in README + claim packet; explicit paths verified on disk:

- `REPRODUCING.md` — OK
- `FINDINGS_latent_v0.md` — OK
- `LOG.md` — OK
- `docs/verification_wedge_claim_packet_v0.md` — OK
- `results/20260619_103342_nogit_failuremodes/metrics.json` — OK
- `results/20260619_193900_33faabb_latent/metrics.json` — OK
- `.github/ISSUE_TEMPLATE/critique.md` — OK

**Non-link note:** README line 102 backtick reference `World_Models/src/world_models/models/mlp.py` is provenance prose pointing at the sibling monorepo path — not a relative link, does not resolve inside `Latent_Wedge/`. No fix required.

## Item 6 — Test suite

Re-run from repo root on 2026-07-02:

```bash
python3 tests/test_invariants.py   # 4/4 PASS
python3 tests/test_numpy_swaps.py  # 7/7 PASS
```

## Pending items

| Item | Gate | Action when cleared |
|------|------|---------------------|
| 5 | `curl -sI https://sinani.ai/latent-audit` → 200 | Add `**Write-up:** https://sinani.ai/latent-audit` to README (plain line, no YAML front matter); flip item 5 to PASS |
| merge | Item 5 PASS + README committed | Merge `public-v0.1-release-prep` → `main` and push |

## Commits applied

| When | Files | Message |
|------|-------|---------|
| 2026-07-02 | `RELEASE_CHECK.md` | Add v0.1 release readiness check |
| pending | `README.md`, `RELEASE_CHECK.md` | Add public write-up link (local commit; push after sinani.ai verified live) |

## Local verification (2026-07-02)

Sinani page served locally from `/Users/uljan/Desktop/Sinani`:

```bash
cd /Users/uljan/Desktop/Sinani && python3 -m http.server 8765
# http://localhost:8765/latent-audit/ → 200
# card 003 link → /latent-audit
```

Outbound links on page: all return 200 (github, arxiv×2, permissioning.ai).

**No remote pushes performed** — deploy when locally verified and approved.
