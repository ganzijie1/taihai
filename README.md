# Taiwan Strait Coupled Model

This repository contains the reproducible code path for the Taiwan Strait
evolution and campaign model. V8.3 is the current frozen source specification
and formal-result baseline. Its nine registered variants completed all 270
checkpoint cells, and the independent result audit passed.

The model is a monthly, same-path coupled system. Its active layers include
campaign and alliance intervention, continuous multi-population MFG, competing
risks, social contagion, finance, energy, trade, 140-node MR-CGE, open-economy
DSGE, disequilibrium inventories and backlogs, non-financial SFC, and a
four-party social accounting matrix (SAM). Later source versions add
econometric calibration, information/network/Riemannian geometry, three-sector
structural transformation, continuous state-action MFG, causal optimal
transport, stochastic geometry, symplectic/port-Hamiltonian dynamics and a
  hybrid large-deviation importance sampler. V6.8--V7.7 additionally activate
functional-analytic checks, martingale optimal transport, Tinbergen
reachability and tropical bottlenecks, combinatorial/continuous/Finsler
networks, hypergraph and state-action MFG, exact finite-complex online TDA,
information-bottleneck active observation, FGW/Ollivier/Hodge network
geometry, nested CES production and a signed inertial coupled oscillator.

## Current integrity status

- V5.3 formal-result baseline: retained as historical comparison
- V7.7 traceability and equation-family coverage: 100%
- V8.3 pre-simulation integrity, code-coupling and runtime audits: PASS
- V8.3 formal checkpoint design: nine variants and 270 validated cells
- V8.3 runtime invariants and all registered matched-row ablations: PASS
- V8.3 publication gate: PASS

Passing diagnostics establishes accounting and software consistency. It does
not establish empirical truth or classified-data-level predictive accuracy.

## V7.9 information, criticality and scale extension

V7.9 is implemented and frozen for a future complete run, but no V7.9 outcome
simulation has been executed. It extends the V7.8 mixed potential/non-potential
MFG with correlation-adjusted source exposure, asymmetric rate-distortion
attention, lagged conditional-transfer-entropy Graphon calibration and one
four-domain self-organized quasi-critical DEDS cascade. The cascade feeds the
existing social, firm, financial, network and command states on the same path.
A 16-to-8-to-4-node renormalization audit is read-only and never adjusts an
outcome probability.

The non-outcome preparation command is:

```powershell
python work/prepare_taiwan_v79.py
```

It writes only traceability, code-audit, deterministic-diagnostic and frozen
specification records. The complete V7.9 runner remains protected by both the
complete-model gate and `TAIWAN_ALLOW_COMPLETE_RUN=1`.

## V8.0 statistical-field and analytical-mechanics extension

V8.0 is implemented, frozen and has completed one formal 128-path, 360-month,
30-cell checkpointed run. It adds a non-equilibrium four-domain residual
field, self-consistently fitted DMFT memory and colored noise, event-tail and
finite-resolution criticality diagnostics, and an analytical-mechanics
constraint layer. The mechanics layer uses a discrete Lagrange-d'Alembert
step, a skew Dirac interconnection, dissipative adjustment, and read-only
Noether/PMP checks. It owns only residual adjustment coordinates and does not
replace CGE, DSGE, logistics, command, social or continuous-MFG state owners.

The residual field affects the current outer consumers through the existing
cascade contract and affects MFG optimization with one explicit monthly lag.
SOC evidence and Noether/PMP quantities are diagnostic only. No component
patches terminal probabilities.

```powershell
python work/prepare_taiwan_v80.py
```

The preparation command runs diagnostics, coupling audits and manifest
validation, then freezes the V8.0 specification without producing outcome
probabilities. The completed run is stored as
`outputs/taiwan_v80_simulation.json` and
`outputs/taiwan_v80_results.csv`; its artifact-integrity audit is PASS.
Publication validation remains blocked until one-at-a-time outcome ablations
for the V8.0 statistical-field and analytical-mechanics additions are run.
The checkpoint aggregate rows also omit case labels, so interpretation uses
the correctly labelled conditional rows rather than the aggregate array.

## V8.1--V8.3 completed extensions

V8.1 adds differential-manifold state handling and its inherited ablations.
V8.2 adds actor-specific agriculture, industrial organization, development and
labor mechanisms without duplicating the CGE, DQ-SFC or structural-employment
state owners. V8.3 adds particle intent and structural international-economics
channels, including gravity/Eaton-Kortum, Melitz and external-liquidity
ablations. All layers execute on the same monthly paths and feed the existing
economic, social, MFG, DEDS and campaign consumers.

The formal V8.3 run is checkpointed and guarded:

```powershell
python work/prepare_taiwan_v83.py
$env:TAIWAN_ALLOW_COMPLETE_RUN='1'
python work/run_taiwan_v83_formal.py
python work/audit_taiwan_v83_results.py
```

The final audit reports all nine variants present, all 270 checkpoints valid,
runtime invariants satisfied and the publication gate passed. Generated run
outputs and checkpoints remain under `outputs/` and are intentionally excluded
from Git; the repository publishes the code, compact public priors and methods
documentation needed to rebuild them.

## Layout

- `work/`: executable dependency tree, preparation gates, checkpointed runners,
  audits and public GIS cache through V8.3
- `src/`: historical source mirror, diagnostics, audits, ablations and
  sensitivity scripts retained for version comparison
- `data/`: compact aggregated IO/MRIO and V8.2--V8.3 structural-prior inputs
- `artifacts/`: historical baselines plus V6.8--V7.7 traceability, audits,
  diagnostics, frozen specifications and the audited V7.7 formal result
- `work/gis_data/`: cached aggregate public terrain and road-network inputs
- `requirements-frozen-v77.txt`: exact core V7.7 numerical environment
- `requirements-wsl-jax-cuda.txt`: frozen WSL2 GPU environment
- `requirements-windows-cuda.txt`: Windows PyTorch CUDA environment

## Prepare the artifact layout

The research workspace stores generated records under `outputs/`, while this
repository keeps published records under `artifacts/`:

```powershell
New-Item -ItemType Directory -Force outputs
Copy-Item artifacts/*.json outputs/
```

## Run the V7.7 integrity gate

```powershell
python work/prepare_taiwan_v77.py
```

This command rebuilds diagnostic, traceability, static-coupling and frozen
specification artifacts. It does not produce substantive outcome probabilities.

## Reproduce the complete V7.7 simulation

Use the checkpointed entry point. Each completed scenario is written
atomically and is reusable only when paths, months, scenario and frozen-spec
hash all match:

```powershell
$env:TAIWAN_ALLOW_COMPLETE_RUN='1'
$env:TAIWAN_V77_PATHS='128'
$env:TAIWAN_V77_MONTHS='360'
python work/simulate_taiwan_v77_checkpointed.py
python work/audit_taiwan_v77_results.py
```

The checkpointed runner accepts a checkpoint only when its scenario, case,
path count, horizon and frozen-specification hash match. The result audit checks
150 conditional rows, 30 diagnostic cells, probability bounds, the complete
model gate and registered solver residuals.

## Result interpretation

`artifacts/taiwan_v77_result_summary.json` reports conditional broad-control
rates by intervention regime and horizon. Registered trade scenarios are
equal-weight sensitivity cases, not estimated scenario probabilities. A zero
Monte Carlo estimate means no qualifying terminal event was observed in the
finite weighted sample; it is not a proof of zero real-world probability.

The model is a public-data conditional mechanism study, not a classified
operational forecast. Conflict onset, intervention regime, conditional campaign
outcome and unconditional joint outcome must not be conflated.

## Reproduce the V5.3 formal baseline

The complete run is intentionally guarded. It can take hours on a CPU:

```powershell
$env:TAIWAN_ALLOW_COMPLETE_RUN='1'
$env:TAIWAN_V53_PATHS='128'
$env:TAIWAN_V53_MONTHS='360'
$env:TAIWAN_EQUILIBRIUM_SOLVER='picard'
python src/simulate_taiwan_fully_closed_v53.py
```

The code resolves input data relative to the repository root. Optional JAX,
PyTorch and GeomLoss paths degrade to documented CPU fallbacks when unavailable.
GPU execution must be benchmarked against identical seeds and error tolerances;
device discovery alone is not evidence of a valid speedup.

## Data and identification boundary

The compact IO files are derived from OECD ICIO 2022 and CEADS China MRIO.
Production, intermediate use, value added, final demand and trade control totals
are observed or derived public accounts. Wartime adjustment speeds, household
group splits where harmonized microdata are unavailable, and rare-event
parameters are scenario priors and require sensitivity analysis.

## Reproducibility

The accepted formal baseline is frozen in
`artifacts/taiwan_v77_frozen_specification.json`. Any source change requires a
new diagnostic pass, manifest audit and freeze before an outcome run can be
treated as comparable. The full result, compact summary and independent audit
are stored as `taiwan_v77_simulation.json`, `taiwan_v77_result_summary.json`
and `taiwan_v77_result_audit.json` under `artifacts/`.
