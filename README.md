# Taiwan Strait Coupled Model

This repository contains the reproducible code path for the Taiwan Strait
evolution and campaign model. The current frozen baseline is V5.3.

The model is a monthly, same-path coupled system. Its active layers include
campaign and alliance intervention, continuous multi-population MFG, competing
risks, social contagion, finance, energy, trade, 140-node MR-CGE, open-economy
DSGE, disequilibrium inventories and backlogs, non-financial SFC, and a
four-party social accounting matrix (SAM).

## Current integrity status

- V5.3 diagnostic gate: PASS
- Static coupling audit: PASS
- Active modules in the traceability manifest: 47
- SAM accounts per actor: 19
- Formal configuration: 128 paths, 360 months, all intervention and trade cases
- Superseded pre-SAM freeze is retained in `artifacts/` as a rejected audit record

Passing diagnostics establishes accounting and software consistency. It does
not establish empirical truth or classified-data-level predictive accuracy.

## Layout

- `src/`: current and historical Taiwan-model source, diagnostics, audits,
  ablations and sensitivity scripts
- `data/`: compact aggregated IO/MRIO inputs required by the public code path
- `artifacts/`: V5.3 diagnostic, traceability, coupling audit and frozen spec

## Run the V5.3 diagnostics

```powershell
python src/diagnostic_taiwan_v53.py
python src/audit_taiwan_v53.py
```

The audit scripts expect `artifacts/` to be available as `outputs/` because the
research workspace keeps generated artifacts under that name. For an exact
workspace-style run, create a local `outputs` directory and copy the JSON files:

```powershell
New-Item -ItemType Directory -Force outputs
Copy-Item artifacts/*.json outputs/
```

## Run the complete V5.3 simulation

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

## Data and identification boundary

The compact IO files are derived from OECD ICIO 2022 and CEADS China MRIO.
Production, intermediate use, value added, final demand and trade control totals
are observed or derived public accounts. Wartime adjustment speeds, household
group splits where harmonized microdata are unavailable, and rare-event
parameters are scenario priors and require sensitivity analysis.

## Reproducibility

The frozen specification in `artifacts/taiwan_v53_frozen_specification.json`
contains SHA-256 hashes for the accepted V5.3 baseline. Any source change after
that freeze requires a new diagnostic pass, manifest audit and freeze before a
new outcome run can be treated as comparable.

