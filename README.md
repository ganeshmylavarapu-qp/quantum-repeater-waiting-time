# Quantum Network Teleportation Simulator

A local Streamlit app for chain and star quantum networks. It compares deterministic Model 0 with Models 1 and 2, which provision concurrent communication requests from binomially distributed elementary-pair resources. It includes explanatory text, worked calculations, schedule details, comparison figures, and data exports.

This is the updated binomial app. The original waiting-network app uses different generation equations; its testing guide is not the guide for this version.

## Install and run

Use Python 3.11 or later and upload/download this entire folder, including `.streamlit` and `tests`. No external figure files or original app folder are required to run it. The dependency ranges in `requirements.txt` are not a locked environment; retain your installed versions when reporting results.

Open a terminal in this folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows, replace the activation command with:

```bat
.venv\Scripts\activate
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`. Keep the terminal running; press Ctrl+C to stop. If the browser reports connection refused, start the command again. If the port is occupied, use `python -m streamlit run app.py --server.port 8511` and open the URL it prints.

For numerical checks and a reproducible walkthrough, see [TESTING_GUIDE.md](TESTING_GUIDE.md).

## Using the app

1. Select Chain or Star. The dashboard shows the selected topology's figures and tables.
2. Set the number of nodes, physical parameters, modes per edge, and waiting prescription.
3. Select Model 0, 1, or 2 for worked calculations. The paper dashboard still compares models.
4. Inspect provisioning, path calculations, maximum/minimum schedule details, and numerical values behind the figures.
5. Download CSV data and PNG figures. Record sidebar settings, figure-specific controls, waiting prescription, and search method with each result.

Sidebar changes recompute the results. Some figures have their own mode settings, documented below; changing the main mode count does not override those settings.

## Physical parameters and defaults

Both topologies support 2–50 nodes. Each physical edge has **one shared pool of m modes per generation round**, not m modes for every request.

| Setting | Default |
|---|---|
| Nodes | 5 |
| Chain adjacent-edge length | 100 km |
| Star spoke length | 20 km |
| Chain main mode count | 4000 |
| Star main mode count | 10 |
| Generation and swapping detector efficiencies | 0.95 each |
| Memory retrieval efficiency | 0.95 |
| Source efficiency | 0.9 |
| Fiber attenuation | 0.21 dB/km |
| Fiber refractive index | 1.5 |
| Memory coherence time | 200 ms |
| Path-table swapping interval | 0.1 ms |
| Waiting-plot maximum interval | 200 ms |
| Larger-chain sampled-order setting / seed | 300 / 7 |

Changing node count keeps the chosen edge length fixed. Total network fiber length is `(N−1)L` for either uniform tree; this is not the length of every communication path.

## Requests, resource demand, and fidelity

All `R = N(N−1)/2` unordered node pairs submit one request. R counts requests, K counts required elementary pairs on an edge, and X counts randomly generated pairs. Requests sharing an edge use distinct pairs; consumed pairs cannot be reused. Sufficient memory capacity is assumed.

For physical edge e:

```text
p_e = eta_d1^2 * rho^2 * 10^(-alpha*L_e/10)
X_e ~ Binomial(m, p_e)
B_e(K) = Pr(X_e >= K)
       = sum from x=K to m of binom(m,x)*p_e^x*(1-p_e)^(m-x)
B_e(0) = 1; B_e(K) = 0 for K > m
p_elem = B_e(1) = 1-(1-p_e)^m
```

Modes are independent with identical success probability within an edge, and each yields at most one pair. Generation is also independent across physical edges.

- **Model 0:** generation and swapping succeed deterministically; memory decay remains.
- **Model 1:** every request reserves one pair on every physical edge. Each edge needs K=R pairs, even when the request does not traverse that edge.
- **Model 2:** each request reserves pairs only on its route. Chain edge `(i,i+1)` needs `i(N−i)` pairs, using nodes 1 through N. Star spoke `(0,i)` needs `N−1` pairs, with hub 0 included in N.

Both probabilistic models use complete-workload admission: all edge demands must be met before the workload enters the prescribed swapping schedule.

```text
P_prov = P_link = product over physical edges e of B_e(K_e)
P_swap(P) = (eta_mem*eta_d2)^[2(n_P-1)]
P_succ(P) = P_prov * P_swap(P)
t0 = L*fiber_index/300 ms  (L in km)
tau_P = delta_t * sum of stored-segment interval counts ell_s
D_P = exp[-(n_P*t0 + tau_P)/T]
w_eff = P_succ(P)*D_P
F_FI(P) = (1+w_eff)/2
F_avg,FI = mean over all R requests
```

For Model 0 set P_succ=1. Each direct request has zero swaps. At 100 km, t0=0.5 ms; at 20 km, t0=0.1 ms. Times in these formulas use milliseconds.

Provisioning is common to every path within a model/workload. It is **not** a route-only product and is not `p_elem` raised to the number of required pairs. The selected path determines its swapping contribution and memory exposure.

Under the failure-inclusive convention, failed provisioning or swapping is assigned an effective uncorrelated resource `I_4/4`, giving fidelity 1/2. This is a coarse-grained scoring convention, not a claim that physical loss produces a two-qubit state. Conditional fidelity after successful generation and swapping is `(1+D_P)/2` and is a different quantity.

Generation retries, random request arrivals, replenishment delays, and generation-induced memory exposure beyond baseline storage are not explicitly modeled. Successful swaps are ideal apart from their success probability; heralded failures do not alter the prescribed schedule.

## Waiting and schedule search

The default repeater-induced prescription tracks elementary and intermediate entangled segments as swaps proceed. A segment's dimensionless interval count ell_s gives storage duration `ell_s*delta_t`; tau sums that exposure across segments. Repeater contention is local: the scheduler is not a global queue that blocks all nodes whenever any request is served.

The alternate maximum-wait prescription assigns `n_P` times the largest required repeater waiting count times delta_t. Compare prescriptions on the same order when diagnosing their exposure differences; independently optimized orders may differ.

For a five-node chain, six requests require swapping and all `6! = 720` orders are enumerated. Four direct requests remain in the ten-request fidelity average. Larger chains use reproducible candidate/sampled orders when exhaustive search is too large. Report their extrema as **best/worst observed**, not guaranteed global optima.

For a uniform star, the hub serves equal-length leaf-to-leaf requests. Changing their order permutes the waiting times but leaves the workload average unchanged. Thus the maximum and minimum average fidelities coincide under these assumptions.

## Figures and their controls

- **Chain fidelity versus waiting:** Model 0/1/2 panels; editable mode sweep defaults to 1000, 2000, 3000, 4000, 5000, 8000. Orange solid and blue dashed curves distinguish best and worst schedules.
- **Star fidelity versus waiting:** one panel overlays Models 0, 1, and 2. Its separate sidebar control, **Star model-comparison modes**, defaults to 50; it does not change other figures or the main Star mode count.
- **Chain distance–waiting contours:** fixed modes 1000, 3000, 5000. Inter City Model 1 is left and Inter City Model 2 right; maximum average fidelity is the top row and minimum the bottom row.
- **Star distance–waiting contours:** fixed modes 35, 50, 65, 80, 95, 110, arranged three above and three below within each model group. These are six different modes, not separate maximum/minimum rows.
- Contours use a shared fidelity color scale, solid black group borders, and no selected-setting circle markers. The dashed fidelity boundary marks 2/3 where crossed.
- **Schedule gap:** maximum minus minimum average fidelity at the main selected mode count; zero for the uniform star.
- **Critical distance versus modes:** dashed reference curves at 99% **single-request generation availability**, with marked configured/current mode counts and corresponding distances. These curves exclude swapping and memory decay; they are not a 99% full-workload criterion or a fidelity=2/3 boundary.

The editable Star sweep defaults to 35, 50, 65, 80. Fixed contour modes remain independent of the editable sweeps. Figure-specific controls and the main mode count can therefore intentionally differ.

## Source layout

| File | Purpose |
|---|---|
| `app.py` | Streamlit entry point, controls, tables, and exports |
| `model.py` | Binomial provisioning, topology paths, scheduling, and fidelity |
| `paper_dashboard.py` | Explanations and paper-style figures |
| `schedule_details.py` | Detailed schedule and path calculations |
| `tests/` | Automated numerical and interaction checks |
| `.streamlit/config.toml` | App theme |
| `requirements.txt` | Python dependency ranges |
| `TESTING_GUIDE.md` | Numerical examples, checks, and reporting guidance |

For GitHub, include these source files, documentation, tests, and configuration. A minimal `.gitignore` is provided. Existing PNG/CSV files are snapshots; run the app to regenerate results for the current controls. No software license is asserted by this README; repository owners should choose a license before advertising reuse permissions.
