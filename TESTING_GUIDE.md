# Testing and reproducing results

This guide applies to the **binomial workload app**, launched with `app.py`, for both chain and star networks. Installation and assumptions are in [README.md](README.md). Older testing instructions for `quantum_network_waiting_time.py` describe a different generation model.

## Automated checks

Activate the environment used to run the app, then execute from this folder:

```sh
python -m unittest discover -s tests -v
```

Tests cover binomial tails, boundary/impossible demands, route-derived edge counts, numerical examples, segment waiting, schedule labeling, and Streamlit interactions. Passing these checks validates the cases implemented in the suite; sampled search does not establish global extrema.

## Five-node chain: resource calculation

Use N=5, L=100 km, alpha=0.21 dB/km, eta_d1=0.95, rho=0.9, and m=4000. Other physical defaults remain as described in the README.

```text
p = 0.95^2 * 0.9^2 * 10^(-0.21*100/10)
  = 0.005806737977893178
R = 10 requests
Model 1 demands: (10, 10, 10, 10)
Model 2 demands: (4, 6, 6, 4)
```

With B(K)=Pr[Binomial(4000,p)>=K]:

| Probability | Approximate value |
|---|---:|
| B(4) | 0.999999814973704 |
| B(6) | 0.9999944458561344 |
| B(10) | 0.9993200917967233 |
| Model 1 P_link = B(10)^4 | 0.9972831395808776 |
| Model 2 P_link = B(4)^2 B(6)^2 | 0.9999885216946702 |

These are probabilities of provisioning **all ten requests**, from one pool of 4000 modes on each of four physical edges. They are not probabilities for four separate pools per request.

Compare the same spanning path **1–2–3–4–5** in both models, using the same schedule:

```text
n_P = 4; number of swaps = 3
t0 = 0.5 ms
P_swap = (0.95*0.95)^6 = 0.5403600876626369
D_P = exp[-(2 ms + tau_P)/(200 ms)]
Model 1 F_FI(P) = [1 + 0.9972831395808776*P_swap*D_P]/2
Model 2 F_FI(P) = [1 + 0.9999885216946702*P_swap*D_P]/2
```

Take tau_P from that path's displayed schedule table. Do not invent a waiting count or compare the two models using different orders when checking the generation effect. Even for a shorter path, the app's complete-workload probability still includes all edge demands; only its swapping and memory factors use that shorter route.

Additional chain provisioning reference values:

| m | Model 1 | Model 2 |
|---:|---:|---:|
| 1000 | 0.0000249624735 | 0.1892530502 |
| 2000 | 0.2731554233 | 0.9438058644 |
| 4000 | 0.9972831396 | 0.9999885217 |

Small differences in the last printed digits can result from rounding.

## Five-node star

Select Star and use N=5, L=20 km per spoke, and the same efficiencies/attenuation. Set the **main** mode count to 50 when comparing provisioning tables with the m=50 waiting figure; the separate figure control already defaults to 50.

There are ten requests: four hub-to-leaf and six leaf-to-leaf. Model 1 requires (10,10,10,10) pairs; Model 2 requires (4,4,4,4), not the chain's (4,6,6,4).

```text
Model 1 P_link = B(10)^4 ≈ 0.7211299873
Model 2 P_link = B(4)^4 ≈ 0.9995518866
```

A hub-to-leaf path has one edge and no swap. A leaf-to-leaf path has two edges and one hub swap. Under uniform parameters, reorderings change individual requests' waits but not the average over all requests; best and worst averages coincide.

## Manual checks in the interface

1. For the five-node chain, inspect all 720 orders of the six non-direct requests. Confirm the average includes all ten requests, including four direct ones.
2. Sum the displayed path fidelities and divide by R. This should match the displayed average, allowing for rounded table entries.
3. At delta_t=0, additional schedule-dependent exposure vanishes. Both waiting prescriptions coincide; nonzero t0 still causes baseline decay.
4. For identical settings and the same schedule, Model 2's fidelity is at least Model 1's because its per-edge demands do not exceed Model 1's. A spanning path does **not** make them equal: their full-workload demands still differ.
5. Increasing m cannot reduce provisioning probability with fixed K and p. If any required K exceeds m, provisioning is zero, so Models 1/2 have F_FI=1/2 under the adopted convention.
6. With P_succ=1 and zero storage exposure, F_FI=1. The failure-inclusive score remains between 1/2 and 1.
7. Change N and inspect the regenerated demands: chain edge i needs i(N−i) pairs in Model 2, while every star spoke needs N−1. Model 1 always needs R per edge.
8. Change a sidebar physical parameter and check plotted-value tables and CSV exports. Fixed contour mode lists should stay fixed, while their calculated fidelities change.
9. Check Chain contours: three modes per row, maximum above minimum, Model 1 left/Model 2 right. Check Star contours: six distinct modes per model, three above and three below.
10. Check distance plots against the 99% single-request generation criterion, not a teleportation-fidelity threshold. Their independence from swapping/memory parameters is intentional.

## Reporting results

Record topology, N, edge length, all efficiencies, alpha, refractive index, T, modes, waiting interval/range, waiting prescription, figure-specific settings, and schedule-search seed/sample setting. Save plotted-value CSVs and PNGs together. For larger chains, report the displayed candidate count and state that extrema are sampled/observed if so labeled.

For environment reproducibility, retain:

```sh
python --version
python -m pip freeze > environment-used.txt
```

Dependency ranges allow newer installations to differ. Include the repository commit identifier when sharing numerical results. This simulator assumes adequate memory slots and does not explicitly simulate generation-round retries or asynchronous preparation-time decoherence; see the README before interpreting failure-inclusive fidelity as an experimentally conditioned output fidelity.
