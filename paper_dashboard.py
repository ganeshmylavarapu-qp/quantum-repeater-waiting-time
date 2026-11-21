"""Paper-style figures for the binomial workload model."""
import io
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from matplotlib import patheffects
import numpy as np
import pandas as pd
import streamlit as st

from model import fidelities, paths, provisioning, schedules, single_probability


def explain():
    st.header('Introduction')
    st.markdown(
        'A quantum network connects remote users by distributing entangled pairs. '
        'Neighboring nodes first establish **elementary entangled links**. To connect '
        'more distant users, intermediate repeater nodes perform **entanglement swapping**: '
        'two shorter entangled pairs are consumed to create one pair shared across a longer '
        'distance. The resulting end-to-end pair is then used as a resource for quantum teleportation.'
    )
    st.markdown(
        'This work studies a limitation that remains even after the required elementary pairs '
        'have been generated. Several communication requests may need the same repeater, '
        'but a repeater can perform only one swapping operation at a time. While it serves '
        'one request, the entangled pairs reserved for other requests remain stored in quantum '
        'memories. This **repeater-contention-induced waiting** allows memory noise to reduce '
        'their correlations before they are used.'
    )
    st.header('Problem statement')
    st.markdown(
        'Given a set of communication requests that are ready to be served, how does the '
        'order of service at shared repeaters affect the quality of the final teleportation '
        'resources? We calculate the storage exposure of each elementary and intermediate '
        'entangled segment, obtain the teleportation fidelity for every source–target path, '
        'and average over all unordered source–target pairs.'
    )
    st.markdown(
        'We compare the execution orders that give the **maximum and minimum network-average '
        'fidelity**, under both the repeater-induced and maximum-waiting prescriptions. '
        'Probabilistic generation and imperfect swapping are included through an explicitly '
        'defined failure-inclusive effective state. The result therefore accounts for resource '
        'availability as well as the degradation of successfully established resources.'
    )
    st.header('Starting resources and scheduling assumptions')
    st.markdown(
        'To isolate repeater-contention waiting from generation delays, we consider requests '
        'whose required elementary entangled pairs coexist in memory when scheduling begins. '
        'The waiting calculation describes the branch on which the complete initial resource '
        'allocation has been established. Generation and swapping probabilities then weight '
        'the effective failure-inclusive fidelity; they do not create random retry times in '
        'the scheduling calculation.'
    )
    st.markdown(
        '**Separate pairs for separate requests.** An entangled pair is consumed when it is '
        'used in swapping and cannot be reused for another communication. If several requests '
        'traverse the same physical edge, they require distinct stored pairs on that edge. '
        'One physical edge can therefore support several entangled pairs without becoming '
        'several graph edges. We assume enough memory slots to hold the allocated resources.'
    )
    st.markdown(
        '**Why ready resources still have to wait.** Each repeater can serve at most one '
        'swapping operation at a time. Requests that need that same repeater must be ordered. '
        'A waiting request retains its own unused pairs while earlier requests consume theirs. '
        'Independent generation of the pairs does not give them independent access to a shared repeater.'
    )
    st.markdown(
        '**Example.** Suppose requests **1–2–3** and **1–2–3–4** are ready together. '
        'Each has its own pairs on edges 1–2 and 2–3; the longer request also has a pair '
        'on edge 3–4. If the shorter request is served first at repeater 2, the longer '
        'request must keep its separate resources stored while it waits for that repeater. '
        'Any swaps at other repeaters follow the selected waiting prescription.'
    )
    st.markdown(
        '**Scope of the calculation.** We consider one request per unordered node pair. '
        'Resource replenishment, temporal retries, stochastic arrival queues, and memory-capacity '
        'blocking are excluded. A swapping failure does not change the prescribed schedule. '
        'The binomial section below adds an explicit calculation of whether one generation '
        'round can supply enough pairs for this set of requests.'
    )
    st.subheader('Assumption summary')
    st.table({
        'Assumption': [
            'Concurrent workload', 'Separate pairs', 'Memory capacity', 'Shared repeater capacity',
            'Initial resources', 'Independent generation', 'Successful-state quality',
            'Memory noise', 'Failure handling', 'Prescribed schedule', 'Excluded delays'],
        'Meaning': [
            'Every unordered node pair submits one request: R = N(N−1)/2.',
            'Each request uses its own elementary pairs; consumed pairs cannot be reused.',
            'Enough memory slots are assumed for all allocated pairs.',
            'A repeater serves at most one swap at a time; competing requests are ordered.',
            'Scheduling is evaluated on the branch where the complete initial allocation is ready.',
            'Generation modes are independent within each edge; different edges generate independently.',
            'Generation success produces ideal Bell pairs; successful swaps are ideal apart from the modeled storage noise.',
            'Storage preserves a Werner-like form and multiplies retained correlations by an exponential factor.',
            'Failed provisioning or swapping receives an effective maximally mixed replacement.',
            'A heralded swap failure does not alter the remaining prescribed schedule.',
            'Retry times, replenishment, arrival queues, and storage-capacity blocking are not simulated.',
        ],
    })
    st.markdown(
        'As in the original app, the schedule uses per-repeater contention counters, not a '
        'separate global serial service-time penalty. Requests need not conflict if they use '
        'different repeaters. Direct paths require no swap. The detailed tables distinguish '
        'elapsed repeater waits from accumulated exposure across stored pairs.'
    )
    st.header('Failure-inclusive average teleportation fidelity')
    st.markdown(
        'We use **failure-inclusive** (FI), rather than “failure-induced,” to emphasize that '
        'the reported score averages successful and unsuccessful attempts. Ordinary heralded '
        'fidelity describes only successful resources. Our effective network metric also '
        'assigns a contribution to attempts that fail to establish the required resource.'
    )
    st.subheader('1. Start with a Werner resource')
    st.latex(r'\rho_W(w)=w|\psi^-\rangle\langle\psi^-|+(1-w)\frac{I_4}{4},\qquad 0\le w\le1')
    st.markdown(
        'The singlet Bell component has weight $w$ and the uncorrelated maximally mixed '
        'component has weight $1-w$. At $w=1$ the resource is a perfect Bell pair; at $w=0$ '
        'it is maximally mixed. A locally equivalent Bell choice has the same teleportation relation.'
    )
    st.subheader('2. State the effective failure convention')
    st.markdown(
        'Write $q$ for the probability that the resource-establishment event succeeds. '
        'On success, suppose the stored resource has Werner parameter $D$. '
        'On failure, assign $I_4/4$ within the effective two-qubit description:'
    )
    st.latex(r'\rho_{\rm eff}=q\,\rho_W(D)+(1-q)\frac{I_4}{4}')
    st.markdown(
        '**This is a modeling convention.** Photon loss does not physically produce $I_4/4$, '
        'and a failed event need not produce any two-qubit resource at all. The replacement '
        'defines how unsuccessful attempts are scored. A different failure policy would define '
        'a different metric.'
    )
    st.subheader('3. Collect the Bell and mixed components')
    st.latex(r'\rho_{\rm eff}=qD|\psi^-\rangle\langle\psi^-|+(1-qD)\frac{I_4}{4}=\rho_W(qD)')
    st.markdown(
        'The effective Werner parameter is therefore **$w_{\rm eff}=qD$**. This form follows '
        'from the assumed Werner successful state and maximally mixed failure contribution; '
        'probabilistic generation alone does not imply a Werner state.'
    )
    st.subheader('4. Calculate teleportation fidelity')
    st.latex(r'F(\rho_W(w))=\frac{1+w}{2},\qquad F_{\rm FI}=\frac{1+qD}{2}')
    st.latex(r'F_{\rm FI}=qF_{\rm cond}+(1-q)\frac12,\qquad F_{\rm cond}=\frac{1+D}{2}')
    st.markdown(
        'These fidelities average over input qubit states for the standard teleportation '
        'protocol. A successful stored resource contributes $(1+D)/2$; the assigned failure '
        'contributes $1/2$. Low FI fidelity can therefore reflect low availability even when '
        'successfully established pairs have high conditional quality.'
    )
    st.table({
        'Worked example': ['Establishment success q', 'Retained memory parameter D',
                           'Conditional fidelity', 'Effective Werner parameter qD', 'Failure-inclusive fidelity'],
        'Calculation': ['0.8', '0.9', '(1 + 0.9)/2', '0.8 × 0.9',
                        '0.8 × 0.95 + 0.2 × 0.5 = (1 + 0.72)/2'],
        'Value': [.8, .9, .95, .72, .86],
    })
    st.subheader('5. Include swapping and storage')
    st.latex(r'P_{\rm swap}(P)=(\eta_{\rm mem}\eta_{d2})^{2(n_P-1)},\quad D_P^{(\pi)}=\exp[-(n_Pt_0+\tau_P^{(\pi)})/T]')
    st.markdown(
        'A path with $n_P$ edges requires $n_P-1$ swaps. $P_{\\rm swap}$ is conditional '
        'on having the initial resources, and equals one for a direct path. '
        '$t_0=L n_f/c$ is the baseline storage interval per elementary pair. '
        '$\\tau_P$ sums the additional lifetimes of elementary and intermediate pairs. '
        'All storage times and $T$ are expressed in milliseconds in the calculation.'
    )
    st.markdown(
        '**Repeater-induced waiting** swaps as repeaters become available and tracks each '
        'stored segment. **Maximum waiting** stores all elementary pairs until the last '
        'required repeater is available. Each prescription independently chooses its own '
        'maximum and minimum average over the candidate orders. The same-order tables '
        'separately compare both prescriptions on one fixed order.'
    )

    st.header('Elementary-pair generation and availability')
    st.markdown(
        'Each physical edge supports $m$ independent generation modes, each producing '
        'at most one elementary entangled pair. The number of successful pairs follows '
        'a binomial distribution. The probability of meeting the required pair demand '
        'is calculated for each edge.'
    )
    st.latex(r'p_e=\eta_{d1}^2\rho^2 10^{-\alpha L_e/10},\qquad X_e\sim{\rm Binomial}(m,p_e)')
    st.latex(r'\Pr(X_e=x)=\binom mx p_e^x(1-p_e)^{m-x},\qquad B_e(K)=\sum_{x=K}^m\binom mx p_e^x(1-p_e)^{m-x}')
    st.markdown(
        '$X_e$ counts the available pairs. The binomial tail $B_e(K)$ is the probability '
        'of obtaining at least $K$ pairs, with $B_e(0)=1$ and $B_e(K)=0$ if $K>m$. '
        'The probability of obtaining at least one pair is the special case '
        '$B_e(1)=1-(1-p_e)^m$. We do not raise $B_e(1)$ to $K$ here, since the '
        '$K$ pairs come from one shared mode pool, not $K$ independent pools.'
    )
    st.markdown(
        '**Complete-workload admission:** every edge must meet its demand $K_e$. '
        'If any edge falls short, the entire workload is rejected in this generation round. '
        'All requests then receive the failure contribution. Partial service is not included.'
    )
    st.latex(r'P_{\rm link}^{(r)}=P_{\rm prov}^{(r)}=\prod_{e\in E}B_e(K_e^{(r)})')

    st.header('Models 0, 1, and 2: assumptions and resource allocation')
    st.markdown(
        'All three models share the same topology, workload, memory law, and repeater scheduling '
        'prescription. Models 1 and 2 differ in their pair-allocation demands, not in the '
        'conditional quality of successfully generated pairs.'
    )
    st.subheader('Model 0 · Ideal resource establishment')
    st.markdown(
        'All required pairs are available deterministically and all swaps succeed. Fidelity '
        'can still fall because the resources experience baseline storage and contention waiting.'
    )
    st.latex(r'P_{\rm link}^{(0)}=P_{\rm swap}^{(0)}=1,\qquad F_{P,\rm FI}^{(0,\pi)}=(1+D_P^{(\pi)})/2')
    st.subheader('Model 1 · Network-wide pair allocation')
    st.markdown(
        'Every request reserves one distinct pair on every physical edge, including edges '
        'outside its route. Each edge must therefore supply $R=\\binom N2$ pairs. '
        'For five chain or star nodes, the four demands are $(10,10,10,10)$.'
    )
    st.latex(r'K_e^{(1)}=R,\qquad P_{\rm link}^{(1)}=\prod_e B_e(R)')
    st.subheader('Model 2 · Path-based pair allocation')
    st.markdown(
        'Each request reserves pairs only along its own route. Sum those allocations over '
        'the entire workload to determine each edge’s demand. The resulting joint provisioning '
        'probability is still shared by all requests; it is not restricted to the selected path.'
    )
    st.latex(r'K_e^{(2)}=\sum_{P\in\mathcal R}\mathbf1(e\in P),\qquad P_{\rm link}^{(2)}=\prod_e B_e(K_e^{(2)})')
    st.table({
        'Model / topology': ['Model 1: chain or star', 'Model 2: chain', 'Model 2: star'],
        'Demand on each edge': ['R = N(N−1)/2', 'i(N−i) on chain edge (i, i+1)', 'N−1 on each spoke'],
        'Five-node demands': ['10, 10, 10, 10', '4, 6, 6, 4', '4, 4, 4, 4'],
        'Identical-edge P_link': ['B(10)^4', 'B(4)^2 B(6)^2', 'B(4)^4'],
    })

    st.header('From the effective path resource to the network average')
    st.latex(r'q_P^{(r)}=P_{\rm succ}^{(r)}(P)=P_{\rm link}^{(r)}P_{\rm swap}(P)')
    st.latex(r'w_{P,\rm eff}^{(r,\pi)}=P_{\rm link}^{(r)}P_{\rm swap}(P)D_P^{(\pi)},\qquad F_{P,\rm FI}^{(r,\pi)}=\frac{1+w_{P,\rm eff}^{(r,\pi)}}2')
    st.markdown(
        '$P_{\\rm link}$ describes initial provisioning; $P_{\\rm succ}$ additionally includes '
        'swapping. Once $P_{\\rm succ}$ is used, do not multiply by $P_{\\rm swap}$ a second time. '
        'The model keeps the prescribed memory exposure even on branches where swapping fails.'
    )
    st.latex(r'F_{\rm avg,FI}^{(r,\pi)}=\frac{1}{R}\sum_{P\in\mathcal R}F_{P,\rm FI}^{(r,\pi)},\qquad F_{\max}^{\rm avg,(r)}=\max_{\pi}F_{\rm avg,FI}^{(r,\pi)},\quad F_{\min}^{\rm avg,(r)}=\min_{\pi}F_{\rm avg,FI}^{(r,\pi)}')
    st.markdown(
        'Maximize or minimize the **average over complete execution orders**, rather than '
        'optimizing each path separately. Sampled searches report best and worst observed '
        'values. In a uniform star, changing the leaf-pair order only redistributes the '
        'same hub waiting counts, so the network averages coincide. The path tables below '
        'show how an order leads to waiting, exposure, success factors, and the final average.'
    )
    st.caption(
        'The F = 2/3 line is a reference threshold for the teleportation score. '
        'The separate critical-distance plots show a 99% single-request generation criterion, '
        'not the full-workload FI-fidelity threshold.'
    )
    with st.expander('Physical parameter guide'):
        st.table({
            'Parameter': ['L', 'm', 'α', 'ρ', 'ηd1', 'ηd2', 'ηmem', 'nf', 'T', 'δt'],
            'Meaning': [
                'Length of one chain edge or star spoke, in km',
                'Independent generation modes on each physical edge in one round',
                'Fiber attenuation, in dB/km',
                'Per-source generation efficiency (not a density matrix)',
                'Detector efficiency for elementary pair generation',
                'Detector efficiency for swapping',
                'Memory retrieval efficiency used in swapping',
                'Fiber refractive index; sets baseline propagation/storage time',
                'Effective pair-memory coherence time, in ms',
                'Swapping interval that converts contention counts into storage time, in ms',
            ],
        })


@st.cache_data(show_spinner=False, max_entries=16)
def waiting_curves(settings, m, dt):
    out = []
    for model in (0, 1, 2):
        r = fidelities(model=model, m=m, deltas=dt, **settings)
        out.append((r['best'], r['worst'], r['exact']))
    return out


@st.cache_data(show_spinner=False, max_entries=12)
def contour_data(settings, modes, distances, deltas):
    """Optimize each grid point under the selected waiting prescription."""
    n, topology = settings['n'], settings['topology']
    _, exposure, exact = schedules(n, samples=settings['samples'],
                                    seed=settings['seed'], topology=topology)
    lengths = np.array([len(p) - 1 for p in paths(n, topology)])
    edge_p = [single_probability(d, settings['alpha'], settings['d1'], settings['rho'])
              for d in distances]
    unique = np.unique(lengths)
    t0 = np.asarray(distances) * settings['fiber'] / 300.
    weights = np.exp(-unique[:, None] * t0 / settings['T'])
    weights *= (settings['mem'] * settings['d2']) ** (2 * (unique[:, None] - 1))
    hi, lo = [], []
    for dt in deltas:
        decay = np.exp(-exposure[:, :, settings['policy']] * dt / settings['T'])
        grouped = np.stack([decay[:, lengths == k].sum(axis=1) for k in unique], axis=1)
        quality = grouped @ weights / len(lengths)
        hi.append(quality.max(axis=0))
        lo.append(quality.min(axis=0))
    grids = {}
    for model in (1, 2):
        for m in modes:
            q = [provisioning(n, model, m, p, topology)[0] for p in edge_p]
            grids[model, m] = (.5 + .5 * np.asarray(hi) * q, .5 + .5 * np.asarray(lo) * q)
    return grids, exact


@st.cache_data(show_spinner=False, max_entries=16)
def distance_data(n, topology, modes, alpha, d1, rho, target=.99):
    """Solve for the minimum single-mode probability and convert it to distance."""
    rows = []
    p0 = single_probability(0., alpha, d1, rho)
    for model in (1, 2):
        for m in modes:
            if provisioning(n, model, m, p0, topology)[0] < target:
                length, status = np.nan, 'unattainable at zero distance'
            elif alpha == 0:
                length, status = np.inf, 'unbounded at zero attenuation'
            else:
                lo, hi = 0., p0
                for _ in range(38):
                    mid = (lo + hi) / 2
                    if provisioning(n, model, m, mid, topology)[0] >= target:
                        hi = mid
                    else:
                        lo = mid
                length, status = 10 * math.log10(p0 / hi) / alpha, 'finite'
            rows.append(dict(topology=topology, model=model, m=m, target=target,
                             max_edge_km=length, max_total_fiber_km=(n-1)*length,
                             status=status))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=16)
def request_distance_data(n, topology, modes, alpha, d1, rho, target=.99):
    """Single-request limits, on the same total-network-fiber axis as the workload."""
    p0 = single_probability(0., alpha, d1, rho)
    rows = []
    longest = n-1 if topology == 'chain' else min(n-1, 2)
    for model, edge_counts in ((1, (n-1,)), (2, range(1, longest+1))):
        for count in edge_counts:
            label = 'One request · all edges' if model == 1 else f'One request · {count} edge' + ('s' if count != 1 else '')
            failure = -math.expm1(math.log(target)/count)
            for m in modes:
                p_min = -math.expm1(math.log(failure)/m)
                if p_min > p0:
                    length, status = np.nan, 'unattainable at zero distance'
                elif alpha == 0:
                    length, status = np.inf, 'unbounded at zero attenuation'
                else:
                    length, status = 10*math.log10(p0/p_min)/alpha, 'finite'
                rows.append(dict(topology=topology, model=model, m=m, target=target,
                                 max_edge_km=length, max_total_fiber_km=(n-1)*length,
                                 status=status, series=label, required_edges=count))
    return pd.DataFrame(rows)



def output(fig, data, name):
    st.pyplot(fig)
    png = io.BytesIO()
    fig.savefig(png, format='png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    frame = pd.DataFrame(data)
    with st.expander('View plotted values'):
        st.dataframe(frame, hide_index=True, width='stretch', key=name+'_values')
    left, right = st.columns(2)
    left.download_button('Download figure (PNG)', png.getvalue(), name+'.png', key=name+'_png')
    right.download_button('Download data (CSV)', frame.to_csv(index=False),
                          name+'.csv', key=name+'_csv')


def render(settings_by_topology, modes_by_topology, selected_modes, delta_max, selected_model=None, selected_delta=None):
    chosen = tuple(settings_by_topology)
    st.header('Paper figures · ' + ' and '.join(t.title() for t in chosen))
    st.caption('Waiting curves compare all three models. The contour and critical-distance figures '
               'compare Models 1 and 2; Model 0 assumes deterministic provisioning.')
    dt = tuple(sorted(set(np.linspace(0, delta_max, 81)) |
                      ({selected_delta} if selected_delta is not None and 0 <= selected_delta <= delta_max else set())))
    for topology, number in [('chain', 5), ('star', 8)]:
        if topology not in chosen:
            continue
        settings = settings_by_topology[topology]
        star = topology == 'star'
        m = selected_modes[topology]
        st.subheader(f'Figure {number} · {topology.title()} fidelity versus waiting')
        curves = waiting_curves(settings, m, dt)
        if star:
            figure_m = st.sidebar.number_input('Star model-comparison modes', min_value=1,
                                               max_value=100000, value=50, step=1,
                                               key='star_single_comparison_modes')
            fig_curves = waiting_curves(settings, figure_m, dt)
            st.caption(f'{settings["n"]} nodes · {settings["distance"]:g} km per spoke · m={figure_m}. '
                       'All three models share one panel. Maximum and minimum workload averages coincide.')
            with plt.rc_context({'font.family':'serif','font.serif':['STIXGeneral','DejaVu Serif'],
                                 'mathtext.fontset':'stix','xtick.direction':'in','ytick.direction':'in',
                                 'xtick.top':True,'ytick.right':True}):
                fig, ax = plt.subplots(figsize=(9,5.6))
                fig.subplots_adjust(left=.115,right=.955,bottom=.145,top=.93)
                rows = []
                for model,(hi,lo,exact) in enumerate(fig_curves):
                    ax.plot(dt,hi,color=['#8172b2','#e69f00','#e76f51'][model],
                            linewidth=1.65,label=f'Model {model}')
                    rows.extend(dict(model=model,m=figure_m,delta_ms=d,best=h,worst=l,
                                     max_minus_min=h-l,exact=exact) for d,h,l in zip(dt,hi,lo))
                ax.set(xlim=(0,delta_max),ylim=(.5,1.01),
                       xlabel=r'Waiting interval, $\delta_t$ (ms)',
                       ylabel='Failure-inclusive average fidelity')
                ax.minorticks_on()
                ax.grid(alpha=.25,linewidth=.45)
                ax.legend(loc='upper right',frameon=True,fancybox=False,fontsize=10)
                fig.add_artist(Rectangle((.025,.035),.965,.94,transform=fig.transFigure,
                                         fill=False,edgecolor='black',linewidth=.9))
                output(fig,rows,f'figure_{number}_{topology}_waiting')
        else:
            sweep = tuple(modes_by_topology[topology])
            selection_key = f'{topology}_waiting_modes'
            if topology == 'chain' and st.session_state.get(selection_key) == [1000, 3000, 5000]:
                st.session_state[selection_key] = list(sweep)
    
            if topology == 'chain' and any(v not in sweep for v in st.session_state.get(selection_key, [])):
                st.session_state[selection_key] = list(sweep)
            shown_modes = st.multiselect('Modes to compare in the waiting figure', sweep,
                                         default=sweep[:6], key=f'{topology}_waiting_modes')
            if not shown_modes:
                shown_modes = list(sweep[:6])
                st.caption('Showing the first six configured mode counts.')
            st.caption(f'{settings["n"]} nodes · {settings["distance"]:g} km per edge. '
                       'Models share the same axes. Marker shapes distinguish mode counts when several are shown. '
                       + ('For equal spokes, all orders give the same workload average.' if star
                          else ('Orange solid: best order. Blue dashed: worst order.' if curves[0][2]
                                else 'Orange solid / blue dashed: best / worst among sampled orders.')))
            fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True, sharex=True)
            rows = []
            mode_curves = {mv: waiting_curves(settings, mv, dt) for mv in shown_modes}
            markers = ('o','s','^','D','v','P','X','<','>','h')
            for model, ax in enumerate(axes):
                ax.set_title(f'Model {model}', fontweight='normal')
                traces = [(None, curves[0])] if model == 0 else [
                    (mv, mode_curves[mv][model]) for mv in shown_modes]
                for i, (mv, (hi, lo, exact)) in enumerate(traces):
                    marker = markers[i % len(markers)] if mv is not None and len(shown_modes) > 1 else None
                    best_color = ['#8172b2','#e69f00','#e76f51'][model] if star else '#e69f00'
                    ax.plot(dt, hi, color=best_color, label='Mode-independent' if mv is None else f'm={mv}',
                            marker=marker, markevery=(i % 8, 12), markersize=3.6,
                            linewidth=2 if model == selected_model else 1.5)
                    if not star:
                        ax.plot(dt, lo, '--', color='#4c78a8', linewidth=1.15,
                                marker=marker, markevery=(i % 8, 12), markersize=3.6)
                    rows.extend(dict(model=model, m=mv, delta_ms=d, best=h, worst=l,
                                     max_minus_min=h-l, exact=exact) for d,h,l in zip(dt,hi,lo))
            for ax in axes:
                ax.axhline(2/3, ls=':', lw=.8, color='gray')
                if selected_delta is not None and 0 <= selected_delta <= delta_max:
                    ax.axvline(selected_delta, color='#455c55', ls=':', lw=.8)
                ax.set(xlabel='Waiting interval δt (ms)', ylim=(.49,1.01))
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
            axes[0].set_ylabel('Average failure-inclusive fidelity')
            fig.tight_layout()
            output(fig, rows, f'figure_{number}_{topology}_waiting')
    

        st.subheader('Difference between maximum and minimum average fidelity')
        st.caption(f'The gap uses the single sidebar setting m={m} at each waiting interval. '
                   + ('All three curves are zero for a uniform star.' if star else
                      'The gap uses exact extrema when available, or the best and worst sampled orders.'))
        gap_fig, gap_ax = plt.subplots(figsize=(8, 3.4), layout='constrained')
        for model, (hi, lo, _) in enumerate(curves):
            gap_ax.plot(dt, hi-lo, color=['#694498','#aa5512','#17698b'][model],
                        linestyle=['-','--','-.'][model], label=f'Model {model}',
                        linewidth=2.4 if model == selected_model else 1.5)
        gap_ax.set(xlabel='Waiting interval δt (ms)', ylabel='Maximum − minimum average fidelity',
                   xlim=(0,delta_max), ylim=(0, max(1e-4, 1.1*max(float((h-l).max()) for h,l,_ in curves))))
        if selected_delta is not None and 0 <= selected_delta <= delta_max:
            gap_ax.axvline(selected_delta, color='#455c55', ls=':', lw=.8)
        gap_ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
        gap_ax.grid(alpha=.2)
        gap_ax.legend(fontsize=8)
        gap_rows = [dict(model=model, m=m, delta_ms=d, best=h, worst=l,
                         max_minus_min=h-l, exact=exact)
                    for model,(hi,lo,exact) in enumerate(curves) for d,h,l in zip(dt,hi,lo)]
        output(gap_fig, gap_rows, f'{topology}_schedule_gap')


    st.header('Quantum-advantage contours')
    st.markdown('Each panel sweeps elementary-edge distance and swapping interval at a fixed '
                'mode count. The white dashed line marks average fidelity $F=2/3$ where '
                'the grid crosses that threshold. Thin solid lines mark other fidelity levels. Both models use the original cividis palette on the same fidelity scale.')
    if st.checkbox('Show contour figures', value=True):
        for topology, number in [('chain', 6), ('star', 9)]:
            if topology not in chosen:
                continue
            settings = settings_by_topology[topology]
            star = topology == 'star'
            modes = (1000, 3000, 5000) if topology == 'chain' else (35, 50, 65, 80, 95, 110)
            st.subheader(f'Figure {number} · {topology.title()} distance–waiting contours')
            if topology == 'chain':
                st.caption('Contour modes: 1000, 3000, and 5000. Top row: maximum average fidelity; bottom row: minimum average fidelity.')
            else:
                st.caption('Top row: m=35, 50, 65. Bottom row: m=80, 95, 110. Each panel shows the common workload-average fidelity.')
            ds = tuple(np.linspace(10,40,31) if star else np.linspace(1,200,31))
            dts = tuple(np.linspace(0,delta_max,31))
            contour_settings = {k:v for k,v in settings.items() if k != 'distance'}
            with st.spinner(f'Calculating {topology} contours…'):
                grids, exact = contour_data(contour_settings, modes, ds, dts)
            st.caption('Model 1 is on the left and Model 2 is on the right. '
                       + ('Exact schedule extrema.' if exact else 'Schedule extrema are sampled.'))
            cols = 3 if star else len(modes)
            if star:
                st.caption('For identical star spokes the maximum and minimum averages coincide; each mode is therefore shown only once.')
            fig = plt.figure(figsize=(max(13, 4.1*cols), 4.1))
            widths = [1]*(2*cols) + [.09]
            grid = fig.add_gridspec(2, len(widths), width_ratios=widths,
                                   left=.065, right=.93, bottom=.14, top=.86,
                                   wspace=.42, hspace=.25)
            rows = []
            levels = np.linspace(.5, 1, 21)
            model_axes = {}
            for model, offset in ((1,0), (2,cols)):
                axes = np.array([[fig.add_subplot(grid[r,offset+c])
                                  for c in range(cols)] for r in range(2)])
                model_axes[model] = list(axes.flat)
                center = (axes[0,0].get_position().x0 + axes[0,-1].get_position().x1)/2
                fig.text(center,.95,f'Inter City Model {model}' if topology == 'chain' else f'Model {model}',ha='center',fontsize=13,
                         fontweight='normal',color='black')
                fig.text(center,.035,'Edge length (km)',ha='center',fontsize=10)
                for j,m in enumerate(modes):
                    panels = [(j//cols, j%cols, 0, 'average')] if star else [
                        (index, j, index, suffix) for index,suffix in enumerate(('best','worst'))]
                    for row, col, index, suffix in panels:
                        z = grids[model,m][index]
                        ax = axes[row,col]
                        mesh = ax.contourf(ds,dts,z,levels=levels,cmap='cividis')
                        guides = [v for v in (.505,.525,.55,.60,.75,.85,.95)
                                  if z.min()<v<z.max()]
                        if guides:
                            ax.contour(ds,dts,z,levels=guides,colors='white',
                                       linewidths=.45,alpha=.65)
                        if z.min()<2/3<z.max():
                            threshold = ax.contour(ds,dts,z,levels=[2/3],colors='white',
                                                   linestyles='--',linewidths=1.2)
                            threshold.set_path_effects([patheffects.withStroke(
                                linewidth=2.1,foreground='#17231f')])
                        ax.set_title(f'm={m}',fontsize=9)
                        ax.set_box_aspect(.76)
                        ax.tick_params(labelsize=8,labelbottom=row==1,labelleft=col==0 and model==1)
                        if col==0 and model==1:
                            if star:
                                ax.set_ylabel('δt (ms)',fontsize=9)
                            else:
                                label = r'$F_{\max}^{\mathrm{avg}}$' if row==0 else r'$F_{\min}^{\mathrm{avg}}$'
                                ax.set_ylabel(label+'\nδt (ms)',fontsize=9)
                        rows.extend(dict(model=model,m=m,order=suffix,distance_km=d,
                                         delta_ms=t,fidelity=z[y,x],exact=exact)
                                    for y,t in enumerate(dts) for x,d in enumerate(ds))
            cax=fig.add_subplot(grid[:,-1])
            fig.colorbar(mesh,cax=cax,label='Average FI fidelity')
            # Include tick labels and axis labels, not just plotting rectangles.
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bounds = {
                model: Bbox.union([ax.get_tightbbox(renderer) for ax in axes])
                           .transformed(fig.transFigure.inverted())
                for model, axes in model_axes.items()
            }
            bottom = min(.09, min(box.y0 for box in bounds.values())-.012)
            top = max(.915, max(box.y1 for box in bounds.values())+.012)
            for model, box in bounds.items():
                left, right = box.x0-.008, box.x1+.008
                fig.add_artist(Rectangle((left,bottom),right-left,top-bottom,
                                         transform=fig.transFigure,fill=False,
                                         edgecolor='black',linestyle='-',linewidth=.8,
                                         gid=f'model-{model}-border'))
            output(fig,rows,f'figure_{number}_{topology}_contours')

    st.header('Critical distance versus modes')
    st.markdown('These curves show the largest total network fiber length at which one request '
                'can be provisioned with at least 99% probability in one generation round. '
                'Total fiber length is (N−1) times the uniform edge length. Waiting and swapping '
                'are excluded. Missing points indicate an unattainable target or an unbounded '
                'distance at zero attenuation; the table gives the status.')
    for topology,number in [('chain',10),('star',11)]:
        if topology not in chosen:
            continue
        settings = settings_by_topology[topology]
        mvals = tuple(sorted(set(np.geomspace(1 if topology=='star' else 10,
                                              max(modes_by_topology[topology]),60).astype(int))
                             | set(modes_by_topology[topology])))
        mvals = tuple(sorted(set(mvals) | {selected_modes[topology]}))
        st.subheader(f'Figure {number} · {topology.title()} distance versus modes')
        df = request_distance_data(settings['n'],topology,mvals,settings['alpha'],
                                   settings['d1'],settings['rho']).copy()
        selected_m = selected_modes[topology]
        marked = set(modes_by_topology[topology]) | {selected_m}
        df['marked'] = df.m.isin(marked)
        hops = sorted(df.loc[df.model == 2, 'required_edges'].unique())
        if len(hops) > 6:
            visible_hops = st.multiselect('Model 2 request lengths to show', hops,
                                          default=hops[:5]+hops[-1:], key=f'{topology}_distance_hops')
            df = df[(df.model == 1) | df.required_edges.isin(visible_hops)].copy()
        st.caption('Markers show the configured mode counts and the current sidebar count. '
                   'The table below gives the exact distance for every marked point.')
        fig, axes = plt.subplots(1,2,figsize=(12,4.5),sharey=True,layout='constrained')
        palette = plt.get_cmap('tab10')
        for model,ax in zip((1,2),axes):
            part = df[df.model == model]
            for j,(label, trace) in enumerate(part.groupby('series',sort=False)):
                color = palette(j%10)
                finite = np.isfinite(trace.max_total_fiber_km)
                ax.plot(trace.m,trace.max_total_fiber_km.where(finite), color=color,
                        linestyle='--', linewidth=1.4, label=label)
                points = trace[trace.marked & finite]
                ax.scatter(points.m,points.max_total_fiber_km,color=color,
                           marker='s', s=18, zorder=4)
            ax.axvline(selected_m, color='#455c55', ls=':', lw=.8)
            ax.set(title=f'Model {model}',xscale='log',xlabel='Modes per edge m')
            # Keep logarithmic spacing, but label the actual mode counts.
            lower, upper = min(mvals), max(mvals)
            decades = {10**power for power in range(6) if lower <= 10**power <= upper}
            tick_values = sorted(marked | decades)
            ax.set_xlim(lower, upper if upper > lower else lower*1.1)
            ax.xaxis.set_major_locator(FixedLocator(tick_values))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis='x', labelsize=8, labelrotation=45)

            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        axes[0].set_ylabel('Maximum total network fiber length (km)')
        output(fig,df,f'figure_{number}_{topology}_distance')
        with st.expander('Critical distances at marked mode counts', expanded=True):
            points = df[df.marked][['model','series','m','max_edge_km','max_total_fiber_km','status']]
            if points.empty:
                st.info('No marked mode counts are available for the current settings.')
            else:
                st.table(points.sort_values(['model','m','series']).rename(columns={
                    'model': 'Model', 'series': 'Request allocation', 'm': 'Modes per edge',
                    'max_edge_km': 'Maximum edge length (km)',
                    'max_total_fiber_km': 'Maximum total fiber length (km)',
                    'status': 'Status',
                }).reset_index(drop=True))
