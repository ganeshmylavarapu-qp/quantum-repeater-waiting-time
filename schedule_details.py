"""Worked schedule tables for the workload provisioning model."""
import math

import pandas as pd
import streamlit as st

from model import binomial_tail, edges, paths


def render(calculate, params, m, delta, policy_name, model):
    topology, n = params['topology'], params['n']
    st.header('Orders behind the maximum and minimum fidelity')
    st.markdown('For each execution order, calculate every path’s fidelity and take their arithmetic mean. '
                'Choose the maximum and minimum independently for each waiting prescription. '
                'Direct paths contribute to the average without occupying a swapping repeater.')
    st.caption(f'Model {model} · {topology.title()} · {n} nodes · {params["distance"]:g} km per edge · '
               f'{m} modes per edge · δt = {delta:g} ms · T = {params["T"]:g} ms.')
    policies = ['Repeater-induced waiting', 'Maximum waiting']
    results = [calculate(model, m, (delta,), dict(params, policy=i)) for i in range(2)]
    summary, order_columns = [], {}
    for label, result in zip(policies, results):
        for title, key in [('Maximum', 'best'), ('Minimum', 'worst')]:
            order = result['orders'][result[key+'_ix'][0]]
            sequence = ['–'.join(map(str, p)) for p in order if len(p) > 2]
            summary.append({'Waiting prescription': label, 'Extremum': title,
                            'Average FI fidelity': result[key][0],
                            'Search': ('Exact by symmetry' if topology == 'star' else
                                       'Exact' if result['exact'] else 'Sampled'),
                            'Orders evaluated': len(result['orders']),
                            'Swapping order': ' → '.join(sequence)})
            order_columns[f'{label} · {title.lower()}'] = sequence
    summary_df = pd.DataFrame(summary)
    st.subheader('Average fidelities and the orders that produce them')
    st.table(summary_df.drop(columns='Swapping order').round(9))
    gaps = pd.DataFrame([
        {'Waiting prescription': label, 'Maximum average FI fidelity': r['best'][0],
         'Minimum average FI fidelity': r['worst'][0],
         'Max − min': r['best'][0] - r['worst'][0]}
        for label, r in zip(policies, results)
    ])
    st.markdown('**Schedule gap at the selected waiting interval**')
    st.table(gaps.style.format({c: '{:.9g}' for c in gaps.columns[1:]}))
    st.caption('The gap uses the selected model and the sidebar waiting interval. '
               'It is zero when the two averages coincide.')
    st.download_button('Download schedule gaps (CSV)', gaps.to_csv(index=False),
                       f'{topology}_model_{model}_schedule_gaps.csv')

    if any(order_columns.values()):
        orders = pd.DataFrame(order_columns)
        orders.index = pd.RangeIndex(1, len(orders)+1, name='Service position')
        st.table(orders)
    else:
        st.caption('This topology has no paths that require swapping.')
    st.caption('Each column is one representative order. Ties are possible; for a uniform star all orders have the same workload average. Sampled extrema are the best and worst observed.')
    st.download_button('Download schedule summary (CSV)', summary_df.to_csv(index=False),
                       f'{topology}_schedule_summary.csv')

    selected = policies.index(policy_name)
    result = results[selected]
    st.subheader(f'Worked path calculation · {policy_name}')
    st.markdown('**Request-only** probability shows what that request would need if it ran alone. '
                '**Full workload** is the actual link-generation probability used for each path below: '
                'every physical edge must meet its total demand before any request starts.')
    route_indices = {path:i for i,path in enumerate(paths(n, topology))}
    physical = edges(n, topology)
    b1 = binomial_tail(m, result['p'], 1)
    example = (tuple(range(1, n+1)) if topology == 'chain' else
               next((path for path in route_indices if len(path) == 3), physical[0]))
    example_name = '–'.join(map(str, example))
    route_edges = [tuple(sorted(e)) for e in zip(example, example[1:])]
    with st.expander(f'Worked example: P_link for path {example_name}', expanded=True):
        if model == 0:
            st.markdown('Model 0 assumes deterministic pair generation, so **P_link = 1**. '
                        'No binomial calculation is needed.')
        else:
            u, v = route_edges[0]
            edge_index = physical.index((u,v))
            k, b = result['k'][edge_index], result['b'][edge_index]
            route_names = ', '.join(f'{a}–{b}' for a,b in route_edges)
            st.markdown(f'Path **{example_name}** traverses **{route_names}**. '
                        + ('Model 1 allocates a pair on every physical edge for this request.'
                           if model == 1 else 'Model 2 allocates a pair only on those route edges.'))
            st.markdown(f'**1. One mode.** On edge **{u}–{v}**, one mode has a '
                        f'**{result["p"]:.10g}** probability of producing a usable entangled pair.')
            st.latex(r'p_e=\eta_{d1}^2\rho^2\,10^{-\alpha L_e/10}='
                     + rf'({params["d1"]:g})^2({params["rho"]:g})^2'
                     + rf'10^{{-({params["alpha"]:g})({params["distance"]:g})/10}}')
            st.markdown(f'**2. One edge.** The {m:,} modes can produce at most {m:,} pairs. '
                        f'The full workload needs **at least {k}** on edge **{u}–{v}**.')
            st.latex(rf'X_e\sim\mathrm{{Binomial}}({m},p_e)')
            if k <= m:
                st.latex(rf'B_e({k})=\Pr(X_e\ge {k})'
                         + rf'=1-\sum_{{x=0}}^{{{k-1}}}'
                         + rf'\binom{{{m}}}{{x}}p_e^x(1-p_e)^{{{m}-x}}')
                st.markdown(f'For this edge, **B({k}) = {b:.10g}**.')
            else:
                st.markdown(f'**B({k}) = 0**, because the edge cannot generate more than {m:,} pairs in one round.')
            st.markdown('**3. All edges.** Multiply the probability of meeting each edge’s '
                        'own demand. Generation is independent across physical edges.')
            st.latex(r'P_{\rm link}=\prod_{e\in E} B_e(K_e)')
            if len(physical) <= 6:
                factors = ' × '.join(f'{value:.10g}' for value in result['b'])
                st.markdown(f'**{factors} = {result["q"]:.10g}** (using unrounded values).')
            else:
                st.markdown(f'The product over all {len(physical)} edges is **{result["q"]:.10g}**.')
            st.caption(f'The path table uses the full-workload value {result["q"]:.10g} '
                       'for fidelity, since all requests must be provisioned together.')
    st.subheader(f'Same-path comparison: {example_name}')
    reference_ix = result['best_ix'][0]
    reference_order = result['orders'][reference_ix]
    example_j = route_indices[example]
    tau = float(result['exposures'][reference_ix, example_j, selected] * delta)
    baseline = (len(example)-1) * result['t0']
    D = math.exp(-(baseline+tau)/params['T'])
    both = {r: calculate(r, m, (delta,), dict(params, policy=selected)) for r in (1,2)}
    st.markdown(
        f'Both models below use **path {example_name}** and **the same execution order**: '
        'the representative maximum-average order for the selected model and waiting prescription. '
        'We hold swapping and memory exposure fixed to isolate the generation probability.'
    )
    if topology == 'chain':
        st.caption(f'This path traverses all {len(physical)} physical edges, so every edge '
                   'in the provisioning product is also an edge of this example path.')
    else:
        st.caption('In the star, a leaf-to-leaf path uses two spokes. The current complete-workload '
                   'admission rule still requires all spokes to meet their demands.')
    st.markdown('**1. Count the required pairs and evaluate each binomial probability**')
    edge_rows = []
    for j, (u,v) in enumerate(physical):
        edge_rows.append({
            'Edge':f'{u}–{v}', 'On example path':(u,v) in route_edges,
            'Single-mode p_e':result['p'], 'Modes m':m,
            'Model 1 K':both[1]['k'][j], 'Model 1 B(K)':both[1]['b'][j],
            'Model 2 K':both[2]['k'][j], 'Model 2 B(K)':both[2]['b'][j]})
    st.table(pd.DataFrame(edge_rows).set_index('Edge').style.format(
        {'Single-mode p_e':'{:.10g}','Model 1 B(K)':'{:.10g}','Model 2 B(K)':'{:.10g}'}))
    for r in (1,2):
        factors = [
            rf'B_{{{u},{v}}}({k})' for (u,v),k in zip(physical,both[r]['k'])]
        expression = r'\,'.join(factors) if len(factors) <= 8 else r'\prod_{e\in E}B_e(K_e^{('+str(r)+')})'
        st.latex(rf'P_{{\rm link}}^{{({r})}}='+expression)
        st.markdown(f'**Model {r}:** ' + ' × '.join(f'{b:.10g}' for b in both[r]['b'])
                    + f' = **{both[r]["q"]:.10g}** (calculated at full precision).')
    st.markdown('**2. Use the same swaps and storage exposure**')
    swaps = len(example)-2
    st.markdown(f'The path contains **{len(example)-1} edges** and requires **{swaps} swaps**. '
                f'Baseline exposure is **{baseline:.9g} ms**; additional exposure is '
                f'**{tau:.9g} ms** at the selected waiting interval.')
    common_swap = float(both[1]['swap'][example_j])
    st.latex(rf'P_{{\rm swap}}=(\eta_{{\rm mem}}\eta_{{d2}})^{{{2*swaps}}}'
             + rf'=({params["mem"]:g}\times {params["d2"]:g})^{{{2*swaps}}}')
    st.markdown(f'**P_swap = {common_swap:.10g}**, and **D = {D:.10g}** for both models.')
    st.markdown('**3. Calculate the failure-inclusive fidelity of this same path**')
    comparison = []
    for r in (1,2):
        q = both[r]['q']
        succ = q * common_swap
        fidelity = (1+succ*D)/2
        comparison.append({'Model':r,'Path':example_name,'P_link':q,'P_swap':common_swap,
                           'P_success':succ,'Baseline ms':baseline,'Additional exposure ms':tau,
                           'D':D,'Path FI fidelity':fidelity})
        st.markdown(f'**Model {r}: F = (1 + {q:.10g} × {common_swap:.10g} × '
                    f'{D:.10g}) / 2 = {fidelity:.10g}.**')
    compare_df = pd.DataFrame(comparison)
    st.table(compare_df.set_index('Model').style.format(
        {key:'{:.10g}' for key in compare_df.columns if key not in ('Model','Path')}))
    st.metric('Model 2 − Model 1 path fidelity',
              f'{comparison[1]["Path FI fidelity"]-comparison[0]["Path FI fidelity"]:.10g}')
    st.download_button('Download same-path model comparison (CSV)',compare_df.to_csv(index=False),
                       f'{topology}_same_path_model_comparison.csv')
    with st.expander('Execution order used in this comparison'):
        st.table(pd.DataFrame({'Service position':range(1,len(reference_order)+1),
                              'Path':['–'.join(map(str,p)) for p in reference_order]}))
    tabs = st.tabs(['Maximum average: order and calculation', 'Minimum average: order and calculation'])
    for tab, title, key in zip(tabs, ['Maximum', 'Minimum'], ['best', 'worst']):
        with tab:
            ix = result[key+'_ix'][0]
            order = result['orders'][ix]
            counts, rows = {}, []
            for rank,path in enumerate(order, 1):
                j = route_indices[path]
                route_edges = [tuple(sorted(e)) for e in zip(path,path[1:])]
                allocated = physical if model == 1 else route_edges if model == 2 else []
                allocated_names = ', '.join(f'{u}–{v}' for u,v in allocated)
                own_factors = ' × '.join(f'B({u}–{v}, 1)' for u,v in allocated) or '1'
                ri, mw = result['exposures'][ix,j] * delta
                baseline = len(route_edges) * result['t0']
                tau = (ri,mw)[selected]
                memory = math.exp(-(baseline + tau) / params['T'])
                succ = result['q'] * result['swap'][j]
                waits = []
                for repeater in path[1:-1]:
                    count = counts.get(repeater, 0)
                    waits.append(f'R{repeater}: {count} ({count*delta:g} ms)')
                    counts[repeater] = count+1
                rows.append({'Order':rank,'Path':'–'.join(map(str,path)),'Edges':len(route_edges),
                             'Request allocation':allocated_names or 'Deterministic',
                             'Request-only factors':own_factors,'B(1) per edge': b1 if model else 1.,
                             'Request-only P_link':b1**len(allocated),
                             'P_link used (full workload)':result['q'],
                             'Repeater waits':', '.join(waits) or 'No swap',
                             'Baseline exposure ms':baseline,'RIWT exposure ms':ri,
                             'Maximum-wait exposure ms':mw,'Selected exposure ms':tau,
                             'P_swap':result['swap'][j],'P_success':succ,'D':memory,
                             'F_FI':(1+succ*memory)/2,
                             'FI fidelity: repeater-induced':(1+succ*math.exp(-(baseline+ri)/params['T']))/2,
                             'FI fidelity: maximum waiting':(1+succ*math.exp(-(baseline+mw)/params['T']))/2})
            df = pd.DataFrame(rows)
            st.markdown('**1. Read the execution order and repeater waits**')
            st.table(df[['Order','Path','Edges','Repeater waits']].set_index('Order'))
            st.markdown('**2. Calculate each path’s link probability**')
            st.table(df[['Path','Request allocation','Request-only factors','B(1) per edge',
                         'Request-only P_link','P_link used (full workload)']].set_index('Path'))
            st.markdown('**3. Apply the waiting prescription, swapping probability, and memory decay**')
            st.table(df[['Path','Baseline exposure ms','RIWT exposure ms','Maximum-wait exposure ms',
                         'Selected exposure ms']].set_index('Path').round(8))
            st.dataframe(df[['Path','P_link used (full workload)','P_swap','P_success','D','F_FI']],
                         hide_index=True,width='stretch')
            row = df.iloc[-1]
            st.markdown(f'For **{row["Path"]}**, multiply link probability **{result["q"]:.9g}** '
                        f'by swapping probability **{row["P_swap"]:.9g}**. Then '
                        f'$F=(1+P_{{\\rm success}}D)/2$ gives **{row["F_FI"]:.9f}** '
                        f'with $D={row["D"]:.9g}$.')
            st.markdown('**4. Average over every request**')
            total = df.F_FI.sum()
            st.latex(r'F_{\rm avg,FI}=\frac{'+f'{total:.9f}'+r'}{'+str(len(df))+'}='+f'{total/len(df):.9f}')
            st.metric(f'{title} average FI fidelity',f'{df.F_FI.mean():.9f}')
            with st.expander('Compare both waiting prescriptions on this same order'):
                st.table(df[['Path','FI fidelity: repeater-induced','FI fidelity: maximum waiting']].set_index('Path').round(9))
            st.download_button(f'Download {title.lower()} worked calculation (CSV)',
                               df.to_csv(index=False),f'{topology}_{key}_worked_calculation.csv')
