"""Run with: python3 -m streamlit run app.py"""
import io
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from model import edges, paths, demands, provisioning, single_probability, fidelities
from paper_dashboard import explain, render
from schedule_details import render as render_schedule_details

st.set_page_config(page_title='Quantum Network Teleportation Simulator', layout='wide')
# Match the Connecting Stars light palette and page gradient.
st.markdown('<style>\n    .stApp {\n        color: #17231f;\n        background:\n            radial-gradient(circle at 18% 6%, rgba(47, 133, 90, 0.12), transparent 30%),\n            radial-gradient(circle at 92% 14%, rgba(70, 104, 180, 0.10), transparent 26%),\n            linear-gradient(180deg, #f8faf9 0%, #eef3f0 100%);\n    }\n    .stApp,\n    .stApp p,\n    .stApp li,\n    .stApp label,\n    .stApp span,\n    .stApp div,\n    .stApp h1,\n    .stApp h2,\n    .stApp h3,\n    .stApp h4 {\n        color: #17231f;\n    }\n    [data-testid="stSidebar"] {\n        background: #ffffff;\n        border-right: 1px solid #d9e2dc;\n    }\n    [data-testid="stSidebar"] * {\n        color: #17231f;\n    }\n    [data-testid="stMarkdownContainer"] {\n        color: #17231f;\n    }\n    [data-testid="stExpander"] {\n        background-color: #f3faf6 !important;\n        border: 1px solid #cfe1d6 !important;\n        color: #17231f !important;\n    }\n    [data-testid="stExpander"] details {\n        background-color: #f3faf6 !important;\n        color: #17231f !important;\n    }\n    [data-testid="stExpander"] summary {\n        background-color: #e4f2ea !important;\n        color: #17231f !important;\n        border-radius: 0 !important;\n    }\n    [data-testid="stExpander"] summary *,\n    [data-testid="stExpander"] div,\n    [data-testid="stExpander"] p,\n    [data-testid="stExpander"] li,\n    [data-testid="stExpander"] span {\n        color: #17231f !important;\n    }\n    [data-testid="stMetricLabel"],\n    [data-testid="stMetricValue"],\n    [data-testid="stMetricDelta"] {\n        color: #17231f;\n    }\n    div[data-baseweb="select"] > div,\n    div[data-baseweb="input"] > div,\n    input,\n    textarea {\n        color: #17231f !important;\n        background-color: #ffffff !important;\n        border-color: #b8c7bf !important;\n    }\n    div[data-baseweb="select"] span {\n        color: #17231f !important;\n    }\n    code,\n    pre {\n        color: #17231f !important;\n        background-color: #eef5f1 !important;\n    }\n    [data-testid="stDataFrame"],\n    [data-testid="stDataFrame"] div {\n        background-color: #ffffff !important;\n        color: #17231f !important;\n    }\n    [data-testid="stDownloadButton"] button,\n    [data-testid="stDownloadButton"] button[kind="secondary"],\n    .stDownloadButton button,\n    button[kind="secondary"],\n    button[data-testid="baseButton-secondary"],\n    .stApp button {\n        background-color: #e4f2ea !important;\n        color: #17231f !important;\n        border: 1px solid #9fc6af !important;\n        border-radius: 8px !important;\n    }\n    [data-testid="stDownloadButton"] button *,\n    .stDownloadButton button *,\n    button[kind="secondary"] *,\n    button[data-testid="baseButton-secondary"] *,\n    .stApp button * {\n        color: #17231f !important;\n    }\n    [data-testid="stDownloadButton"] button:hover,\n    .stDownloadButton button:hover,\n    button[kind="secondary"]:hover,\n    button[data-testid="baseButton-secondary"]:hover,\n    .stApp button:hover {\n        background-color: #d2eadb !important;\n        border-color: #7fb090 !important;\n        color: #17231f !important;\n    }\n    [data-testid="stTable"],\n    [data-testid="stTable"] * {\n        background-color: #ffffff !important;\n        color: #17231f !important;\n    }\n    [data-testid="stHeader"] { background: transparent; }\n</style>', unsafe_allow_html=True)
st.title('Quantum Network Teleportation Simulator')
st.caption('Repeater contention, multiplexed entanglement generation, and failure-inclusive fidelity in chain and star networks.')

with st.sidebar:
    st.header('Topology and physical parameters')
    topology = st.selectbox('Topology', ['Chain', 'Star']).lower()
    is_star = topology == 'star'
    selected_model = st.selectbox('Model for worked calculations', [0, 1, 2], index=1,
                                  format_func=lambda value: f'Model {value}')
    n = st.number_input('Number of nodes N', 2, 50, 5, 1)
    distance = st.number_input('Adjacent-edge distance (km)', 0., 1000., 20. if is_star else 100., 10., key=f'distance_{topology}')
    st.caption(f'Total fiber length: {(n-1)*distance:g} km · {n*(n-1)//2} unordered requests')
    m = st.number_input('Modes per physical edge m', 1, 100000, 10 if is_star else 4000, 1 if is_star else 100, key=f'modes_{topology}')
    alpha = st.number_input('Fiber attenuation α (dB/km)', 0., 10., .21, .01)
    d1 = st.number_input('Generation detector efficiency ηd1', 0., 1., .95, .01)
    rho = st.number_input('Pair-generation efficiency ρ', 0., 1., .9, .01)
    d2 = st.number_input('Swapping detector efficiency ηd2', 0., 1., .95, .01)
    mem = st.number_input('Memory retrieval efficiency ηmem', 0., 1., .95, .01)
    fiber = st.number_input('Fiber refractive index', 1., 3., 1.5, .1)
    T = st.number_input('Memory coherence time T (ms)', .001, 100000., 200., 10.)
    st.header('Waiting and mode sweep')
    delta = st.number_input('Waiting interval δt for path table (ms)', 0., 10000., .1, .1)
    delta_max = st.number_input('Waiting plot maximum δt (ms)', .1, 10000., 200., 10.)
    policy_name = st.selectbox('Waiting prescription', ['Repeater-induced waiting', 'Maximum waiting'])
    policy = int(policy_name == 'Maximum waiting')
    if st.session_state.get('sweep_chain') in ('1000,2000,4000,6000,8000,10000', '1000,2000,4000,8000', '1000,2000,3000,4000,5000,8000', '1000,3000,5000'):
        st.session_state['sweep_chain'] = '1000,2000,3000,4000,5000,8000'
    if st.session_state.get('sweep_star') == '35,50,65,80,95,110':
        st.session_state['sweep_star'] = '35,50,65,80'
    mode_text = st.text_input('Mode sweep (comma-separated)', '35,50,65,80' if is_star else '1000,2000,3000,4000,5000,8000', key=f'sweep_{topology}')
    samples = st.number_input('Sampled orders for larger chains', 10, 2000, 300, 10)
    seed = st.number_input('Schedule sampling seed', 0, 100000, 7, 1)

try:
    mode_values = sorted(set(int(x.strip()) for x in mode_text.split(',') if x.strip()))
    if not mode_values or any(x < 1 or x > 100000 for x in mode_values):
        raise ValueError
except ValueError:
    st.error('Enter integer mode counts between 1 and 100000, separated by commas.')
    st.stop()

R = n*(n-1)//2
p = single_probability(distance, alpha, d1, rho)
params = dict(n=n, topology=topology, distance=distance, T=T, alpha=alpha, d1=d1, d2=d2,
              rho=rho, mem=mem, fiber=fiber, policy=policy, samples=samples, seed=seed)

@st.cache_data(show_spinner=False, max_entries=32)
def calculate(model, modes, deltas, settings):
    return fidelities(model=model, m=modes, deltas=deltas, **settings)

explain()
render({topology: params}, {topology: mode_values}, {topology: m}, delta_max,
       selected_model=selected_model, selected_delta=delta)
st.header(f'Explore the {topology} workload')

with st.expander('Model definition and interpretation', expanded=True):
    st.markdown('Every unordered source–target pair makes one request. All requests form one initial workload. '
                'Each physical edge has **one pool of m independent generation modes**, each producing at most one pair. '
                'This is not m modes separately assigned to every request.')
    st.latex(r'X_e\sim\mathrm{Binomial}(m,p_e),\quad p_e=\eta_{d1}^2\rho^2 10^{-\alpha L_e/10},\quad B_e(K)=\Pr(X_e\ge K)')
    if is_star:
        st.latex(r'K_{0i}^{(1)}=R=\binom{N}{2},\qquad K_{0i}^{(2)}=N-1')
        st.markdown('Node 0 is the hub. Each spoke serves one direct request and N−2 leaf-to-leaf requests.')
    else:
        st.latex(r'K_{i,i+1}^{(1)}=R=\binom{N}{2},\qquad K_{i,i+1}^{(2)}=i(N-i)')
    st.markdown('**Model 1** reserves one pair on every network edge for every request. '
                '**Model 2** reserves a pair only on edges used by the request. '
                'Each model admits the complete workload only if every edge meets its demand; otherwise every request receives the maximally mixed fallback. '
                'This is an effective scoring convention, not a claim that physical photon loss creates a mixed two-qubit state.')
    st.latex(r'P_{\rm prov}^{(r)}=\prod_e B_e(K_e^{(r)}),\quad P_{\rm succ}^{(r)}(P)=P_{\rm prov}^{(r)}P_{\rm swap}(P)')
    st.latex(r'F_{P,\rm FI}^{(r,\pi)}=\frac{1+P_{\rm prov}^{(r)}P_{\rm swap}(P)\exp[-(n_Pt_0+\tau_P^{(\pi)})/T]}{2}')
    st.markdown('Swapping failures do not change the prescribed schedule. Generation is one round; no retries or generation-time storage penalties are added. '
                'The waiting calculation retains the original app’s per-repeater contention counters and segment lifetimes. '
                'It orders requests at shared repeaters; it does not add a global service-time delay to all requests. '
                'Model 0 remains a deterministic reference.')

st.subheader('Required entangled pairs per edge')
st.caption('Each row shows how many distinct pairs the complete set of communication requests requires on that physical edge.')
physical_edges = edges(n, topology)
ks1, ks2 = demands(n, 1, topology), demands(n, 2, topology)
edge_rows = []
for (u, v), k1, k2 in zip(physical_edges, ks1, ks2):
    edge_rows.append({'Edge': f'{u}–{v}', 'Model 1 pairs': k1, 'Model 2 pairs': k2})
st.table(pd.DataFrame(edge_rows).set_index('Edge'))
if m < max(ks1):
    st.warning('Model 1 cannot provision this workload: its demand exceeds m on every edge. Its failure-inclusive fidelity is 0.5.')
if m < max(ks2):
    st.warning('Model 2 cannot provision this workload: at least one edge demand exceeds m. Its failure-inclusive fidelity is 0.5.')
with st.expander('Request-by-edge allocation table'):
    which = st.radio('Allocation model', [1,2], horizontal=True)
    allocation = []
    for path in sorted(paths(n, topology), key=lambda x:(len(x),x)):
        row = {'Request path':'–'.join(map(str,path))}
        traversed = {tuple(sorted(edge)) for edge in zip(path, path[1:])}
        row.update({f'{u}–{v}': int(which == 1 or (u, v) in traversed) for u, v in physical_edges})
        allocation.append(row)
    st.table(pd.DataFrame(allocation).set_index('Request path'))

st.subheader('Generation probability and provisioning')
st.write(f'Single-mode probability pᵢⱼ = {p:.12g}; expected pairs per edge m pᵢⱼ = {m*p:.6g}.')
provision_rows=[]
for mv in mode_values:
    row={'Modes m':mv}
    for model in [1,2]:
        q,ks,bs,logq=provisioning(n,model,mv,p,topology)
        row[f'Model {model} provisioning']=q
        row[f'Model {model} log10 provisioning']=logq/np.log(10)
    provision_rows.append(row)
provision_df=pd.DataFrame(provision_rows)
st.table(provision_df.set_index('Modes m').style.format(precision=10))
st.download_button('Download provisioning CSV', provision_df.to_csv(index=False), f'{topology}_provisioning.csv')

st.subheader('Average failure-inclusive fidelity versus waiting')
st.caption(f'Selected mode count m={m}. The full vertical range includes generation failures near F=0.5.')
dt = tuple(np.linspace(0,delta_max,81))
results={}
with st.spinner(f'Evaluating {topology} schedules…'):
    for model in [0,1,2]:
        results[model]=calculate(model,m,dt,params)
ref=results[1]
if is_star:
    st.caption('Exact workload average by symmetry: all leaf-to-leaf requests share the same hub and have identical parameters. Changing their order only reassigns waiting times, so the best and worst average curves coincide.')
elif ref['exact']:
    st.caption(f'Exact reduced schedule enumeration: {len(ref["orders"])} orders. Direct requests have no swap-resource waiting in the inherited model.')
else:
    st.info(f'{len(ref["orders"])} sampled/candidate orders; extrema below are best/worst observed, not guaranteed global extrema. Seed={seed}.')
fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
plot_rows=[]
for model,ax in zip([1,2],axes):
    res=results[model]
    ax.plot(dt,res['best'],color='#d98b16',label='Best = worst' if is_star else ('Best' if res['exact'] else 'Best observed'))
    if not is_star:
        ax.plot(dt,res['worst'],'--',color='#356da8',label='Worst' if res['exact'] else 'Worst observed')
    ax.axhline(2/3,color='gray',lw=.8,ls=':')
    ax.set(title=f'Model {model}',xlabel='Waiting interval δt (ms)',ylim=(.49,1.01))
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    plot_rows.extend({'model':model,'m':m,'delta_ms':d,'Favg_best':h,'Favg_worst':l,'exact_orders':res['exact']} for d,h,l in zip(dt,res['best'],res['worst']))
axes[0].set_ylabel('Average failure-inclusive fidelity')
fig.tight_layout();st.pyplot(fig)
png=io.BytesIO();fig.savefig(png,format='png',dpi=220,bbox_inches='tight');plt.close(fig)
st.download_button('Download fidelity figure',png.getvalue(),f'{topology}_fidelity.png')
st.download_button('Download waiting sweep CSV',pd.DataFrame(plot_rows).to_csv(index=False),f'{topology}_waiting_sweep.csv')

st.subheader('Editable mode sweep at the selected waiting interval')
sweep_rows=[]
for mv in mode_values:
    for model in [1,2]:
        res=calculate(model,mv,(delta,),params)
        sweep_rows.append({'model':model,'m':mv,'delta_ms':delta,'provisioning':res['q'],
                           'Favg_best':res['best'][0],'Favg_worst':res['worst'][0],
                           'Max − min':res['best'][0]-res['worst'][0],'exact_orders':res['exact']})
sweep_df=pd.DataFrame(sweep_rows)
st.table(sweep_df.rename(columns={
    'model': 'Model', 'm': 'Modes per edge', 'delta_ms': 'Waiting interval (ms)',
    'provisioning': 'Provisioning probability', 'Favg_best': 'Maximum average fidelity',
    'Favg_worst': 'Minimum average fidelity', 'exact_orders': 'Exact search',
}).set_index(['Model', 'Modes per edge']).style.format({
    'Waiting interval (ms)': '{:.3g}', 'Provisioning probability': '{:.10g}',
    'Maximum average fidelity': '{:.10f}', 'Minimum average fidelity': '{:.10f}',
    'Max − min': '{:.6g}',
}))
st.download_button('Download mode sweep CSV',sweep_df.to_csv(index=False),f'{topology}_mode_sweep.csv')

st.subheader('Models 1 and 2 side by side · all selected mode values')
st.caption('Colors identify m. ' + ('Each curve is both the best and worst workload average.' if is_star else 'Solid/dashed curves show best/worst schedules.') + ' Both panels use the same axes.')
fig,axes=plt.subplots(1,2,figsize=(13,4.8),sharex=True,sharey=True)
comparison_rows=[]
colors=plt.get_cmap('viridis')(np.linspace(.08,.88,len(mode_values)))
with st.spinner('Calculating comparison curves…'):
    for model,ax in zip([1,2],axes):
        for mv,color in zip(mode_values,colors):
            r=calculate(model,mv,dt,params)
            ax.plot(dt,r['best'],color=color,lw=1.5,label=f'm={mv}')
            if not is_star:
                ax.plot(dt,r['worst'],color=color,lw=1.2,ls='--')
            comparison_rows.extend({'model':model,'m':mv,'delta_ms':d,'Favg_best':h,'Favg_worst':l,
                                    'P_prov':r['q'],'exact_orders':r['exact']} for d,h,l in zip(dt,r['best'],r['worst']))
        ax.axhline(2/3,color='gray',lw=.8,ls=':')
        ax.set(title=f'Model {model}',xlabel='Waiting interval δt (ms)',ylim=(.49,1.01),xlim=(0,delta_max))
        ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
axes[0].set_ylabel('Average failure-inclusive fidelity')
fig.tight_layout();st.pyplot(fig)
buf=io.BytesIO();fig.savefig(buf,format='png',dpi=220,bbox_inches='tight');plt.close(fig)
st.download_button('Download side-by-side comparison PNG',buf.getvalue(),f'{topology}_models_1_2_comparison.png')
st.download_button('Download side-by-side comparison CSV',pd.DataFrame(comparison_rows).to_csv(index=False),f'{topology}_models_1_2_comparison.csv')

render_schedule_details(calculate, params, m, delta, policy_name, selected_model)

archive=io.BytesIO()
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for name in ['app.py','model.py','paper_dashboard.py','schedule_details.py','requirements.txt','README.md','.streamlit/config.toml']:
        file=Path(__file__).parent / name
        if file.exists():z.write(file,name)
st.download_button('Download this separate app',archive.getvalue(),'repeater_binomial_app.zip')
