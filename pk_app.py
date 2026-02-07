import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
import io

# --- Page Config ---
st.set_page_config(page_title="Advanced PK Analysis Tool", layout="wide")
st.title("🧪 Advanced Pharmacokinetics Analysis & Simulation")

# --- Constants & Data ---
PK_UNITS = {
    'Cmax': 'ng/mL', 'C0': 'ng/mL', 'Tmax': 'hr', 'Half_life': 'hr',
    'AUC_last': 'ng·h/mL', 'AUC_inf': 'ng·h/mL', 'MRT_inf': 'hr',
    'ke': '1/hr', 'ka': '1/hr', 'Cl': 'mL/h', 'Vz': 'mL', 'Vss': 'mL',
    'Cl_F': 'mL/h', 'Vz_F': 'mL', 'Vc': 'mL', 'V2': 'mL', 'V3': 'mL',
    'k10': '1/hr', 'k12': '1/hr', 'k21': '1/hr',
    't1/2_alpha': 'hr', 't1/2_beta': 'hr'
}

# --- Core Analysis Functions (Consolidated from Prototype) ---
def calculate_auc_aumc_interval(t1, t2, c1, c2, method='Linear-up Log-down', tmax=None):
    dt = t2 - t1
    if dt <= 0: return 0, 0
    if method == 'Linear Trapezoidal':
        auc = (c1 + c2) / 2 * dt
    elif method == 'Linear-up Log-down':
        if c2 >= c1: auc = (c1 + c2) / 2 * dt
        else:
            if c1 <= 0 or c2 <= 0: auc = (c1 + c2) / 2 * dt
            else: auc = (c1 - c2) / np.log(c1 / c2) * dt
    else: auc = (c1 + c2) / 2 * dt
    aumc = (t1 * c1 + t2 * c2) / 2 * dt
    return auc, aumc

def calculate_single_nca(time, concentration, dose=1, route='Oral', method='Linear-up Log-down'):
    df = pd.DataFrame({'Time': time, 'Concentration': concentration}).sort_values('Time').reset_index(drop=True)
    cmax = df['Concentration'].max()
    tmax = df.loc[df['Concentration'].idxmax(), 'Time']
    auc_last = aumc_last = 0
    for i in range(len(df) - 1):
        auc, aumc = calculate_auc_aumc_interval(df['Time'][i], df['Time'][i+1], df['Concentration'][i], df['Concentration'][i+1], method=method, tmax=tmax)
        auc_last += auc; aumc_last += aumc
    
    # Simple elimination estimation
    last_points = df.tail(3)
    if (last_points['Concentration'] > 0).all() and len(last_points) >= 2:
        slope, _ = np.polyfit(last_points['Time'], np.log(last_points['Concentration']), 1)
        ke = -slope
        auc_inf = auc_last + (df['Concentration'].iloc[-1] / ke) if ke > 0 else np.nan
    else: ke = auc_inf = np.nan

    if route.upper() == 'IV':
        cl = dose / auc_inf if auc_inf > 0 else np.nan
        params = {'Cmax': cmax, 'Tmax': tmax, 'AUC_last': auc_last, 'AUC_inf': auc_inf, 'ke': ke, 'Cl': cl}
    else:
        cl_f = dose / auc_inf if auc_inf > 0 else np.nan
        params = {'Cmax': cmax, 'Tmax': tmax, 'AUC_last': auc_last, 'AUC_inf': auc_inf, 'ke': ke, 'Cl_F': cl_f}
    return params

# --- ODE Models ---
def tmdd_model_ode(t, y, params):
    L, R, LR = y
    kel, kon, koff, kint, ksyn, kdeg = params['kel'], params['kon'], params['koff'], params['kint'], params['ksyn'], params['kdeg']
    dLdt = -kel * L - kon * L * R + koff * LR
    dRdt = ksyn - kdeg * R - kon * L * R + koff * LR
    dLRdt = kon * L * R - koff * LR - kint * LR
    return [dLdt, dRdt, dLRdt]

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Analysis Settings")
mode = st.sidebar.selectbox("Analysis Mode", ["NCA & Fitting", "TMDD Simulation"])
route = st.sidebar.radio("Route", ["IV", "Oral"])
dose = st.sidebar.number_input("Dose (Unit)", value=100.0)

if mode == "NCA & Fitting":
    st.sidebar.subheader("Visualization")
    show_log = st.sidebar.checkbox("Log Scale", value=True)
    
    uploaded_file = st.sidebar.file_uploader("Upload PK Data (CSV)", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        st.info("Using default sample data. Upload your CSV to analyze.")
        # Default sample data
        data = pd.DataFrame({
            'Subject': ['S1']*5 + ['S2']*5,
            'Time': [0, 1, 4, 8, 24]*2,
            'Concentration': [100, 60, 25, 12, 2, 95, 58, 22, 10, 1.5]
        })

    # Execution
    st.subheader("📊 PK Profile & Results")
    fig, ax = plt.subplots(figsize=(10, 6))
    for sub in data['Subject'].unique():
        sub_data = data[data['Subject'] == sub]
        ax.plot(sub_data['Time'], sub_data['Concentration'], 'o-', alpha=0.5, label=sub)
    
    if show_log: ax.set_yscale('log')
    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("Concentration")
    ax.legend()
    st.pyplot(fig)

    # NCA Table
    st.subheader("📈 NCA Parameters")
    results = []
    for sub in data['Subject'].unique():
        sub_data = data[data['Subject'] == sub]
        res = calculate_single_nca(sub_data['Time'].values, sub_data['Concentration'].values, dose=dose, route=route)
        res['Subject'] = sub
        results.append(res)
    st.dataframe(pd.DataFrame(results))

elif mode == "TMDD Simulation":
    st.sidebar.subheader("TMDD Parameters")
    params = {
        'kel': st.sidebar.slider("Elimination (kel)", 0.001, 0.5, 0.02),
        'kon': st.sidebar.slider("On-rate (kon)", 0.01, 2.0, 0.1),
        'koff': st.sidebar.slider("Off-rate (koff)", 0.001, 0.5, 0.01),
        'kint': st.sidebar.slider("Internalization (kint)", 0.001, 0.5, 0.05),
        'ksyn': st.sidebar.slider("Target Syn (ksyn)", 0.1, 10.0, 1.0),
        'kdeg': st.sidebar.slider("Target Deg (kdeg)", 0.01, 0.5, 0.1)
    }
    t_end = st.sidebar.number_input("End Time (hr)", value=168)

    # Simulation
    R0 = params['ksyn'] / params['kdeg']
    y0 = [dose, R0, 0]
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, 500)
    sol = solve_ivp(tmdd_model_ode, t_span, y0, t_eval=t_eval, args=(params,), method='RK45')
    
    # Plotly or Matplotlib
    st.subheader("📉 TMDD & RO Projection")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    ax1.plot(sol.t, sol.y[0], label="Free Drug (L)")
    ax1.plot(sol.t, sol.y[1], label="Free Target (R)")
    ax1.plot(sol.t, sol.y[2], label="Complex (LR)")
    ax1.set_yscale('log')
    ax1.set_ylabel("Conc (nM)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ro = (sol.y[2] / (sol.y[1] + sol.y[2])) * 100
    ax2.plot(sol.t, ro, color='black', label="Receptor Occupancy")
    ax2.set_ylabel("RO (%)")
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig)

st.divider()
st.caption("Developed by Antigravity PK Engine | Automatic Updates via GitHub")
