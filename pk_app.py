import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp, trapezoid
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

def auto_select_lambda_z(time, conc, tmax):
    """WinNonlin-style: 터미널 단계를 자동으로 탐지하여 최적의 Lambda_z 선택"""
    t_post_max = time[time >= tmax]
    c_post_max = conc[time >= tmax]
    if len(t_post_max) < 3: return np.nan, np.nan, 0
    
    best_r2 = -1
    best_ke = np.nan
    best_points = 0
    
    # 마지막 n개 포인트를 사용하여 선형 회귀 (최소 3개 ~ 전체)
    for n in range(3, len(t_post_max) + 1):
        t_sel = t_post_max[-n:]
        c_sel = c_post_max[-n:]
        if (c_sel <= 0).any(): continue
        
        log_c = np.log(c_sel)
        slope, intercept = np.polyfit(t_sel, log_c, 1)
        ke = -slope
        if ke <= 0: continue
        
        # R-squared 계산
        preds = intercept + slope * t_sel
        ss_res = np.sum((log_c - preds)**2)
        ss_tot = np.sum((log_c - np.mean(log_c))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # 조정된 R2 (Adjusted R-squared) 준거로 최적 포인트 선택
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 2)
        if adj_r2 > best_r2:
            best_r2 = adj_r2
            best_ke = ke
            best_points = n
            
    return best_ke, best_r2, best_points

def calculate_single_nca(time, concentration, dose=1, route='Oral', method='Linear-up Log-down'):
    df = pd.DataFrame({'Time': time, 'Concentration': concentration}).sort_values('Time').reset_index(drop=True)
    cmax = df['Concentration'].max()
    tmax = df.loc[df['Concentration'].idxmax(), 'Time']
    auc_last = aumc_last = 0
    for i in range(len(df) - 1):
        auc, aumc = calculate_auc_aumc_interval(df['Time'][i], df['Time'][i+1], df['Concentration'][i], df['Concentration'][i+1], method=method, tmax=tmax)
        auc_last += auc; aumc_last += aumc
    
    # Automated terminal phase selection
    ke, r2, n_points = auto_select_lambda_z(df['Time'].values, df['Concentration'].values, tmax)
    
    if not np.isnan(ke) and ke > 0:
        clast = df['Concentration'].iloc[-1]
        auc_inf = auc_last + (clast / ke)
        half_life = np.log(2) / ke
    else:
        auc_inf = half_life = np.nan

    if route.upper() == 'IV':
        cl = dose / auc_inf if (auc_inf and auc_inf > 0) else np.nan
        vz = cl / ke if (cl and ke) else np.nan
        params = {'Cmax': cmax, 'Tmax': tmax, 'AUC_last': auc_last, 'AUC_inf': auc_inf, 'Half_life': half_life, 'Cl': cl, 'Vz': vz, 'R2_adj': r2}
    else:
        cl_f = dose / auc_inf if (auc_inf and auc_inf > 0) else np.nan
        vz_f = cl_f / ke if (cl_f and ke) else np.nan
        params = {'Cmax': cmax, 'Tmax': tmax, 'AUC_last': auc_last, 'AUC_inf': auc_inf, 'Half_life': half_life, 'Cl_F': cl_f, 'Vz_F': vz_f, 'R2_adj': r2}
    return params

# --- ODE Models ---
def tmdd_model_ode(t, y, params):
    """
    Standard TMDD Model ODEs:
    y = [Free_Drug (L), Free_Target (R), Complex (LR)]
    """
    L, R, LR = y
    kel = params.get('kel', 0.01)
    kon = params.get('kon', 0.1)
    koff = params.get('koff', 0.01)
    kint = params.get('kint', 0.05)
    ksyn = params.get('ksyn', 1.0)
    kdeg = params.get('kdeg', 0.1)
    
    dLdt = -kel * L - kon * L * R + koff * LR
    dRdt = ksyn - kdeg * R - kon * L * R + koff * LR
    dLRdt = kon * L * R - koff * LR - kint * LR
    
    return [dLdt, dRdt, dLRdt]

def pk_pd_link_model_ode(t, y, params):
    """
    1-Compartment PK + Effect Compartment (Link Model)
    y = [Central_Amt, Effect_Conc]
    """
    Ac, Ce = y
    vd = params.get('Vd', 10.0)
    cl = params.get('Cl', 1.0)
    keo = params.get('keo', 0.1)
    
    cp = Ac / vd
    
    # PK: Central Compartment
    ke = cl / vd
    dAc_dt = -ke * Ac
    
    # Link: Effect Compartment
    dCe_dt = keo * (cp - Ce)
    
    return [dAc_dt, dCe_dt]

def pk_mm_iv_ode(t, y, vmax, km, vd):
    A = y[0]
    C = A / vd
    dAdt = -(vmax * C) / (km + C)
    return [dAdt]

def pk_mm_oral_ode(t, y, ka, vmax, km, vd):
    Agut, Acentral = y
    c_central = Acentral / vd
    dAgut = -ka * Agut
    dAcentral = ka * Agut - (vmax * c_central) / (km + c_central)
    return [dAgut, dAcentral]

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Analysis Settings")
mode = st.sidebar.selectbox("Analysis Mode", ["NCA & Fitting", "TMDD Simulation", "PK/PD Correlation", "Population Analysis"])
route = st.sidebar.radio("Route", ["IV", "Oral"])

if mode == "NCA & Fitting":
    st.sidebar.subheader("Data Input")
    input_method = st.sidebar.radio("Input Method", ["Manual Entry", "Upload CSV"])
    show_log = st.sidebar.checkbox("Log Scale", value=True)
    
    if input_method == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload PK Data (CSV)", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
        else:
            st.info("Using default sample data.")
            data = pd.DataFrame({
                'Group': ['G1']*10 + ['G2']*10,
                'Subject': ['S1','S2','S3','S4','S5']*4,
                'Sex': ['M','F','M','F','M']*4,
                'Dose': [100]*10 + [300]*10,
                'Time': [0, 1, 4, 8, 24]*4,
                'Concentration': [100, 60, 25, 12, 2, 95, 58, 22, 10, 1.5, 305, 180, 75, 35, 6, 290, 175, 70, 30, 5]
            })
    else:
        st.sidebar.info("WinNonlin 스타일로 여러 개체(N)의 데이터를 입력하세요.")
        if 'manual_data' not in st.session_state:
            st.session_state['manual_data'] = pd.DataFrame({
                'Group': ['Test Group']*6,
                'Subject': ['S1']*6,
                'Sex': ['M']*6,
                'Dose': [100.0]*6,
                'Time': [0.0, 1.0, 2.0, 4.0, 8.0, 24.0],
                'Concentration': [100.0, 80.0, 60.0, 40.0, 20.0, 5.0]
            })
        
        st.subheader("✍️ Advanced Data Editor (WinNonlin Table Style)")
        data = st.data_editor(
            st.session_state['manual_data'],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Time": st.column_config.NumberColumn("Time (hr)", min_value=0, format="%.2f"),
                "Concentration": st.column_config.NumberColumn("Conc (ng/mL)", min_value=0, format="%.2f"),
                "Dose": st.column_config.NumberColumn("Dose (mg)", min_value=0),
                "Group": st.column_config.SelectboxColumn("Group", options=["Control", "Test Group", "Group A", "Group B"]),
                "Sex": st.column_config.SelectboxColumn("Sex", options=["M", "F", "N/A"])
            }
        )
        st.session_state['manual_data'] = data

    # Statistical Aggregation
    st.subheader("📊 PK Profile & Group Statistics")
    
    # Plotting logic with group stats
    fig, ax = plt.subplots(figsize=(10, 6))
    groups = data['Group'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    
    for g, color in zip(groups, colors):
        g_data = data[data['Group'] == g]
        # Individual lines
        for sub in g_data['Subject'].unique():
            sub_data = g_data[g_data['Subject'] == sub]
            ax.plot(sub_data['Time'], sub_data['Concentration'], '-', alpha=0.2, color=color, linewidth=1)
        
        # Group Mean & Error Bar
        stats = g_data.groupby('Time')['Concentration'].agg(['mean', 'std']).reset_index()
        ax.errorbar(stats['Time'], stats['mean'], yerr=stats['std'], fmt='o-', color=color, capsize=5, label=f"{g} (Mean ± SD)")

    if show_log: ax.set_yscale('log')
    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("Concentration (ng/mL)")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    st.pyplot(fig)

    # Expanded NCA Table
    st.subheader("📈 Professional NCA Results (Grouped)")
    all_nca = []
    for (g, sub), sub_data in data.groupby(['Group', 'Subject']):
        d_val = sub_data['Dose'].iloc[0] if 'Dose' in sub_data.columns else 100
        sex_val = sub_data['Sex'].iloc[0] if 'Sex' in sub_data.columns else 'N/A'
        res = calculate_single_nca(sub_data['Time'].values, sub_data['Concentration'].values, dose=d_val, route=route)
        res.update({'Group': g, 'Subject': sub, 'Sex': sex_val})
        all_nca.append(res)
    
    nca_df = pd.DataFrame(all_nca)
    # Reorder columns for better view
    cols = ['Group', 'Subject', 'Sex', 'Cmax', 'Tmax', 'AUC_last', 'AUC_inf', 'Half_life', 'R2_adj']
    st.dataframe(nca_df[cols].style.format(precision=4))

    # Download Results
    csv = nca_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download NCA Results (CSV)", csv, "pk_nca_results.csv", "text/csv")

    st.divider()

    # --- Section: Best Compartment Model Recommendation ---
    st.subheader("🤖 Intelligent Model Recommendation")
    
    selected_group_res = st.selectbox("Select Group for CA Recommendation", groups)
    g_avg = data[data['Group'] == selected_group_res].groupby('Time')['Concentration'].mean().reset_index()
    t_avg, c_avg = g_avg['Time'].values, g_avg['Concentration'].values
    g_dose = data[data['Group'] == selected_group_res]['Dose'].iloc[0]

    # Simplified Model Recommendation Logic (AIC based)
    def one_comp_iv(t, c0, ke): return c0 * np.exp(-ke * t)
    def two_comp_iv(t, A, alpha, B, beta): return A * np.exp(-alpha * t) + B * np.exp(-beta * t)
    
    # New Oral Models
    def one_comp_oral(t, ka, ke, v_f, dose):
        return (dose * ka / (v_f * (ka - ke))) * (np.exp(-ke * t) - np.exp(-ka * t))
    
    def one_comp_oral_fit(t, ka, ke, v_f):
        # dose is captured from outer scope
        return one_comp_oral(t, ka, ke, v_f, g_dose)

    # Nonlinear MM Fit Wrappers
    def mm_iv_fit(t_eval, vmax, km, vd):
        y0 = [g_dose]
        sol = solve_ivp(pk_mm_iv_ode, (0, max(t_eval)), y0, t_eval=t_eval, args=(vmax, km, vd))
        return sol.y[0] / vd if sol.success else np.full_like(t_eval, 1e-6)

    def mm_oral_fit(t_eval, ka, vmax, km, vd):
        y0 = [g_dose, 0]
        sol = solve_ivp(pk_mm_oral_ode, (0, max(t_eval)), y0, t_eval=t_eval, args=(ka, vmax, km, vd))
        return sol.y[1] / vd if sol.success else np.full_like(t_eval, 1e-6)
    
    st.write(f"Analyzing **{selected_group_res}** (Dose: {g_dose}) to find the best fit...")
    
    best_model = "None"
    best_aic = float('inf')
    rec_results = {}

    # 1-Comp Try
    use_wls = st.sidebar.checkbox("Use WLS (1/Y² Weighting)", value=True, help="저농도 구간 정밀도를 높이기 위해 WinNonlin 스타일 가중치 적용")
    weights = 1.0 / (c_avg**2 + 1e-6) if use_wls else None # Add small epsilon to avoid div by zero

    try:
        if route == "IV":
            popt1, pcov1 = curve_fit(one_comp_iv, t_avg, c_avg, p0=[c_avg[0], 0.1], bounds=(0, np.inf), sigma=weights)
        else:
            popt1, pcov1 = curve_fit(one_comp_oral_fit, t_avg, c_avg, p0=[1.0, 0.1, 10.0], bounds=(0, np.inf), sigma=weights)
        
        func1 = one_comp_iv if route == "IV" else one_comp_oral_fit
        rss = np.sum((c_avg - func1(t_avg, *popt1))**2)
        aic = 2*len(popt1) + len(t_avg) * np.log(rss/len(t_avg))
        
        # Calculate CV% (Standard Error / Estimate * 100)
        perr1 = np.sqrt(np.diag(pcov1))
        cv1 = (perr1 / popt1) * 100
        
        rec_results['1-Comp'] = {'aic': aic, 'popt': popt1, 'func': func1, 'cv': cv1}
    except Exception as e: 
        st.write(f"1-Comp Fit Error: {e}")

    # 2-Comp Try
    if len(t_avg) >= 5 and route == "IV": # 2-comp Oral is complex for simple recommendation, keep IV for now
        try:
            popt2, pcov2 = curve_fit(two_comp_iv, t_avg, c_avg, p0=[c_avg[0]*0.8, 1.0, c_avg[0]*0.2, 0.1], bounds=(0, np.inf), sigma=weights)
            rss = np.sum((c_avg - two_comp_iv(t_avg, *popt2))**2)
            aic = 2*len(popt2) + len(t_avg) * np.log(rss/len(t_avg))
            
            perr2 = np.sqrt(np.diag(pcov2))
            cv2 = (perr2 / popt2) * 100
            
            rec_results['2-Comp'] = {'aic': aic, 'popt': popt2, 'func': two_comp_iv, 'cv': cv2}
        except: pass

    # MM Nonlinear Try
    try:
        if route == "IV":
            # Initial guess: Vmax ~ Dose/10, Km ~ AvgConc
            popt3, pcov3 = curve_fit(mm_iv_fit, t_avg, c_avg, p0=[g_dose/2, np.mean(c_avg), 10.0], bounds=(0, [np.inf, np.inf, np.inf]), sigma=weights)
            func3 = mm_iv_fit
            p_names3 = ['Vmax', 'Km', 'Vd']
        else:
            popt3, pcov3 = curve_fit(mm_oral_fit, t_avg, c_avg, p0=[1.0, g_dose/2, np.mean(c_avg), 10.0], bounds=(0, [np.inf, np.inf, np.inf, np.inf]), sigma=weights)
            func3 = mm_oral_fit
            p_names3 = ['ka', 'Vmax', 'Km', 'Vd']
            
        rss = np.sum((c_avg - func3(t_avg, *popt3))**2)
        aic = 2*len(popt3) + len(t_avg) * np.log(rss/len(t_avg))
        perr3 = np.sqrt(np.diag(pcov3))
        cv3 = (perr3 / popt3) * 100
        rec_results['MM-Nonlinear'] = {'aic': aic, 'popt': popt3, 'func': func3, 'cv': cv3, 'pnames': p_names3}
    except Exception as e:
        # st.write(f"MM Fit Debug: {e}") # Keep quiet unless needed
        pass

    if rec_results:
        best_model = min(rec_results, key=lambda x: rec_results[x]['aic'])
        st.success(f"✅ **Best Model Recommended: {best_model}** (Based on AIC)")
        
        # Diagnostic Plots
        st.subheader("🩺 Diagnostic Plots (Selected Model)")
        col1, col2 = st.columns(2)
        
        best_info = rec_results[best_model]
        pred = best_info['func'](t_avg, *best_info['popt'])
        resid = c_avg - pred
        
        # Display Parameter Confidence
        st.write("📊 **Parameter Estimates & Reliability (WinNonlin Style)**")
        if best_model == 'MM-Nonlinear':
            param_names = best_info['pnames']
        else:
            param_names = ['C0', 'ke'] if best_model == '1-Comp' and route == 'IV' else (['ka', 'ke', 'V/F'] if best_model == '1-Comp' else ['A', 'alpha', 'B', 'beta'])
        
        perf_data = []
        for name, val, cv in zip(param_names, best_info['popt'], best_info['cv']):
            perf_data.append({'Parameter': name, 'Estimate': val, 'CV%': cv})
        st.table(pd.DataFrame(perf_data).set_index('Parameter'))

        with col1:
            fig_diag1, ax_diag1 = plt.subplots()
            ax_diag1.scatter(pred, c_avg, color='blue', alpha=0.6)
            ax_diag1.plot([0, max(c_avg)], [0, max(c_avg)], 'r--')
            ax_diag1.set_xlabel("Predicted Conc")
            ax_diag1.set_ylabel("Observed Conc")
            ax_diag1.set_title("Obs vs Pred")
            st.pyplot(fig_diag1)

        with col2:
            fig_diag2, ax_diag2 = plt.subplots()
            ax_diag2.scatter(t_avg, resid, color='red', alpha=0.6)
            ax_diag2.axhline(0, color='black', linestyle='--')
            ax_diag2.set_xlabel("Time (hr)")
            ax_diag2.set_ylabel("Residual")
            ax_diag2.set_title("Residual Plot")
            st.pyplot(fig_diag2)

        # --- Automated Clinical Interpretation ---
        st.subheader("💡 Automated Clinical Insights")
        
        # Pulling some params for interpretation
        nca_g = nca_df[nca_df['Group'] == selected_group_res]
        avg_hl = nca_g['Half_life'].mean()
        avg_r2 = nca_g['R2_adj'].mean()
        
        insight_text = ""
        if avg_r2 > 0.95: insight_text += "✅ **High reliability**: Terminal phase fitting is excellent ($R^2 > 0.95$).\n"
        else: insight_text += "⚠️ **Review needed**: Terminal phase fitting has some noise. Consider adjusting sampling points.\n"
        
        if best_model == "2-Comp":
            insight_text += "🔄 **Distribution Phase**: Significant distribution observed. Multi-compartment modeling is recommended.\n"
        
        if avg_hl > 24: insight_text += "⏳ **Long Half-life**: Drug remains in system for a prolonged period. Consider potential accumulation.\n"
        else: insight_text += "⚡ **Rapid Elimination**: Drug is cleared relatively quickly.\n"
        
        st.info(insight_text)
    else:
        st.warning("Could not converge on a compartment model for this group.")

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
    dose = st.sidebar.number_input("Dose (nM)", value=100.0)
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

elif mode == "PK/PD Correlation":
    st.sidebar.subheader("PK Parameters")
    vd = st.sidebar.number_input("Volume (Vd, L)", value=10.0)
    cl = st.sidebar.number_input("Clearance (Cl, L/hr)", value=1.0)
    dose_input = st.sidebar.text_input("Doses (separated by comma)", value="10, 50, 200")
    
    st.sidebar.subheader("PD Parameters (Sigmoid Emax)")
    emax = st.sidebar.number_input("Emax", value=100.0)
    ec50 = st.sidebar.number_input("EC50", value=20.0)
    gamma = st.sidebar.slider("Hill Coefficient (gamma)", 0.5, 5.0, 1.0)
    keo = st.sidebar.slider("Equilibrium (keo, 1/hr)", 0.01, 2.0, 0.2)
    
    t_end = st.sidebar.number_input("Simulation Time (hr)", value=48)
    dose_norm = st.sidebar.checkbox("Dose-Normalized Scale (C/Dose, E/Dose)", value=False)

    # Parse doses
    try:
        doses = [float(d.strip()) for d in dose_input.split(',')]
    except:
        st.error("Invalid dose input. Using default [10, 50, 200].")
        doses = [10, 50, 200]

    all_results = []
    
    st.subheader("📊 PK/PD Simulation Results")
    
    fig_pkpd, (ax_pk, ax_pd) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    fig_corr, (ax_prop, ax_hys) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(doses)))
    
    summary_data = []

    for dose, color in zip(doses, colors):
        params = {'Vd': vd, 'Cl': cl, 'keo': keo}
        y0 = [dose, 0] # Central Amt, Effect Conc
        t_span = (0, t_end)
        t_eval = np.linspace(0, t_end, 1000)
        
        sol = solve_ivp(pk_pd_link_model_ode, t_span, y0, t_eval=t_eval, args=(params,), method='RK45')
        
        cp = sol.y[0] / vd
        ce = sol.y[1]
        effect = (emax * (ce**gamma)) / (ec50**gamma + ce**gamma)
        
        # Scaling for dose normalization
        scale = dose if dose_norm else 1.0
        
        # Plot PK
        ax_pk.plot(sol.t, cp / scale, color=color, label=f"Dose {dose}")
        # Plot PD
        ax_pd.plot(sol.t, effect / scale, color=color, ls='--', label=f"Eff (Dose {dose})")
        
        # Hysteresis
        ax_hys.plot(cp, effect, color=color, label=f"Dose {dose}")
        
        # Summary Params
        cmax = np.max(cp)
        auc = trapezoid(cp, sol.t)
        emax_obs = np.max(effect)
        teff_max = sol.t[np.argmax(effect)]
        
        summary_data.append({
            'Dose': dose,
            'Cmax': cmax,
            'AUC': auc,
            'Peak_Effect': emax_obs,
            'T_eff_max': teff_max,
            'Cmax/Dose': cmax / dose,
            'AUC/Dose': auc / dose,
            'Effect/Dose': emax_obs / dose
        })

    # Finalize PK Plot
    ax_pk.set_ylabel("Conc (ng/mL) / Dose" if dose_norm else "Conc (ng/mL)")
    ax_pk.set_title("PK: Plasma Concentration Over Time")
    ax_pk.legend()
    ax_pk.grid(True, alpha=0.3)
    
    # Finalize PD Plot
    ax_pd.set_ylabel("Effect / Dose" if dose_norm else "Effect (Units)")
    ax_pd.set_xlabel("Time (hr)")
    ax_pd.set_title("PD: Drug Effect Over Time")
    ax_pd.legend()
    ax_pd.grid(True, alpha=0.3)
    
    st.pyplot(fig_pkpd)
    
    # Dose Proportionality Plot
    sum_df = pd.DataFrame(summary_data)
    ax_prop.plot(sum_df['Dose'], sum_df['AUC'], 'o-', label="AUC")
    ax_prop.plot(sum_df['Dose'], sum_df['Cmax'], 's-', label="Cmax")
    ax_prop.set_xlabel("Dose")
    ax_prop.set_ylabel("PK Parameter")
    ax_prop_pd = ax_prop.twinx()
    ax_prop_pd.plot(sum_df['Dose'], sum_df['Peak_Effect'], '^-', color='red', label="Peak Effect")
    ax_prop_pd.set_ylabel("Peak Effect", color='red')
    ax_prop.set_title("Dose Proportionality")
    ax_prop.legend(loc='upper left')
    ax_prop_pd.legend(loc='upper right')
    ax_prop.grid(True, alpha=0.3)
    
    # Hysteresis Plot
    ax_hys.set_xlabel("Concentration (Cp)")
    ax_hys.set_ylabel("Effect (E)")
    ax_hys.set_title("Hysteresis Loop (Cp vs E)")
    ax_hys.legend()
    ax_hys.grid(True, alpha=0.3)
    
    st.subheader("📉 Dose Response & Hysteresis")
    st.pyplot(fig_corr)
    
    # Summary Table
    st.subheader("📋 PK/PD Parameter Summary")
    st.dataframe(sum_df.style.format(precision=2), use_container_width=True)

elif mode == "Population Analysis":
    st.subheader("👨‍👩‍👧‍👦 Population PK Simulation (IIV)")
    st.info("개인 간 변동성(Inter-Individual Variability)을 고려한 집단 PK 시뮬레이션입니다 (Monte Carlo method).")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Model Parameters**")
        pop_dose = st.number_input("Dose", value=100.0)
        pop_cl = st.number_input("Population Cl (L/hr)", value=2.0)
        pop_v = st.number_input("Population V (L)", value=20.0)
    
    with col2:
        st.write("**Variability (CV%)**")
        cv_cl = st.slider("Cl Variability (CV%)", 0, 100, 30)
        cv_v = st.slider("V Variability (CV%)", 0, 100, 20)
        n_subjects = st.select_slider("Number of Subjects", options=[10, 50, 100, 500], value=100)

    t_eval = np.linspace(0, 48, 100)
    pop_results = []
    
    # Monte Carlo Logic
    for i in range(n_subjects):
        # Lognormal distribution for positive parameters
        cl_i = pop_cl * np.exp(np.random.normal(0, cv_cl/100))
        v_i = pop_v * np.exp(np.random.normal(0, cv_v/100))
        ke_i = cl_i / v_i
        cp_i = (pop_dose / v_i) * np.exp(-ke_i * t_eval)
        pop_results.append(cp_i)
    
    pop_array = np.array(pop_results)
    p5 = np.percentile(pop_array, 5, axis=0)
    p50 = np.percentile(pop_array, 50, axis=0)
    p95 = np.percentile(pop_array, 95, axis=0)
    
    fig_pop, ax_pop = plt.subplots(figsize=(10, 6))
    for i in range(min(n_subjects, 50)): # Plot first 50 spaghetti lines
        ax_pop.plot(t_eval, pop_array[i], color='gray', alpha=0.1)
    
    ax_pop.plot(t_eval, p50, 'r-', linewidth=2, label="Median (P50)")
    ax_pop.fill_between(t_eval, p5, p95, color='red', alpha=0.2, label="90% Prediction Interval (P5-P95)")
    
    ax_pop.set_xlabel("Time (hr)")
    ax_pop.set_ylabel("Concentration (ng/mL)")
    ax_pop.set_title(f"Population PK Simulation (N={n_subjects})")
    ax_pop.set_yscale('log')
    ax_pop.legend()
    ax_pop.grid(True, which='both', linestyle='--', alpha=0.5)
    st.pyplot(fig_pop)
    
    st.success(f"✅ Simulation Complete. 90% PI generated for Cl={pop_cl} L/hr (CV {cv_cl}%) and V={pop_v} L (CV {cv_v}%).")

st.divider()
st.caption("Developed by Antigravity PK Engine | Automatic Updates via GitHub")
