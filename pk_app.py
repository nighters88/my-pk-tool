import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp, trapezoid
from scipy.stats import linregress
import io
import plotly.express as px
import plotly.graph_objects as go
import easyocr
import streamlit.components.v1 as components
from PIL import Image
import base64

import sqlite3
import json
from datetime import datetime

# --- Database Layer (Phase 3: Persistence & Audit Trail) ---
class PKDatabase:
    def __init__(self, db_path="pk_projects.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # Projects Table
        c.execute('''CREATE TABLE IF NOT EXISTS projects 
                     (id INTEGER PRIMARY KEY, name TEXT UNIQUE, created_at TEXT, last_modified TEXT, settings TEXT)''')
        # Data Table
        c.execute('''CREATE TABLE IF NOT EXISTS datasets 
                     (id INTEGER PRIMARY KEY, project_id INTEGER, name TEXT, content TEXT, 
                      FOREIGN KEY(project_id) REFERENCES projects(id))''')
        # Audit Trail Table
        c.execute('''CREATE TABLE IF NOT EXISTS audit_trail 
                     (id INTEGER PRIMARY KEY, project_id INTEGER, action TEXT, timestamp TEXT, details TEXT,
                      FOREIGN KEY(project_id) REFERENCES projects(id))''')
        conn.commit()
        conn.close()

    def save_project(self, name, settings, data_df, data_name="primary_data"):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        now = datetime.now().isoformat()
        settings_json = json.dumps(settings)
        
        c.execute("INSERT OR REPLACE INTO projects (name, created_at, last_modified, settings) VALUES (?, ?, ?, ?)",
                  (name, now, now, settings_json))
        project_id = c.lastrowid
        
        # Save Dataset
        data_json = data_df.to_json()
        c.execute("INSERT OR REPLACE INTO datasets (project_id, name, content) VALUES (?, ?, ?)",
                  (project_id, data_name, data_json))
        
        # Audit Trail
        c.execute("INSERT INTO audit_trail (project_id, action, timestamp, details) VALUES (?, ?, ?, ?)",
                  (project_id, "Save Project", now, f"Project '{name}' saved with {len(data_df)} datapoints."))
        
        conn.commit()
        conn.close()
        return project_id

    def load_project(self, name):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id, settings FROM projects WHERE name=?", (name,))
        res = c.fetchone()
        if res:
            project_id, settings_json = res
            settings = json.loads(settings_json)
            c.execute("SELECT content FROM datasets WHERE project_id=? ORDER BY id DESC LIMIT 1", (project_id,))
            data_res = c.fetchone()
            data_df = pd.read_json(io.StringIO(data_res[0])) if data_res else pd.DataFrame()
            conn.close()
            return settings, data_df
        conn.close()
        return None, None

    def list_projects(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT name FROM projects")
        names = [r[0] for r in c.fetchall()]
        conn.close()
        return names

    def get_audit_trail(self, project_name):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id FROM projects WHERE name=?", (project_name,))
        res = c.fetchone()
        if res:
            project_id = res[0]
            c.execute("SELECT action, timestamp, details FROM audit_trail WHERE project_id=? ORDER BY timestamp DESC", (project_id,))
            trails = c.fetchall()
            conn.close()
            return pd.DataFrame(trails, columns=['Action', 'Timestamp', 'Details'])
        conn.close()
        return pd.DataFrame()

# --- OCR Engine (Phase 4: Image Data Entry) ---
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['en'])

def run_ocr(image):
    reader = get_ocr_reader()
    results = reader.readtext(np.array(image))
    if not results: return pd.DataFrame()

    # Canonical mapping for headers
    HEADER_MAP = {
        'time': 'Time', 'hr': 'Time', 'hour': 'Time', 't': 'Time',
        'conc': 'Concentration', 'concentration': 'Concentration', 'c': 'Concentration', 'pg/ml': 'Concentration', 'ng/ml': 'Concentration',
        'group': 'Group', 'grp': 'Group', 'dose': 'Dose', 'mg/kg': 'Dose',
        'subject': 'Subject', 'id': 'Subject', 'animal': 'Subject', 'pat': 'Subject',
        'sex': 'Sex', 'gender': 'Sex', 'effect': 'Effect', 'pd': 'Effect', 'response': 'Effect'
    }

    # Common OCR character errors in numeric contexts
    OCR_CORRECTION = {'o': '0', 'O': '0', 'l': '1', 'I': '1', 'i': '1', 's': '5', 'S': '5', 'b': '6', 'g': '9', 'z': '2', 'Z': '2'}

    header_locs = {} # {ColumnName: X_Center}
    data_items = []

    # 1. Identify Headers
    for (bbox, text, prob) in results:
        clean_text_lower = text.lower().strip()
        matched_header = None
        for kw, col in HEADER_MAP.items():
            if kw == clean_text_lower or (len(kw) > 2 and kw in clean_text_lower):
                matched_header = col
                break
        
        x_center = (bbox[0][0] + bbox[2][0]) / 2
        y_center = (bbox[0][1] + bbox[2][1]) / 2

        if matched_header:
            if matched_header not in header_locs or y_center < 200: # Heuristic for top row
                header_locs[matched_header] = x_center
        else:
            data_items.append({'text': text, 'x': x_center, 'y': y_center})

    # 2. Group data into rows 
    data_items.sort(key=lambda item: (round(item['y'] / 15) * 15, item['x']))
    
    rows = []
    current_row_y = -1
    row_threshold = 20
    current_row_items = []
    
    for item in data_items:
        if current_row_y == -1 or abs(item['y'] - current_row_y) <= row_threshold:
            current_row_items.append(item)
            if current_row_y == -1: current_row_y = item['y']
        else:
            rows.append(current_row_items)
            current_row_items = [item]
            current_row_y = item['y']
    if current_row_items: rows.append(current_row_items)

    parsed_final = []
    if header_locs:
        for row in rows:
            row_dict = {}
            for item in row:
                closest_col = min(header_locs.keys(), key=lambda h: abs(header_locs[h] - item['x']))
                val_text = item['text'].replace(',', '.').strip()
                
                # Apply numeric corrections for expected number columns
                if closest_col in ['Time', 'Concentration', 'Dose', 'Effect']:
                    # Normalize characters
                    for err, corr in OCR_CORRECTION.items():
                        val_text = val_text.replace(err, corr)
                    val_text = "".join([c for c in val_text if c.isdigit() or c == '.' or c == '-'])
                    try: row_dict[closest_col] = float(val_text)
                    except: pass
                else:
                    row_dict[closest_col] = val_text
            
            if row_dict:
                for col, def_val in [('Time', 0.0), ('Concentration', 0.0), ('Group', 'Group 1'), ('Subject', 'S1'), ('Dose', 100.0), ('Sex', 'M')]:
                    if col not in row_dict: row_dict[col] = def_val
                parsed_final.append(row_dict)
    else:
        # Fallback for simple Time/Conc pair tables
        for row in rows:
            row_vals = []
            for item in row:
                txt = item['text'].replace(',', '.').strip()
                for err, corr in OCR_CORRECTION.items():
                    txt = txt.replace(err, corr)
                clean = "".join([c for c in txt if c.isdigit() or c == '.' or c == '-'])
                try: row_vals.append(float(clean))
                except: continue
            if len(row_vals) >= 2:
                parsed_final.append({'Time': row_vals[0], 'Concentration': row_vals[1], 'Group': 'Group 1', 'Subject': 'S1', 'Dose': 100.0, 'Sex': 'M'})

    if parsed_final:
        return pd.DataFrame(parsed_final)
    return pd.DataFrame()

def render_data_input_sidebar(state_key, example_func, example_args=(), mode_label="Mode"):
    """Modular data input sidebar supporting CSV, OCR, and Smart Paste."""
    st.sidebar.subheader(f"Data Input ({mode_label})")
    input_method = st.sidebar.radio("Input Method", 
                                    ["Manual Entry", "Upload CSV", "Photo/Image (OCR)", "Smart Paste (Text)"],
                                    key=f"input_method_{state_key}")
    
    if input_method == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(f"Upload {mode_label} Data (CSV)", type="csv", key=f"csv_{state_key}")
        if uploaded_file is not None:
            st.session_state[state_key] = pd.read_csv(uploaded_file)
            st.success("CSV Uploaded!")
            
    elif input_method == "Photo/Image (OCR)":
        st.sidebar.markdown("### 📸 Image Paste Hub")
        st.sidebar.text_input("hidden_clipboard_data", label_visibility="collapsed", key=f"hidden_clip_{state_key}")
        clipboard_image_listener_with_key(f"hidden_clip_{state_key}")
        
        img_src = None
        img_file = st.sidebar.file_uploader("이미지 파일 선택/드롭", type=['png', 'jpg', 'jpeg'], key=f"img_{state_key}")
        if img_file:
            img_src = Image.open(img_file)
            st.sidebar.image(img_src, caption="업로드된 이미지", use_container_width=True)
            
        clip_data = st.session_state.get(f"hidden_clip_{state_key}")
        if img_src is None and clip_data:
            try:
                header, encoded = clip_data.split(",", 1)
                img_data = base64.b64decode(encoded)
                img_src = Image.open(io.BytesIO(img_data))
                st.sidebar.image(img_src, caption="클립보드 이미지", use_container_width=True)
            except: pass

        if img_src and st.sidebar.button("🔍 AI 데이터 추출 시작 (OCR)", type="primary", use_container_width=True, key=f"ocr_btn_{state_key}"):
            with st.spinner("이미지 분석 중..."):
                ocr_df = run_ocr(img_src)
                if not ocr_df.empty:
                    st.session_state[state_key] = ocr_df
                    st.sidebar.success("데이터 추출 성공!")
                    st.rerun()
                else:
                    st.sidebar.error("데이터 인식 실패.")
                    
    elif input_method == "Smart Paste (Text)":
        st.sidebar.caption("Shift+Win+S 캡처 후 [모든 텍스트 복사] 하여 아래 붙여넣으세요.")
        paste_text = st.sidebar.text_area("텍스트 데이터 붙여넣기", height=150, placeholder="0  10.2\n1  25.4...", key=f"paste_{state_key}")
        if st.sidebar.button("⚡ 데이터로 변환하기", type="primary", use_container_width=True, key=f"paste_btn_{state_key}"):
            if paste_text:
                paste_df = parse_smart_paste(paste_text)
                if not paste_df.empty:
                    st.session_state[state_key] = paste_df
                    st.sidebar.success(f"성공: {len(paste_df)}개의 데이터!")
                    st.rerun()
    
    else: # Manual / Example
        if st.sidebar.button("🔄 Reset to Example Data", key=f"reset_{state_key}"):
            st.session_state[state_key] = example_func(*example_args)
            st.rerun()

    if state_key not in st.session_state:
        st.session_state[state_key] = example_func(*example_args)

def clipboard_image_listener_with_key(key):
    """Custom HTML component to capture clipboard images and send to a specific state key."""
    html_code = f"""
    <div id="paste-zone-{key}" style="border: 2px dashed #4A90E2; padding: 20px; border-radius: 10px; text-align: center; background-color: #f0f8ff; cursor: pointer;">
        <p style="margin: 0; color: #1e3a8a; font-weight: bold;">[클릭 후 붙여넣기]</p>
    </div>
    <script>
    const zone = document.getElementById('paste-zone-{key}');
    zone.addEventListener('paste', async (e) => {{
        const items = e.clipboardData.items;
        for (let i = 0; i < items.length; i++) {{
            if (items[i].type.indexOf('image') !== -1) {{
                const blob = items[i].getAsFile();
                const reader = new FileReader();
                reader.onload = (event) => {{
                    const base64 = event.target.result;
                    const allInputs = window.parent.document.querySelectorAll('input');
                    const target = Array.from(allInputs).find(input => 
                        input.getAttribute('aria-label') === '{key}' || 
                        input.getAttribute('placeholder') === '{key}'
                    );
                    if (target) {{
                        target.value = base64;
                        target.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        target.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    }}
                }};
                reader.readAsDataURL(blob);
            }}
        }}
    }});
    </script>
    """
    components.html(html_code, height=90)


# --- Smart Paste Parser (Phase 4.1/4.7) ---
def parse_smart_paste(text):
    import re
    text_clean = text.replace(',', '.')
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text_clean)
    nums = [float(n) for n in nums]
    rows = []
    for i in range(0, len(nums) - 1, 2):
        rows.append({'Time': nums[i], 'Concentration': nums[i+1], 'Group': 'Group 1', 'Subject': 'S1', 'Dose': 100})
    return pd.DataFrame(rows)

def clipboard_image_listener():
    # A custom HTML component to capture clipboard images without opening file dialog
    html_code = """
    <div id="paste-zone" style="border: 2px dashed #4A90E2; padding: 20px; border-radius: 10px; text-align: center; background-color: #f0f8ff; cursor: pointer;">
        <p style="margin: 0; color: #1e3a8a; font-weight: bold;">[여기를 한 번 클릭] 후 [Ctrl+V]를 누르세요</p>
        <p style="margin: 5px 0 0 0; font-size: 0.8em; color: #4A90E2;">(파일 선택 창이 뜨지 않습니다)</p>
    </div>
    <script>
    const zone = document.getElementById('paste-zone');
    zone.addEventListener('paste', async (e) => {
        const items = e.clipboardData.items;
        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                const blob = items[i].getAsFile();
                const reader = new FileReader();
                reader.onload = (event) => {
                    const base64 = event.target.result;
                    // Find the hidden input by its label text or aria-label
                    const allInputs = window.parent.document.querySelectorAll('input');
                    const target = Array.from(allInputs).find(input => 
                        input.getAttribute('aria-label') === 'hidden_clipboard_data' || 
                        input.getAttribute('placeholder') === 'hidden_clipboard_data'
                    );
                    if (target) {
                        const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
                        nativeInputValueSetter.call(target, base64);
                        target.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                };
                reader.readAsDataURL(blob);
            }
        }
    });
    </script>
    """
    components.html(html_code, height=100)

pk_db = PKDatabase()

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

def generate_3x3_example(route="IV"):
    """3 Dose levels, N=3 subjects per dose (Total 9)"""
    doses = [10, 30, 100]
    times = [0, 0.5, 1, 2, 4, 8, 12, 24] if route == "IV" else [0, 0.5, 1, 2, 4, 8, 12, 24, 48]
    data = []
    
    # Simple PK models for generating 'realistic' noise data
    def iv_1c(t, d, v, cl): return (d/v) * np.exp(-(cl/v)*t)
    def oral_1c(t, d, ka, v, cl): return (d*ka/(v*(ka-cl/v))) * (np.exp(-(cl/v)*t) - np.exp(-ka*t))
    
    for i, d in enumerate(doses):
        grp = f"Group {i+1} ({d}mg)"
        for s in range(3):
            sub = f"S{i*3 + s + 1}"
            # Add some variability
            v_i = 10 * np.exp(np.random.normal(0, 0.1))
            cl_i = 1 * np.exp(np.random.normal(0, 0.1))
            ka_i = 1.5 * np.exp(np.random.normal(0, 0.1))
            
            for t in times:
                if route == "IV":
                    conc = iv_1c(t, d, v_i, cl_i)
                else:
                    conc = oral_1c(t, d, ka_i, v_i, cl_i)
                
                # Add assay noise (5% CV)
                conc *= np.exp(np.random.normal(0, 0.05))
                if t == 0 and route == "Oral": conc = 0
                
                data.append({
                    'Group': grp, 'Subject': sub, 'Sex': 'M' if s%2==0 else 'F',
                    'Dose': d, 'Time': t, 'Concentration': round(conc, 3)
                })
    return pd.DataFrame(data)

def generate_3x3_tmdd_example():
    """TMDD specific example with 3 doses and N=3"""
    doses = [10, 50, 200]
    times = [0, 1, 4, 8, 24, 48, 72, 120, 168]
    data = []
    # Base params for generation
    params = {'kel': 0.02, 'kon': 0.1, 'koff': 0.01, 'kint': 0.05, 'ksyn': 1.0, 'kdeg': 0.1}
    R0 = params['ksyn'] / params['kdeg']
    
    for i, d in enumerate(doses):
        grp = f"Dose {d}nm"
        for s in range(3):
            sub = f"Subj_{i*3+s+1}"
            # Subject specific noise
            p_i = {k: v * np.exp(np.random.normal(0, 0.05)) for k, v in params.items()}
            y0 = [d, R0, 0]
            sol = solve_ivp(tmdd_model_ode, (0, 168), y0, t_eval=times, args=(p_i,))
            for j, t in enumerate(times):
                data.append({
                    'Group': grp, 'Subject': sub, 'Dose': d, 'Time': t,
                    'Concentration': round(sol.y[0][j] * np.exp(np.random.normal(0, 0.05)), 3),
                    'Free_Target': round(sol.y[1][j], 3),
                    'Complex': round(sol.y[2][j], 3)
                })
    return pd.DataFrame(data)

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

def preprocess_blq(df, lloq, method='M1', max_consecutive_blq=2):
    """
    LLOQ (Lower Limit of Quantitation) 처리 로직
    - M1: BLQ 값을 모두 제거
    - M2: 첫 BLQ는 0으로 처리, 이후는 제거
    - M4: BLQ를 LLOQ/2로 대체
    - 연속 BLQ 처리: max_consecutive_blq 이상 연속 시 이후 데이터 Missing/Blank 처리
    """
    if lloq <= 0: return df
    df = df.copy().sort_values('Time').reset_index(drop=True)
    conc = df['Concentration'].values
    blq_mask = conc < lloq
    
    # 연속 BLQ 감지
    consecutive_count = 0
    missing_start_idx = len(df)
    for i, is_blq in enumerate(blq_mask):
        if is_blq:
            consecutive_count += 1
        else:
            consecutive_count = 0
        
        if consecutive_count >= max_consecutive_blq:
            missing_start_idx = i # 현재 인덱스부터 missing 처리
            break
            
    # missing_start_idx 이후 데이터 제거
    df = df.iloc[:missing_start_idx].copy()
    if df.empty: return df
    
    conc = df['Concentration'].values
    blq_mask = conc < lloq
    
    if method == 'M1':
        df = df[~blq_mask]
    elif method == 'M2':
        first_blq_done = False
        keep_mask = []
        new_concs = []
        for i, val in enumerate(conc):
            if val < lloq:
                if not first_blq_done:
                    new_concs.append(0.0)
                    keep_mask.append(True)
                    first_blq_done = True
                else:
                    keep_mask.append(False)
            else:
                new_concs.append(val)
                keep_mask.append(True)
        df = df[keep_mask].copy()
        df['Concentration'] = new_concs
    elif method == 'M4':
        df.loc[blq_mask, 'Concentration'] = lloq / 2
        
    return df

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

def calculate_single_nca(time, concentration, dose=1, route='Oral', method='Linear-up Log-down', lloq=0, blq_method='M1'):
    df_raw = pd.DataFrame({'Time': time, 'Concentration': concentration})
    df = preprocess_blq(df_raw, lloq, method=blq_method)
    
    if df.empty:
        return {k: np.nan for k in ['Cmax', 'Tmax', 'AUC_last', 'AUC_inf', 't1/2', 'Cl', 'Vz', 'Vss', 'MRT_inf']}

    cmax = df['Concentration'].max()
    tmax = df.loc[df['Concentration'].idxmax(), 'Time']
    # Filter for valid terminal phase fitting (non-zero concentrations)
    df_clean = df[df['Concentration'] > 0].copy()
    
    auc_last = aumc_last = 0
    for i in range(len(df) - 1):
        auc, aumc = calculate_auc_aumc_interval(df['Time'][i], df['Time'][i+1], df['Concentration'][i], df['Concentration'][i+1], method=method, tmax=tmax)
        auc_last += auc
        aumc_last += aumc
    
    # Automated terminal phase selection (WinNonlin-style)
    if df_clean.empty:
        return {'Cmax': cmax, 'Tmax': tmax, 'AUC_last': auc_last, 'AUC_inf': auc_last, 'AUC_%extrap': 0, 't1/2': np.nan, 'Cl': np.nan, 'Vz': np.nan, 'Cl_F': np.nan, 'Vz_F': np.nan, 'Vss': np.nan, 'MRT_inf': np.nan, 'Lambda_z': np.nan, 'R2_lz': np.nan}

    t_clean = df_clean['Time'].values
    c_clean = df_clean['Concentration'].values
    
    # Filter for post-Tmax concentrations for terminal phase
    t_post_max = t_clean[t_clean >= tmax]
    c_post_max = c_clean[t_clean >= tmax]
    
    best_r2 = -1
    best_lambda_z = np.nan
    
    if len(t_post_max) >= 3:
        t_log = t_post_max
        c_log = np.log(c_post_max)
        
        # Try last 3, 4, 5 points for linear regression
        for n in range(3, min(6, len(t_log) + 1)):
            t_sel = t_log[-n:]
            c_sel = c_log[-n:]
            
            # Ensure there's variability for regression
            if len(np.unique(t_sel)) < 2 or len(np.unique(c_sel)) < 2:
                continue
            
            try:
                slope, intercept, r_value, _, _ = linregress(t_sel, c_sel)
                # Only consider positive slopes (negative ke)
                if slope < 0:
                    r2_adj = 1 - (1 - r_value**2) * (n - 1) / (n - 2) # Adjusted R2
                    if r2_adj > best_r2:
                        best_r2 = r2_adj
                        best_lambda_z = -slope
            except ValueError: # Handle cases where linregress fails (e.g., all same values)
                pass

    thalf = np.log(2) / best_lambda_z if best_lambda_z > 0 else np.nan
    
    # AUC INF calculations
    clast = df_clean['Concentration'].iloc[-1] if not df_clean.empty else 0
    tlast = df_clean['Time'].iloc[-1] if not df_clean.empty else 0

    auc_extrap = clast / best_lambda_z if best_lambda_z > 0 else 0
    auc_inf = auc_last + auc_extrap
    auc_pextrap = (auc_extrap / auc_inf * 100) if auc_inf > 0 else np.nan
    
    # Clearance and Volume
    cl = dose / auc_inf if auc_inf > 0 else np.nan
    vz = cl / best_lambda_z if best_lambda_z > 0 else np.nan
    
    # MRT and Vss
    # AUMC_extrap = C_last * T_last / Lambda_z + C_last / (Lambda_z^2)
    aumc_extrap = (clast * tlast / best_lambda_z) + (clast / (best_lambda_z**2)) if best_lambda_z > 0 else 0
    aumc_inf = aumc_last + aumc_extrap
    mrt_inf = aumc_inf / auc_inf if auc_inf > 0 else np.nan
    vss = cl * mrt_inf if not np.isnan(cl) and not np.isnan(mrt_inf) else np.nan

    # For oral, Cl and Vz are apparent (Cl/F, Vz/F)
    if route.upper() == 'ORAL':
        cl_f = cl
        vz_f = vz
        cl = np.nan # Set to nan for IV-specific parameters
        vz = np.nan
    else: # IV
        cl_f = np.nan # Set to nan for Oral-specific parameters
        vz_f = np.nan

    return {
        'Cmax': cmax, 'Tmax': tmax, 'AUC_last': auc_last, 'AUC_inf': auc_inf,
        'AUC_%extrap': auc_pextrap, 't1/2': thalf, 'Cl': cl, 'Vz': vz,
        'Cl_F': cl_f, 'Vz_F': vz_f,
        'Vss': vss, 'MRT_inf': mrt_inf, 'Lambda_z': best_lambda_z, 'R2_lz': best_r2
    }

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

def parent_metabolite_model_ode(t, y, params):
    """
    Parent (1-3 Comp) to Metabolite (1-2 Comp) Model
    y = [Ap, Ap1, Ap2, Am, Am1]
    """
    n_p = params.get('n_p', 1) # Parent compartments
    n_m = params.get('n_m', 1) # Metabolite compartments
    
    Ap = y[0]
    Am = y[n_p]
    
    # Parent rates
    kel_p = params.get('kel_p', 0.1)
    k_pm = params.get('k_pm', 0.05) # Conversion to metabolite
    
    # Metabolite rates
    kel_m = params.get('kel_m', 0.1)
    
    dAp = -(kel_p + k_pm) * Ap
    dAm = k_pm * Ap - kel_m * Am
    
    dAp1 = dAp2 = dAm1 = 0
    
    # Multi-compartment Parent
    if n_p >= 2:
        Ap1 = y[1]
        k12p, k21p = params.get('k12p', 0.05), params.get('k21p', 0.05)
        dAp -= k12p * Ap - k21p * Ap1
        dAp1 = k12p * Ap - k21p * Ap1
    if n_p >= 3:
        Ap2 = y[2]
        k13p, k31p = params.get('k13p', 0.02), params.get('k31p', 0.02)
        dAp -= k13p * Ap - k31p * Ap2
        dAp2 = k13p * Ap - k31p * Ap2
        
    # Multi-compartment Metabolite
    if n_m >= 2:
        Am1 = y[n_p + 1]
        k12m, k21m = params.get('k12m', 0.05), params.get('k21m', 0.05)
        dAm -= k12m * Am - k21m * Am1
        dAm1 = k12m * Am - k21m * Am1
        
    dydt = [dAp]
    if n_p >= 2: dydt.append(dAp1)
    if n_p == 3: dydt.append(dAp2)
    dydt.append(dAm)
    if n_m == 2: dydt.append(dAm1)
    
    return dydt

# --- PBPK Physiology Data (Simcyp/GastroPlus Standards) ---
PBPK_PHYSIOLOGY = {
    "Human": {
        "BW": 70, # kg
        "QC": 390, # L/h (Cardiac Output)
        "V": {"Liver": 1.69, "Kidney": 0.28, "Gut": 1.17, "Lung": 0.48, "Rest": 60.0}, # Volumes (L)
        "Q": {"Liver": 97.5, "Kidney": 74.1, "Gut": 42.9, "Lung": 390.0, "Rest": 175.5} # Blood flow (L/h)
    },
    "Rat": {
        "BW": 0.25,
        "QC": 5.4,
        "V": {"Liver": 0.009, "Kidney": 0.002, "Gut": 0.008, "Lung": 0.001, "Rest": 0.2},
        "Q": {"Liver": 0.9, "Kidney": 0.8, "Gut": 0.7, "Lung": 5.4, "Rest": 3.0}
    }
}

def pbpk_model_ode(t, y, params):
    """
    Standard 5-Organ PBPK Model
    y = [C_liver, C_kidney, C_gut, C_lung, C_rest, C_venous, C_arterial]
    """
    Cl, Ck, Cg, Clu, Cr, Cv, Ca = y
    
    # Physiology
    vol = params['V']
    flow = params['Q']
    QC = params['QC']
    
    # Compound specific
    Kp = params.get('Kp', {"Liver": 1, "Kidney": 1, "Gut": 1, "Lung": 1, "Rest": 1})
    CL_h = params.get('CL_liver', 0.1)
    CL_r = params.get('CL_renal', 0.05)
    
    # ODEs
    # Venous Blood
    dCv = (flow['Liver']*Cl/Kp['Liver'] + flow['Kidney']*Ck/Kp['Kidney'] + flow['Rest']*Cr/Kp['Rest'] - QC*Cv) / 0.1 # 0.1L Blood Vol
    # Lung
    dClu = (QC*Cv - QC*Clu/Kp['Lung']) / vol['Lung']
    # Arterial Blood
    dCa = (QC*Clu/Kp['Lung'] - QC*Ca) / 0.1
    # Liver (receives Hepatic Artery + Portal Vein from Gut)
    dCl = (flow['Liver']*Ca + flow['Gut']*Cg/Kp['Gut'] - (flow['Liver']+flow['Gut'])*Cl/Kp['Liver'] - CL_h*Cl/Kp['Liver']) / vol['Liver']
    # Kidney
    dCk = (flow['Kidney']*Ca - flow['Kidney']*Ck/Kp['Kidney'] - CL_r*Ck/Kp['Kidney']) / vol['Kidney']
    # Gut
    dCg = (flow['Gut']*Ca - flow['Gut']*Cg/Kp['Gut']) / vol['Gut']
    # Rest of Body
    dCr = (flow['Rest']*Ca - flow['Rest']*Cr/Kp['Rest']) / vol['Rest']
    
    return [dCl, dCk, dCg, dClu, dCr, dCv, dCa]

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

# PD Model Library (Direct models)
def pd_linear(c, slope): return slope * c
def pd_emax(c, emax, ec50): return (emax * c) / (ec50 + c)
def pd_sigmoid(c, emax, ec50, gamma): return (emax * (c**gamma)) / (ec50**gamma + c**gamma)
def pd_inhibitory(c, e0, imax, ic50): return e0 * (1 - (imax * c) / (ic50 + c))

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
st.sidebar.header("📁 Project Management")
project_names = pk_db.list_projects()
selected_project = st.sidebar.selectbox("Load Project", ["None"] + project_names)

project_name_input = st.sidebar.text_input("Project Name", value="New Project")

if st.sidebar.button("💾 Save Project"):
    # Gather current settings
    current_settings = {
        "mode": st.session_state.get('last_mode', "NCA & Fitting"),
        "eval_type": st.session_state.get('last_eval_type', "Preclinical (TK/Linearity)"),
        "route": st.session_state.get('last_route', "IV"),
        "lloq": lloq,
        "blq_method": blq_method,
        "max_blq": max_blq,
        "tau": tau
    }
    # We'll need a way to handle 'data' depending on the mode
    if 'nca_manual' in st.session_state:
        pk_db.save_project(project_name_input, current_settings, st.session_state['nca_manual'])
        st.sidebar.success(f"Project '{project_name_input}' saved!")
    else:
        st.sidebar.error("No data to save.")

if selected_project != "None":
    if st.sidebar.button("📂 Load Selected"):
        loaded_settings, loaded_data = pk_db.load_project(selected_project)
        if loaded_settings:
            st.session_state['nca_manual'] = loaded_data
            st.session_state['loaded_settings'] = loaded_settings
            st.sidebar.success(f"Project '{selected_project}' loaded!")
            st.rerun()

st.sidebar.divider()

# Helper to get index for selectbox from loaded settings
def get_index(options, key, default_val):
    if 'loaded_settings' in st.session_state:
        val = st.session_state['loaded_settings'].get(key, default_val)
        if val in options:
            return options.index(val)
    return options.index(default_val)

mode_options = ["NCA & Fitting", "TMDD Simulation", "PK/PD Correlation", "Population Analysis", "Dose-Response & PD Modeling", "Parent-Metabolite Modeling", "Full PBPK Engine"]
mode = st.sidebar.selectbox("Analysis Mode", mode_options, index=get_index(mode_options, "mode", "NCA & Fitting"))
st.session_state['last_mode'] = mode

eval_options = ["Preclinical (TK/Linearity)", "Clinical (Variability/Accumulation)"]
eval_type = st.sidebar.radio("Evaluation Context", eval_options, index=get_index(eval_options, "eval_type", "Preclinical (TK/Linearity)"))
st.session_state['last_eval_type'] = eval_type

route_options = ["IV", "Oral"]
route = st.sidebar.radio("Route", route_options, index=get_index(route_options, "route", "IV"))
st.session_state['last_route'] = route

st.sidebar.subheader("🛡️ Data Quality Controls")
lloq_init = st.session_state['loaded_settings'].get('lloq', 0.0) if 'loaded_settings' in st.session_state else 0.0
lloq = st.sidebar.number_input("LLOQ (Lower Limit of Quantitation)", value=lloq_init, min_value=0.0)

blq_options = ["M1 (Exclude)", "M2 (First BLQ 0)", "M4 (LLOQ/2)"]
blq_method = st.sidebar.selectbox("BLQ Handling Method", blq_options, index=get_index(blq_options, "blq_method", "M1 (Exclude)"))

max_blq_init = st.session_state['loaded_settings'].get('max_blq', 2) if 'loaded_settings' in st.session_state else 2
max_blq = st.sidebar.slider("Consecutive BLQ Limit", 1, 5, max_blq_init)

tau = 24.0 # Default
if eval_type == "Clinical (Variability/Accumulation)":
    tau = st.sidebar.number_input("Dosing Interval (Tau, hr)", value=24.0, help="축적성(Rac) 및 Steady-state 계산을 위한 투여 간격")

if mode == "NCA & Fitting":
    render_data_input_sidebar('nca_manual', generate_3x3_example, (route,), "NCA & Fitting")
    data = st.session_state['nca_manual']
    show_log = st.sidebar.checkbox("Log Scale", value=True)

    # --- Main Tabbed Interface (Phase 4 Mobile Optimization) ---
    tab1, tab2, tab3 = st.tabs(["📊 Data & QC", "📈 Results & Visuals", "🧬 Advanced Analysis"])

    with tab1:
        st.subheader("✍️ Data Editor & Quality Control")
        # Ensure 'data' is tied to session state for editing
        data = st.data_editor(
            st.session_state.get('nca_manual', generate_3x3_example(route)),
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
        st.session_state['nca_manual'] = data

    with tab2:
        st.subheader("📊 PK Profile & Group Statistics")

    with tab2:
        # Statistical Aggregation
        st.subheader("📊 PK Profile & Group Statistics")
        
        # Outlier Detection
        use_outlier = st.sidebar.checkbox("Auto-detect Outliers (IQR)", value=True)
        if use_outlier:
            def detect_outliers_iqr(df):
                if df.empty: return df
                # Ensure required columns exist for grouping
                required = ['Group', 'Time', 'Concentration']
                for col in required:
                    if col not in df.columns:
                        df[col] = 0 if col == 'Time' else 'Unknown/Empty'
                        if col == 'Concentration': df[col] = 0.0

                masks = []
                try:
                    for (g, t), g_t_df in df.groupby(['Group', 'Time']):
                        if len(g_t_df) >= 3:
                            q1 = g_t_df['Concentration'].quantile(0.25)
                            q3 = g_t_df['Concentration'].quantile(0.75)
                            iqr = q3 - q1
                            mask = (g_t_df['Concentration'] < (q1 - 1.5*iqr)) | (g_t_df['Concentration'] > (q3 + 1.5*iqr))
                            masks.append(mask)
                    if masks:
                        total_mask = pd.concat(masks)
                        df['Is_Outlier'] = total_mask.reindex(df.index, fill_value=False)
                    else:
                        df['Is_Outlier'] = False
                except Exception as e:
                    st.warning(f"Outlier detection failed (Data structure issue): {e}")
                    df['Is_Outlier'] = False
                return df
            data = detect_outliers_iqr(data)

        # Plotly Individual Profiles
        groups = data['Group'].unique()
        fig = px.line(data, x='Time', y='Concentration', color='Group', line_group='Subject',
                     hover_data=['Subject', 'Dose'], markers=True, 
                     title="Individual PK Profiles (Interactive)")
        if use_outlier and 'Is_Outlier' in data.columns:
            outliers = data[data['Is_Outlier']]
            fig.add_trace(go.Scatter(x=outliers['Time'], y=outliers['Concentration'], mode='markers', 
                                     name='Potential Outlier', marker=dict(color='red', size=10, symbol='x')))
        
        if show_log: fig.update_yaxes(type='log')
        st.plotly_chart(fig, use_container_width=True)

        # Expanded NCA Table
        st.subheader("📈 Professional NCA Results (Grouped)")
        all_nca = []
        for (g, sub), sub_data in data.groupby(['Group', 'Subject']):
            d_val = sub_data['Dose'].iloc[0] if 'Dose' in sub_data.columns else 100
            sex_val = sub_data['Sex'].iloc[0] if 'Sex' in sub_data.columns else 'N/A'
            res = calculate_single_nca(sub_data['Time'].values, sub_data['Concentration'].values, 
                                       dose=d_val, route=route, method='Linear-up Log-down', 
                                       lloq=lloq, blq_method=blq_method)
            res.update({'Group': g, 'Subject': sub, 'Sex': sex_val, 'Dose': d_val})
            all_nca.append(res)
        
        nca_df = pd.DataFrame(all_nca)
        
        # Result Display based on Context
        if eval_type == "Preclinical (TK/Linearity)":
            st.subheader("🔬 Preclinical Evaluation (Linearity & TK)")
            # Show Dose Proportionality logic
            d_range = sorted(data['Dose'].unique())
            if len(d_range) >= 2:
                st.write("**Dose Proportionality Plot & Power Model**")
                dp_data = nca_df.groupby('Dose').agg({'AUC_last': 'mean', 'Cmax': 'mean'}).reset_index()
                # Power Model: ln(y) = a + b * ln(dose)
                try:
                    log_d = np.log(nca_df['Dose'])
                    log_auc = np.log(nca_df['AUC_last'])
                    slope, intercept, r_val, _, _ = linregress(log_d, log_auc)
                    st.info(f"📈 **Power Model Analysis**: $\\beta = {slope:.3f}$ ($R^2 = {r_val**2:.3f}$)")
                    st.caption("$\beta \approx 1$ indicates dose proportionality. $\beta > 1$ (Supra-linear), $\beta < 1$ (Sub-linear)")
                except: pass

                fig_dp = go.Figure()
                fig_dp.add_trace(go.Scatter(x=dp_data['Dose'], y=dp_data['AUC_last'], mode='lines+markers', name='AUC_last'))
                fig_dp.add_trace(go.Scatter(x=dp_data['Dose'], y=dp_data['Cmax'], mode='lines+markers', name='Cmax', yaxis='y2'))
                fig_dp.update_layout(title="Dose Proportionality", xaxis_title="Dose", yaxis_title="AUC_last",
                                     yaxis2=dict(title="Cmax", overlaying='y', side='right'))
                st.plotly_chart(fig_dp, use_container_width=True)
            else:
                st.info("Need at least 2 unique doses for dose proportionality analysis.")
        else: # Clinical (Variability/Accumulation)
            st.subheader("🏥 Clinical Evaluation (Variability & Steady-state)")
            
            # 1. Inter-subject Variability (CV%)
            if route == 'IV': params_to_check = ['Cmax', 'AUC_last', 'Cl', 'Vz']
            else: params_to_check = ['Cmax', 'AUC_last', 'Cl_F', 'Vz_F']
            available_params = [p for p in params_to_check if p in nca_df.columns and not nca_df[p].isnull().all()]

            if available_params:
                group_cv = nca_df.groupby('Group')[available_params].agg(lambda x: (np.std(x)/np.mean(x)*100) if np.mean(x)!=0 else np.nan)
                st.write("**Inter-subject Variability (CV%)**")
                st.dataframe(group_cv.style.format("{:.1f}%"))

            # 2. Steady-state & Accumulation (Estimation)
            st.write(f"**Steady-state & Accumulation Metrics (Estimated for Tau={tau}h)**")
            acc_results = []
            for g, g_df in nca_df.groupby('Group'):
                lz = g_df['Lambda_z'].mean()
                auc_inf = g_df['AUC_inf'].mean()
                if lz > 0:
                    rac = 1 / (1 - np.exp(-lz * tau))
                    cavg_ss = auc_inf / tau
                    acc_results.append({'Group': g, 'Rac (Accumulation)': rac, 'Cavg,ss (ng/mL)': cavg_ss})
            if acc_results:
                st.dataframe(pd.DataFrame(acc_results).set_index('Group').style.format(precision=2))

            # 3. Bioequivalence-lite (Forest Plot)
            st.write("**Geometric Mean & 90% Confidence Interval (Forest Plot)**")
            be_results = []
            for g, g_df in nca_df.groupby('Group'):
                for p in ['Cmax', 'AUC_last']:
                    vals = g_df[p].values[g_df[p] > 0]
                    if len(vals) >= 2:
                        log_vals = np.log(vals)
                        mean_log = np.mean(log_vals)
                        se_log = np.std(log_vals, ddof=1) / np.sqrt(len(vals))
                        t_val = 1.645 
                        ci_lower, ci_upper = np.exp(mean_log - t_val * se_log), np.exp(mean_log + t_val * se_log)
                        geo_mean = np.exp(mean_log)
                        be_results.append({'Group': g, 'Parameter': p, 'GeoMean': geo_mean, 'Lower': ci_lower, 'Upper': ci_upper})
            
            if be_results:
                be_df = pd.DataFrame(be_results)
                fig_forest = px.scatter(be_df, x='GeoMean', y='Group', color='Parameter', 
                                       error_x=be_df['Upper']-be_df['GeoMean'], 
                                       error_x_minus=be_df['GeoMean']-be_df['Lower'],
                                       title="PK Parameter Forest Plot (90% CI)")
                st.plotly_chart(fig_forest, use_container_width=True)
                st.table(be_df.set_index(['Group', 'Parameter']))

        # Reorder columns for better view
        if route == 'IV':
            cols = ['Group', 'Subject', 'Sex', 'Cmax', 'Tmax', 'AUC_last', 'AUC_inf', 'AUC_%extrap', 't1/2', 'Cl', 'Vz', 'Vss', 'MRT_inf', 'Lambda_z', 'R2_lz']
        else: # Oral
            cols = ['Group', 'Subject', 'Sex', 'Cmax', 'Tmax', 'AUC_last', 'AUC_inf', 'AUC_%extrap', 't1/2', 'Cl_F', 'Vz_F', 'Vss', 'MRT_inf', 'Lambda_z', 'R2_lz']
        
        # Filter columns that actually exist in nca_df
        display_cols = [col for col in cols if col in nca_df.columns]
        st.dataframe(nca_df[display_cols].style.format(precision=4))

        # Download Results
        csv = nca_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download NCA Results (CSV)", csv, "pk_nca_results.csv", "text/csv")

    with tab3:

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
            # Handle t=0 for oral absorption phase
            if isinstance(t, np.ndarray):
                t_safe = np.where(t == 0, 1e-9, t) # Replace 0 with a small number
            else:
                t_safe = 1e-9 if t == 0 else t
            
            # Avoid division by zero if ka == ke
            if np.isclose(ka, ke):
                return (dose * ka / v_f) * t_safe * np.exp(-ke * t_safe)
            else:
                return (dose * ka / (v_f * (ka - ke))) * (np.exp(-ke * t_safe) - np.exp(-ka * t_safe))
        
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
                # Initial guess for oral: ka > ke, V/F reasonable
                popt1, pcov1 = curve_fit(one_comp_oral_fit, t_avg, c_avg, p0=[1.0, 0.1, 10.0], bounds=([0.001, 0.001, 0.001], [np.inf, np.inf, np.inf]), sigma=weights)
            
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
            avg_hl = nca_g['t1/2'].mean() # Changed to t1/2
            avg_r2 = nca_g['R2_lz'].mean() # Changed to R2_lz
            
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
    render_data_input_sidebar('tmdd_manual', generate_3x3_tmdd_example, (), "TMDD Obs")
    
    st.subheader("✍️ TMDD Observation Data Editor")
    data = st.data_editor(st.session_state['tmdd_manual'], num_rows="dynamic", use_container_width=True)
    st.session_state['tmdd_manual'] = data
    
    st.sidebar.divider()
    st.sidebar.subheader("TMDD Model Parameters")
    params = {
        'kel': st.sidebar.slider("Elimination (kel)", 0.001, 0.5, 0.02),
        'kon': st.sidebar.slider("On-rate (kon)", 0.01, 2.0, 0.1),
        'koff': st.sidebar.slider("Off-rate (koff)", 0.001, 0.5, 0.01),
        'kint': st.sidebar.slider("Internalization (kint)", 0.001, 0.5, 0.05),
        'ksyn': st.sidebar.slider("Target Syn (ksyn)", 0.1, 10.0, 1.0),
        'kdeg': st.sidebar.slider("Target Deg (kdeg)", 0.01, 0.5, 0.1)
    }
    t_end = st.sidebar.number_input("End Time (hr)", value=168)

    # Simulation logic using parameters and THE FIRST dose from the table as a reference, 
    # but the plot will show ALL data points from the table.
    primary_dose = data['Dose'].iloc[0] if not data.empty else 100
    R0 = params['ksyn'] / params['kdeg']
    y0 = [primary_dose, R0, 0]
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, 500)
    sol = solve_ivp(tmdd_model_ode, t_span, y0, t_eval=t_eval, args=(params,), method='RK45')
    
    # Plotting
    st.subheader("📉 TMDD Simulation & Observations")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Simulation Lines
    ax1.plot(sol.t, sol.y[0], 'k-', alpha=0.5, label="Sim: Free Drug (L)")
    
    # Observation Points from Editor
    groups = data['Group'].unique()
    # Plotly TMDD
    fig_tmdd = go.Figure()
    # Observation points
    # Ensure tmdd_data is defined, assuming it's 'data' from the editor
    tmdd_data = data 
    fig_tmdd.add_trace(go.Scatter(x=tmdd_data['Time'], y=tmdd_data['Concentration'], mode='markers', name='Observed Conc', marker=dict(symbol='circle-open')))
    # Simulation lines - using unique doses from the data editor
    d_range = sorted(tmdd_data['Dose'].unique())
    for i, d in enumerate(d_range):
        y0_sim = [d, R0, 0] # Use 'd' as the initial dose for simulation
        sol_sim = solve_ivp(tmdd_model_ode, (0, t_end), y0_sim, t_eval=np.linspace(0, t_end, 100), args=(params,))
        fig_tmdd.add_trace(go.Scatter(x=sol_sim.t, y=sol_sim.y[0], mode='lines', name=f"Sim {d}nM"))
    fig_tmdd.update_layout(title="TMDD Simulation vs. Observed", xaxis_title="Time", yaxis_title="Concentration", yaxis_type='log')
    st.plotly_chart(fig_tmdd, use_container_width=True)

    ro = (sol.y[2] / (sol.y[1] + sol.y[2])) * 100
    ax2.plot(sol.t, ro, color='black', label="Sim: Receptor Occupancy")
    ax2.set_ylabel("RO (%)")
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig)

elif mode == "PK/PD Correlation":
    render_data_input_sidebar('pkpd_manual', generate_3x3_example, (route,), "PK/PD Data")
    
    st.subheader("✍️ PK/PD Study Data Editor")
    data = st.data_editor(st.session_state['pkpd_manual'], num_rows="dynamic", use_container_width=True)
    st.session_state['pkpd_manual'] = data
    
    st.sidebar.divider()
    st.sidebar.subheader("PK/PD Parameters")
    vd = st.sidebar.number_input("Volume (Vd, L)", value=10.0)
    cl = st.sidebar.number_input("Clearance (Cl, L/hr)", value=1.0)
    
    st.sidebar.subheader("PD Parameters (Sigmoid Emax)")
    emax = st.sidebar.number_input("Emax", value=100.0)
    ec50 = st.sidebar.number_input("EC50", value=20.0)
    gamma = st.sidebar.slider("Hill Coefficient (gamma)", 0.5, 5.0, 1.0)
    keo = st.sidebar.slider("Equilibrium (keo, 1/hr)", 0.01, 2.0, 0.2)
    
    t_end = st.sidebar.number_input("Simulation Time (hr)", value=48)
    dose_norm = st.sidebar.checkbox("Dose-Normalized Scale (C/Dose, E/Dose)", value=False)

    # Use unique doses from table to drive simulation lines
    doses = sorted(data['Dose'].unique())

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
        ax_pk.plot(sol.t, cp / scale, color=color, label=f"Sim Dose {dose}")
        # Plot points from data table
        dose_data = data[data['Dose'] == dose]
        ax_pk.scatter(dose_data['Time'], dose_data['Concentration'] / scale, color=color, alpha=0.6, label=f"Obs Dose {dose}")
        
        # Plot PD
        ax_pd.plot(sol.t, effect / scale, color=color, ls='--', label=f"Sim Eff (Dose {dose})")
        
        # Hysteresis
        ax_hys.plot(cp, effect, color=color, label=f"Dose {dose}")
        
        # Summary Params
        cmax_sim = np.max(cp)
        tmax_sim = sol.t[np.argmax(cp)]
        auc_sim = trapezoid(cp, sol.t)
        
        # PD Metrics
        peak_eff = np.max(effect)
        teff_max = sol.t[np.argmax(effect)]
        auec = trapezoid(effect, sol.t)
        e0 = effect[0]
        
        all_results.append({
            'Dose': dose, 'Cmax': cmax_sim, 'Tmax': tmax_sim, 'AUC': auc_sim,
            'Peak_Effect': peak_eff, 'Teff_max': teff_max, 'AUEC': auec, 'E0': e0
        })
        
        summary_data.append({
            'Dose': dose,
            'Cmax': cmax_sim,
            'AUC': auc_sim,
            'Peak_Effect': peak_eff,
            'T_eff_max': teff_max,
            'Cmax/Dose': cmax_sim / dose,
            'AUC/Dose': auc_sim / dose,
            'Effect/Dose': peak_eff / dose
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

    st.subheader("👨‍👩‍👧‍👦 Population PK Simulation (IIV)")
    st.info("개인 간 변동성(Inter-Individual Variability)을 고려한 집단 PK 시뮬레이션입니다.")
    
    if 'pop_manual' not in st.session_state:
        # Create a small population table
        st.session_state['pop_manual'] = pd.DataFrame({
            'Subject': [f'Subj_{i+1}' for i in range(9)],
            'Dose': [10]*3 + [30]*3 + [100]*3,
            'Cl_Baseline': [2.0]*9,
            'V_Baseline': [20.0]*9
        })
    
    if st.sidebar.button("🔄 Reset Pop Example"):
        st.session_state['pop_manual'] = pd.DataFrame({
            'Subject': [f'Subj_{i+1}' for i in range(9)],
            'Dose': [10]*3 + [30]*3 + [100]*3,
            'Cl_Baseline': [2.0]*9,
            'V_Baseline': [20.0]*9
        })
        
    st.subheader("✍️ Population Metadata Editor (3 Doses, N=3)")
    pop_meta = st.data_editor(st.session_state['pop_manual'], use_container_width=True)
    st.session_state['pop_manual'] = pop_meta

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

    t_eval = np.linspace(0, 48, 100)
    pop_results = []
    
    # Simulation based on metadata table
    for idx, row in pop_meta.iterrows():
        # Apply variability to baseline values in table
        cl_i = row['Cl_Baseline'] * np.exp(np.random.normal(0, cv_cl/100))
        v_i = row['V_Baseline'] * np.exp(np.random.normal(0, cv_v/100))
        ke_i = cl_i / v_i
        
        if route == "IV":
            cp_i = (row['Dose'] / v_i) * np.exp(-ke_i * t_eval)
        else:
            ka_i = 1.5 # Default ka for pop simulation
            cp_i = (row['Dose'] * ka_i / (v_i * (ka_i - ke_i))) * (np.exp(-ke_i * t_eval) - np.exp(-ka_i * t_eval))
            
        pop_results.append(cp_i)
    
    pop_array = np.array(pop_results)
    n_subj_actual = len(pop_array)
    p5 = np.percentile(pop_array, 5, axis=0)
    p50 = np.percentile(pop_array, 50, axis=0)
    p95 = np.percentile(pop_array, 95, axis=0)
    
    fig_pop = go.Figure()
    for i in range(min(n_subj_actual, 50)): 
        fig_pop.add_trace(go.Scatter(x=t_eval, y=pop_array[i], mode='lines', line=dict(color='gray', width=1), opacity=0.1, showlegend=False))
    
    fig_pop.add_trace(go.Scatter(x=t_eval, y=p50, mode='lines', name="Median (P50)", line=dict(color='red', width=3)))
    fig_pop.add_trace(go.Scatter(x=t_eval, y=p95, mode='lines', line=dict(width=0), showlegend=False))
    fig_pop.add_trace(go.Scatter(x=t_eval, y=p5, mode='lines', fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)', name="90% Prediction Interval"))
    
    fig_pop.update_layout(title=f"Population PK Simulation (N={n_subj_actual})", xaxis_title="Time (hr)", yaxis_title="Concentration", yaxis_type='log')
    st.plotly_chart(fig_pop, use_container_width=True)
    
    st.success(f"✅ Simulation Complete for N={n_subj_actual} subjects with IIV (Cl CV {cv_cl}%, V CV {cv_v}%).")

elif mode == "Dose-Response & PD Modeling":
    render_data_input_sidebar('pd_manual', generate_pd_3x3_example, (), "Dose-Response & PD")
    
    st.subheader("🎯 Dose-Response & Advanced PD Modeling")
    pd_data = st.data_editor(st.session_state['pd_manual'], num_rows="dynamic", use_container_width=True)
    st.session_state['pd_manual'] = pd_data
    
    if not pd_data.empty:
        # Aggregation
        avg_data = pd_data.groupby(['Group', 'Time']).agg({'Concentration': 'mean', 'Effect': 'mean', 'Dose': 'first'}).reset_index()
        
        # 1. PD Model Recommendation
        st.subheader("💡 Intelligent PD Model Recommendation")
        c_vals = avg_data['Effect'].values
        conc_vals = avg_data['Concentration'].values
        
        # AUEC for each group
        st.write("**PD Area Under Effect Curve (AUEC) & Baseline (E0)**")
        pd_sum_metrics = pd_data.groupby('Group').apply(lambda x: pd.Series({
            'AUEC_last': trapezoid(x.sort_values('Time')['Effect'], x.sort_values('Time')['Time']),
            'E0': x.sort_values('Time')['Effect'].iloc[0],
            'Teff_max': x.sort_values('Time')['Time'].iloc[np.argmax(x.sort_values('Time')['Effect'])]
        })).reset_index()
        st.dataframe(pd_sum_metrics.style.format(precision=2))
        
        pd_rec_results = {}
        # Try Emax
        try:
            popt_e, pcov_e = curve_fit(pd_emax, conc_vals, c_vals, p0=[max(c_vals), np.median(conc_vals)], bounds=(0, np.inf))
            rss = np.sum((c_vals - pd_emax(conc_vals, *popt_e))**2)
            aic = 2*2 + len(c_vals)*np.log(rss/len(c_vals))
            pd_rec_results['Emax'] = {'aic': aic, 'popt': popt_e, 'func': pd_emax, 'name': ['Emax', 'EC50']}
        except: pass
        
        # Try Sigmoid
        try:
            popt_s, pcov_s = curve_fit(pd_sigmoid, conc_vals, c_vals, p0=[max(c_vals), np.median(conc_vals), 1.0], bounds=(0, np.inf))
            rss = np.sum((c_vals - pd_sigmoid(conc_vals, *popt_s))**2)
            aic = 2*3 + len(c_vals)*np.log(rss/len(c_vals))
            pd_rec_results['Sigmoid'] = {'aic': aic, 'popt': popt_s, 'func': pd_sigmoid, 'name': ['Emax', 'EC50', 'Gamma']}
        except: pass
        
        # Try Linear
        try:
            popt_l, pcov_l = curve_fit(pd_linear, conc_vals, c_vals, p0=[1.0], bounds=(0, np.inf))
            rss = np.sum((c_vals - pd_linear(conc_vals, *popt_l))**2)
            aic = 2*1 + len(c_vals)*np.log(rss/len(c_vals))
            pd_rec_results['Linear'] = {'aic': aic, 'popt': popt_l, 'func': pd_linear, 'name': ['Slope']}
        except: pass

        if pd_rec_results:
            best_pd = min(pd_rec_results, key=lambda x: pd_rec_results[x]['aic'])
            st.success(f"✅ **Best PD Model: {best_pd}**")
            
            # Display Params
            b_info = pd_rec_results[best_pd]
            p_df = pd.DataFrame({'Parameter': b_info['name'], 'Estimate': b_info['popt']})
            st.table(p_df.set_index('Parameter'))

            # Plots
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Conc vs. Effect (PD Model - Interactive)**")
                c_range = np.linspace(min(conc_vals), max(conc_vals), 100)
                fig_pd = go.Figure()
                fig_pd.add_trace(go.Scatter(x=conc_vals, y=c_vals, mode='markers', name='Observed'))
                fig_pd.add_trace(go.Scatter(x=c_range, y=b_info['func'](c_range, *b_info['popt']), mode='lines', name=f'Fit: {best_pd}'))
                fig_pd.update_layout(xaxis_title="Concentration", yaxis_title="Effect")
                st.plotly_chart(fig_pd, use_container_width=True)

            with col2:
                st.write("**Dose vs. Response (Hill Analysis)**")
                dr_data = pd_data.groupby('Group').agg({'Dose': 'first', 'Effect': 'max'}).reset_index().sort_values('Dose')
                fig_dr = px.line(dr_data, x='Dose', y='Effect', markers=True, log_x=True, title="Dose-Response Curve")
                st.plotly_chart(fig_dr, use_container_width=True)
                
                # Hill Fit for Dose-Response
                try:
                    popt_dr, _ = curve_fit(pd_emax, dr_data['Dose'], dr_data['Effect'], p0=[max(dr_data['Effect']), np.median(dr_data['Dose'])])
                    st.write(f"📈 **Dose-Response Hill Fit**: $ED_{{50}}$ = {popt_dr[1]:.2f} mg")
                except: pass

elif mode == "Parent-Metabolite Modeling":
    render_data_input_sidebar('pm_data', generate_pm_example, (), "Parent-Metabolite")
    
    st.subheader("🧬 Parent-Metabolite Integrated Modeling")
    pm_df = st.data_editor(st.session_state['pm_data'], num_rows="dynamic", use_container_width=True)
    st.session_state['pm_data'] = pm_df

    col_p, col_m = st.columns(2)
    with col_p:
        n_p = st.selectbox("Parent Compartments", [1, 2, 3], index=0)
    with col_m:
        n_m = st.selectbox("Metabolite Compartments", [1, 2], index=0)
    
    if not pm_df.empty:
        fig_pm = px.line(pm_df, x='Time', y='Concentration', color='Analyte', markers=True, title="Parent & Metabolite Profiles")
        st.plotly_chart(fig_pm, use_container_width=True)
        
        # Fitting Logic
        if st.button("🚀 Run Integrated Fitting"):
            # Prepare data
            t_eval = sorted(pm_df['Time'].unique())
            p_data = pm_df[pm_df['Analyte'] == 'Parent'].groupby('Time')['Concentration'].mean().reindex(t_eval, fill_value=0).values
            m_data = pm_df[pm_df['Analyte'] == 'Metabolite'].groupby('Time')['Concentration'].mean().reindex(t_eval, fill_value=0).values
            
            combined_obs = np.concatenate([p_data, m_data])
            
            def fit_func(t, kel_p, k_pm, kel_m):
                p_sim = {'kel_p': kel_p, 'k_pm': k_pm, 'kel_m': kel_m, 'n_p': n_p, 'n_m': n_m}
                y0 = [100] + [0]*(n_p + n_m)
                sol = solve_ivp(parent_metabolite_model_ode, (0, max(t_eval)), y0, t_eval=t_eval, args=(p_sim,))
                return np.concatenate([sol.y[0], sol.y[n_p]])

            try:
                popt, _ = curve_fit(fit_func, t_eval, combined_obs, p0=[0.1, 0.05, 0.1], bounds=(0, np.inf))
                st.success(f"Fitting Successful! kel_p: {popt[0]:.4f}, k_pm: {popt[1]:.4f}, kel_m: {popt[2]:.4f}")
                
                # Show Fit
                p_final = {'kel_p': popt[0], 'k_pm': popt[1], 'kel_m': popt[2], 'n_p': n_p, 'n_m': n_m}
                y0_final = [100] + [0]*(n_p + n_m)
                sol_fit = solve_ivp(parent_metabolite_model_ode, (0, max(t_eval)), y0_final, t_eval=np.linspace(0, max(t_eval), 100), args=(p_final,))
                fig_fit = go.Figure()
                fig_fit.add_trace(go.Scatter(x=pm_df[pm_df['Analyte']=='Parent']['Time'], y=pm_df[pm_df['Analyte']=='Parent']['Concentration'], mode='markers', name='Parent Obs'))
                fig_fit.add_trace(go.Scatter(x=sol_fit.t, y=sol_fit.y[0], mode='lines', name='Parent Fit'))
                fig_fit.add_trace(go.Scatter(x=pm_df[pm_df['Analyte']=='Metabolite']['Time'], y=pm_df[pm_df['Analyte']=='Metabolite']['Concentration'], mode='markers', name='Metabolite Obs'))
                fig_fit.add_trace(go.Scatter(x=sol_fit.t, y=sol_fit.y[n_p], mode='lines', name='Metabolite Fit'))
                st.plotly_chart(fig_fit, use_container_width=True)
            except Exception as e:
                st.error(f"Fitting Failed: {e}")

elif mode == "Full PBPK Engine":
    st.subheader("🧬 Full Physiology-Based Pharmacokinetic (PBPK) Engine")
    st.info("Simcyp/GastroPlus 표준을 따르는 장기별 약동학 시뮬레이션 및 민감도 분석을 수행합니다.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Physiological Parameters**")
        species = st.selectbox("Species Reference", ["Human", "Rat"])
        phy = PBPK_PHYSIOLOGY[species]
        
        st.write(f"Body Weight: {phy['BW']} kg")
        st.write(f"Cardiac Output: {phy['QC']} L/h")
        
        st.divider()
        st.write("**Compound Specific (Drug Properties)**")
        dose_pbpk = st.number_input("Dose (mg, IV Bolus)", value=100.0)
        cl_liver = st.slider("Hepatic Clearance (CL_liver, L/h)", 0.0, 50.0, 5.0)
        cl_renal = st.slider("Renal Clearance (CL_renal, L/h)", 0.0, 20.0, 1.0)
        
        kp_val = st.slider("Tissue Partition Coefficient (Kp - Global)", 0.1, 10.0, 1.0)
        kp_dict = {o: kp_val for o in ["Liver", "Kidney", "Gut", "Lung", "Rest"]}
        
    with col2:
        # Simulation
        t_eval = np.linspace(0, 48, 240)
        params = {
            'V': phy['V'], 'Q': phy['Q'], 'QC': phy['QC'],
            'Kp': kp_dict, 'CL_liver': cl_liver, 'CL_renal': cl_renal
        }
        # Initial condition: Dose in Venous Blood (as a simple start for IV bolus)
        # y = [C_liver, C_kidney, C_gut, C_lung, C_rest, C_venous, C_arterial]
        y0 = [0, 0, 0, 0, 0, dose_pbpk/0.1, 0] 
        sol = solve_ivp(pbpk_model_ode, (0, 48), y0, t_eval=t_eval, args=(params,))
        
        st.write("**Organ Distribution Simulation**")
        fig_pbpk = go.Figure()
        organs_to_plot = [("Liver", 0), ("Kidney", 1), ("Arterial", 6)]
        for name, idx in organs_to_plot:
            fig_pbpk.add_trace(go.Scatter(x=sol.t, y=sol.y[idx], mode='lines', name=name))
        fig_pbpk.update_layout(xaxis_title="Time (hr)", yaxis_title="Concentration (mg/L)", yaxis_type='log')
        st.plotly_chart(fig_pbpk, use_container_width=True)

        # Interactive PBPK Schematic (Phase 3 Visual)
        st.divider()
        st.write("**PBPK Model Schematic (GastroPlus Style)**")
        fig_map = go.Figure()
        # Nodes: Venous, Lung, Arterial, Gut, Liver, Kidney, Rest
        nodes = {
            "Venous": [1, 2], "Lung": [2, 3], "Arterial": [3, 2],
            "Gut": [4, 1], "Liver": [4, 2], "Kidney": [4, 3], "Rest": [4, 4]
        }
        for name, pos in nodes.items():
            fig_map.add_trace(go.Scatter(x=[pos[0]], y=[pos[1]], mode='markers+text', 
                                         marker=dict(size=40, color='lightblue'),
                                         text=[name], textposition="middle center", name=name))
        # Edges
        fig_map.add_annotation(x=1, y=2, ax=2, ay=3, xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True) # V to Lu
        fig_map.add_annotation(x=2, y=3, ax=3, ay=2, xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True) # Lu to A
        fig_map.add_annotation(x=3, y=2, ax=4, ay=4, xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True) # A to R
        fig_map.add_annotation(x=4, y=4, ax=1, ay=2, xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True) # R to V
        
        fig_map.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), height=300, title="Structural PBPK Model Map")
        st.plotly_chart(fig_map, use_container_width=True)

        # Sensitivity Analysis (Tornado Plot)
        st.divider()
        st.subheader("🌪️ Sensitivity Analysis (Tornado Plot)")
        st.caption("파라미터 변화가 Cmax(Arterial)에 미치는 영향을 분석합니다. (±20% Variation)")
        
        sensitivity_data = []
        base_cmax = np.max(sol.y[6])
        
        test_params = [
            ("CL_liver", cl_liver), ("CL_renal", cl_renal), ("Kp", kp_val)
        ]
        
        for p_name, p_val in test_params:
            for multiplier in [0.8, 1.2]:
                # Temporary params
                tmp_p = params.copy()
                if p_name == "Kp": tmp_p['Kp'] = {o: p_val*multiplier for o in tmp_p['Kp']}
                else: tmp_p[p_name] = p_val * multiplier
                
                # Re-simulate
                tmp_sol = solve_ivp(pbpk_model_ode, (0, 48), y0, t_eval=t_eval, args=(tmp_p,))
                tmp_cmax = np.max(tmp_sol.y[6])
                change = (tmp_cmax - base_cmax) / base_cmax * 100
                sensitivity_data.append({"Parameter": p_name, "Change": change, "Direction": "Up" if multiplier > 1 else "Down"})
        
        sens_df = pd.DataFrame(sensitivity_data)
        fig_tornado = px.bar(sens_df, x='Change', y='Parameter', color='Direction', orientation='h',
                             title="Sensitivity Analysis: Effect on Cmax",
                             color_discrete_map={"Up": "firebrick", "Down": "steelblue"})
        st.plotly_chart(fig_tornado, use_container_width=True)

st.divider()

# --- Report Generation & Audit Trail Viewer ---
st.subheader("📋 Professional Analysis Report & History")
col_rep1, col_rep2 = st.columns(2)

with col_rep1:
    if st.button("📄 Generate Comprehensive HTML Report"):
        # Audit Trail for current project if loaded
        audit_info = ""
        current_p = st.session_state.get('loaded_project_name', project_name_input)
        trails = pk_db.get_audit_trail(current_p)
        if not trails.empty:
            audit_info = "<h3>Audit Trail (Project History)</h3>" + trails.to_html(index=False)

        report_html = f"""
        <html>
        <head><style>
            body {{ font-family: 'Segoe UI', sans-serif; margin: 40px; color: #333; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #2c3e50; }}
            h2 {{ color: #2980b9; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f8f9fa; }}
            .footer {{ margin-top: 50px; font-size: 0.8em; color: #7f8c8d; border-top: 1px solid #eee; padding-top: 10px; }}
        </style></head>
        <body>
            <h1>PK Analysis Certificate of Analysis (CoA)</h1>
            <p><strong>Project:</strong> {current_p} | <strong>Date:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
            
            <h2>1. Analysis Scope & Methodology</h2>
            <table>
                <tr><th>Category</th><th>Details</th></tr>
                <tr><td>Analysis Mode</td><td>{mode}</td></tr>
                <tr><td>Context</td><td>{eval_type}</td></tr>
                <tr><td>Route of Administration</td><td>{route}</td></tr>
                <tr><td>LLOQ Filtering</td><td>{lloq} (Method: {blq_method})</td></tr>
            </table>

            <h2>2. Executive Summary</h2>
            <p>This automated report summarizes the pharmacometric analysis results. The analysis followed professional industry standards (WinNonlin / Simcyp comparable metrics).</p>
            
            {audit_info}
            
            <div class="footer">
                Developed by Antigravity PK Engine | Enterprise Gold Standard Phase 3 | Secure SQLite Persistence
            </div>
        </body>
        </html>
        """
        st.download_button("📥 Download PDF/HTML Report", report_html, "pk_coa_report.html", "text/html")

with col_rep2:
    if st.button("🔍 View Project Audit Trail"):
        current_p = st.session_state.get('loaded_project_name', project_name_input)
        st.write(f"**Audit Trail for '{current_p}'**")
        trails = pk_db.get_audit_trail(current_p)
        if not trails.empty:
            st.dataframe(trails, use_container_width=True)
        else:
            st.info("No audit trail found for this project.")

st.divider()
st.caption("Developed by Antigravity PK Engine | Phase 3: Commercial-Grade Power-Up")
