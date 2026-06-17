import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KNN DDoS Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #000000;
    color: #E1DCC9;
  }

  .stApp {
    background-color: #000000;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background-color: #0A0804 !important;
    border-right: 1px solid #2A1E10;
  }
  [data-testid="stSidebar"] * {
    color: #E1DCC9 !important;
  }
  [data-testid="stSidebarNav"] a {
    border-radius: 6px;
    margin: 2px 0;
  }

  /* ── Headers ── */
  h1, h2, h3 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    color: #E1DCC9;
    letter-spacing: -0.02em;
  }
  h1 { font-size: 1.85rem; }
  h2 { font-size: 1.35rem; border-bottom: 1px solid #2A1E10; padding-bottom: 0.5rem; }
  h3 { font-size: 1.05rem; color: #B8B09A; }

  /* ── Metric cards ── */
  .metric-card {
    background: #0F0905;
    border: 1px solid #2A1E10;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 0;
    transition: border-color 0.2s;
  }
  .metric-card:hover { border-color: #412D15; }
  .metric-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7A6E5A;
    margin-bottom: 0.45rem;
  }
  .metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #E1DCC9;
    font-variant-numeric: tabular-nums;
    line-height: 1;
  }
  .metric-sub {
    font-size: 0.75rem;
    color: #5A5040;
    margin-top: 0.35rem;
  }

  /* ── Status banner ── */
  .status-normal {
    background: #081408;
    border: 1px solid #1E3D1E;
    border-left: 4px solid #2D6A2D;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    font-size: 1.05rem;
    font-weight: 600;
    color: #7EC87E;
    letter-spacing: 0.01em;
  }
  .status-attack {
    background: #140808;
    border: 1px solid #3D1E1E;
    border-left: 4px solid #8B2020;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    font-size: 1.05rem;
    font-weight: 600;
    color: #E07070;
    letter-spacing: 0.01em;
  }
  .status-label {
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 500;
    opacity: 0.6;
    margin-bottom: 0.3rem;
  }

  /* ── Section divider ── */
  .section-divider {
    border: none;
    border-top: 1px solid #1F150C;
    margin: 1.8rem 0;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: #412D15;
    color: #E1DCC9;
    border: 1px solid #5A3E1E;
    border-radius: 7px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 0.88rem;
    letter-spacing: 0.02em;
    padding: 0.6rem 1.8rem;
    transition: all 0.2s;
    width: 100%;
  }
  .stButton > button:hover {
    background: #5A3E1E;
    border-color: #7A5530;
    color: #F5F0E0;
  }
  .stButton > button:active {
    background: #2E1F0C;
    transform: translateY(1px);
  }

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {
    background: #0F0905;
    border: 1px dashed #2A1E10;
    border-radius: 10px;
    padding: 1rem;
  }
  [data-testid="stFileUploader"]:hover {
    border-color: #412D15;
  }

  /* ── Slider ── */
  .stSlider [data-baseweb="slider"] { padding: 0.5rem 0; }

  /* ── Select box ── */
  .stSelectbox [data-baseweb="select"] > div {
    background: #0F0905 !important;
    border-color: #2A1E10 !important;
    color: #E1DCC9 !important;
  }

  /* ── Tables ── */
  .stDataFrame {
    border: 1px solid #2A1E10;
    border-radius: 8px;
    overflow: hidden;
  }
  [data-testid="stDataFrameResizable"] * {
    background-color: #0A0804 !important;
    color: #E1DCC9 !important;
    font-size: 0.82rem !important;
  }

  /* ── Progress bar ── */
  .stProgress > div > div > div > div {
    background: #412D15;
  }

  /* ── Info / warning boxes ── */
  .stAlert {
    background: #0F0905 !important;
    border: 1px solid #2A1E10 !important;
    color: #E1DCC9 !important;
    border-radius: 8px !important;
  }

  /* ── Sidebar nav items ── */
  .sidebar-nav-item {
    display: block;
    padding: 0.6rem 0.9rem;
    border-radius: 7px;
    font-size: 0.85rem;
    font-weight: 500;
    color: #9A8E7A;
    cursor: pointer;
    margin-bottom: 3px;
    border: 1px solid transparent;
    transition: all 0.15s;
    letter-spacing: 0.01em;
  }
  .sidebar-nav-item:hover {
    background: #1F150C;
    color: #E1DCC9;
    border-color: #2A1E10;
  }
  .sidebar-nav-item.active {
    background: #2A1C0E;
    color: #E1DCC9;
    border-color: #412D15;
    font-weight: 600;
  }
  .sidebar-nav-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    margin-right: 10px;
    background: #412D15;
    vertical-align: middle;
    margin-bottom: 1px;
  }
  .sidebar-nav-item.active .sidebar-nav-dot {
    background: #C8A86A;
  }

  /* ── Page header ── */
  .page-header {
    padding: 1.2rem 0 0.8rem 0;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid #1F150C;
  }
  .page-title {
    font-size: 1.65rem;
    font-weight: 700;
    color: #E1DCC9;
    letter-spacing: -0.025em;
    line-height: 1.2;
  }
  .page-subtitle {
    font-size: 0.82rem;
    color: #6A5E4A;
    margin-top: 0.3rem;
    letter-spacing: 0.01em;
  }

  /* ── Chart container ── */
  .chart-container {
    background: #0A0804;
    border: 1px solid #1F150C;
    border-radius: 10px;
    padding: 0.5rem;
  }

  /* ── Tag badges ── */
  .badge-normal {
    display: inline-block;
    background: #0A1A0A;
    color: #6BBF6B;
    border: 1px solid #1E4A1E;
    border-radius: 4px;
    padding: 0.15rem 0.6rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .badge-ddos {
    display: inline-block;
    background: #1A0A0A;
    color: #CF6060;
    border: 1px solid #4A1E1E;
    border-radius: 4px;
    padding: 0.15rem 0.6rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }

  /* ── Sidebar brand ── */
  .sidebar-brand {
    padding: 1.2rem 0.5rem 1.5rem 0.5rem;
    border-bottom: 1px solid #2A1E10;
    margin-bottom: 1rem;
  }
  .sidebar-brand-title {
    font-size: 1rem;
    font-weight: 700;
    color: #E1DCC9;
    letter-spacing: -0.01em;
  }
  .sidebar-brand-sub {
    font-size: 0.7rem;
    color: #5A5040;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-top: 0.2rem;
  }

  /* ── Mono text ── */
  .mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #080604; }
  ::-webkit-scrollbar-thumb { background: #2A1E10; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #412D15; }

  /* ── Hide default elements ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* ── Plotly chart background fix ── */
  .js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─── Session state ───────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "page": "Dashboard",
        "df": None,
        "model": None,
        "scaler": None,
        "X_test": None,
        "y_test": None,
        "y_pred": None,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "cm": None,
        "df_result": None,
        "trained": False,
        "k_value": 5,
        "test_size": 0.2,
        "feature_cols": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Plotly theme ─────────────────────────────────────────────────────────────
PLOT_BGCOLOR = "#080604"
PAPER_BGCOLOR = "#0A0804"
GRID_COLOR = "#1A1208"
TEXT_COLOR = "#9A8E7A"
ACCENT = "#C8A86A"
BROWN_MED = "#412D15"
CREAM = "#E1DCC9"

def plotly_layout(title="", height=360):
    return dict(
        title=dict(text=title, font=dict(family="Inter", size=13, color=CREAM), x=0.02, xanchor="left"),
        paper_bgcolor=PAPER_BGCOLOR,
        plot_bgcolor=PLOT_BGCOLOR,
        font=dict(family="Inter", color=TEXT_COLOR, size=11),
        height=height,
        margin=dict(l=50, r=30, t=45, b=45),
        xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, tickcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, tickcolor=GRID_COLOR),
    )

LEGEND_H = dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2A1E10", borderwidth=1,
                orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5,
                font=dict(color=CREAM, size=10))


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
      <div class="sidebar-brand-title">KNN Detection</div>
      <div class="sidebar-brand-sub">DDoS Traffic Classifier</div>
    </div>
    """, unsafe_allow_html=True)

    pages = ["Dashboard", "Dataset", "Training Model", "Detection Result"]
    for p in pages:
        is_active = st.session_state.page == p
        active_cls = "active" if is_active else ""
        if st.button(p, key=f"nav_{p}", use_container_width=True):
            st.session_state.page = p
            st.rerun()

    st.markdown("<hr style='border-color:#1F150C; margin: 1.5rem 0 1rem 0'>", unsafe_allow_html=True)

    # Status indicator
    if st.session_state.trained:
        st.markdown("""
        <div style="font-size:0.7rem;letter-spacing:0.08em;text-transform:uppercase;color:#5A5040;margin-bottom:0.5rem;">Model Status</div>
        <div style="display:flex;align-items:center;gap:8px;">
          <div style="width:8px;height:8px;border-radius:50%;background:#2D6A2D;box-shadow:0 0 6px #2D6A2D;"></div>
          <span style="font-size:0.82rem;font-weight:500;color:#7EC87E;">Model Trained</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size:0.7rem;letter-spacing:0.08em;text-transform:uppercase;color:#5A5040;margin-bottom:0.5rem;">Model Status</div>
        <div style="display:flex;align-items:center;gap:8px;">
          <div style="width:8px;height:8px;border-radius:50%;background:#5A3E1E;"></div>
          <span style="font-size:0.82rem;font-weight:500;color:#6A5E4A;">Not Trained</span>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.df is not None:
        n = len(st.session_state.df)
        st.markdown(f"""
        <div style="margin-top:1rem;font-size:0.7rem;letter-spacing:0.08em;text-transform:uppercase;color:#5A5040;margin-bottom:0.4rem;">Dataset</div>
        <div style="font-size:0.82rem;color:#9A8E7A;">{n:,} records loaded</div>
        """, unsafe_allow_html=True)


page = st.session_state.page


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.markdown("""
    <div class="page-header">
      <div class="page-title">Network Traffic Analysis Dashboard</div>
      <div class="page-subtitle">K-Nearest Neighbors — DDoS Attack Detection System</div>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        n = len(st.session_state.df) if st.session_state.df is not None else 0
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Total Records</div>
          <div class="metric-value">{n:,}</div>
          <div class="metric-sub">{'Dataset loaded' if n > 0 else 'No dataset'}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        if st.session_state.df is not None and "Label" in st.session_state.df.columns:
            n_normal = int((st.session_state.df["Label"] == 0).sum())
        else:
            n_normal = 0
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Normal Traffic</div>
          <div class="metric-value" style="color:#7EC87E;">{n_normal:,}</div>
          <div class="metric-sub">Label = 0</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        if st.session_state.df is not None and "Label" in st.session_state.df.columns:
            n_ddos = int((st.session_state.df["Label"] == 1).sum())
        else:
            n_ddos = 0
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">DDoS Traffic</div>
          <div class="metric-value" style="color:#E07070;">{n_ddos:,}</div>
          <div class="metric-sub">Label = 1</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        acc = f"{st.session_state.accuracy*100:.2f}%" if st.session_state.accuracy is not None else "—"
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Model Accuracy</div>
          <div class="metric-value" style="color:{ACCENT};">{acc}</div>
          <div class="metric-sub">{'KNN k=' + str(st.session_state.k_value) if st.session_state.trained else 'Not trained'}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Charts row (only if data is loaded)
    if st.session_state.df is not None and "Label" in st.session_state.df.columns:
        col_a, col_b = st.columns([1, 1])
        df = st.session_state.df

        with col_a:
            counts = df["Label"].value_counts().sort_index()
            labels_map = {0: "Normal", 1: "DDoS"}
            pie_labels = [labels_map.get(i, str(i)) for i in counts.index]
            pie_vals = counts.values.tolist()

            fig_pie = go.Figure(go.Pie(
                labels=pie_labels,
                values=pie_vals,
                hole=0.58,
                marker=dict(colors=["#2D6A2D", "#8B2020"],
                            line=dict(color=PLOT_BGCOLOR, width=2)),
                textfont=dict(family="Inter", size=12, color=CREAM),
                hovertemplate="<b>%{label}</b><br>%{value:,} records<br>%{percent}<extra></extra>",
            ))
            fig_pie.update_layout(
                **plotly_layout("Traffic Distribution", height=320),
                showlegend=True,
                legend=LEGEND_H,
                annotations=[dict(text=f"{len(df):,}<br><span style='font-size:10px'>records</span>",
                                   x=0.5, y=0.5, showarrow=False,
                                   font=dict(size=16, color=CREAM, family="Inter"),
                                   align="center")]
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

        with col_b:
            bar_data = pd.DataFrame({"Category": pie_labels, "Count": pie_vals})
            fig_bar = go.Figure(go.Bar(
                x=bar_data["Category"],
                y=bar_data["Count"],
                marker=dict(color=["#2D6A2D", "#8B2020"],
                            line=dict(color=["#3D8A3D", "#B03030"], width=1)),
                text=bar_data["Count"].apply(lambda x: f"{x:,}"),
                textposition="outside",
                textfont=dict(color=CREAM, family="Inter", size=12),
                hovertemplate="<b>%{x}</b><br>%{y:,} records<extra></extra>",
            ))
            fig_bar.update_layout(
                **plotly_layout("Record Count by Class", height=320),
                bargap=0.35,
                yaxis=dict(showgrid=True, gridcolor=GRID_COLOR, linecolor=GRID_COLOR,
                           tickformat=",", color=TEXT_COLOR),
                xaxis=dict(linecolor=GRID_COLOR, color=CREAM,
                           tickfont=dict(size=13, family="Inter", color=CREAM)),
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    else:
        st.markdown("""
        <div style="background:#0A0804;border:1px dashed #2A1E10;border-radius:10px;
                    padding:3rem 2rem;text-align:center;margin-top:1rem;">
          <div style="font-size:1rem;font-weight:600;color:#4A3E2A;margin-bottom:0.5rem;">No Dataset Loaded</div>
          <div style="font-size:0.82rem;color:#3A3020;">
            Navigate to the Dataset page to upload your CSV file.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Performance metrics row (if trained)
    if st.session_state.trained:
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("<h2>Performance Metrics</h2>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        metrics = [
            ("Accuracy", st.session_state.accuracy, "#C8A86A"),
            ("Precision", st.session_state.precision, "#7EC87E"),
            ("Recall", st.session_state.recall, "#6AACE8"),
            ("K Neighbors", st.session_state.k_value, "#B8A088"),
        ]
        for col, (label, val, color) in zip([m1, m2, m3, m4], metrics):
            with col:
                display = f"{val*100:.2f}%" if isinstance(val, float) else str(val)
                col.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value" style="color:{color};">{display}</div>
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: DATASET
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Dataset":
    st.markdown("""
    <div class="page-header">
      <div class="page-title">Dataset</div>
      <div class="page-subtitle">Upload and inspect your network traffic dataset</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload CSV or XLSX file",
        type=["csv", "xlsx"],
        label_visibility="collapsed",
    )

    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)

            df.columns = df.columns.str.strip()
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            st.session_state.df = df
            st.session_state.trained = False
            st.session_state.accuracy = None
            st.success(f"Dataset loaded — {len(df):,} records, {len(df.columns)} columns")
        except Exception as e:
            st.error(f"Failed to load file: {e}")

    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # Summary stats
        c1, c2, c3, c4 = st.columns(4)
        stats = [
            ("Rows", f"{len(df):,}"),
            ("Columns", str(len(df.columns))),
            ("Normal", f"{int((df['Label']==0).sum()):,}" if 'Label' in df.columns else "—"),
            ("DDoS", f"{int((df['Label']==1).sum()):,}" if 'Label' in df.columns else "—"),
        ]
        for col, (label, val) in zip([c1, c2, c3, c4], stats):
            with col:
                col.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value" style="font-size:1.6rem;">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("<h2>Data Preview</h2>", unsafe_allow_html=True)

        preview_df = df.head(100).copy()
        if "Label" in preview_df.columns:
            preview_df["Label"] = preview_df["Label"].map({0: "Normal", 1: "DDoS"})

        st.dataframe(preview_df, use_container_width=True, height=340)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("<h2>Statistical Summary</h2>", unsafe_allow_html=True)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "Label" in num_cols:
            num_cols.remove("Label")
        st.dataframe(df[num_cols[:20]].describe().round(4), use_container_width=True, height=280)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: TRAINING MODEL
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Training Model":
    st.markdown("""
    <div class="page-header">
      <div class="page-title">Training Model</div>
      <div class="page-subtitle">Configure and train the K-Nearest Neighbors classifier</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.markdown("""
        <div style="background:#0A0804;border:1px solid #2A1E10;border-radius:10px;
                    padding:2rem;text-align:center;">
          <div style="color:#6A5E4A;font-size:0.9rem;">No dataset loaded. Please upload a dataset on the Dataset page first.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df = st.session_state.df

        col_cfg, col_run = st.columns([1, 1], gap="large")

        with col_cfg:
            st.markdown("<h2>Configuration</h2>", unsafe_allow_html=True)

            k_val = st.slider(
                "Number of Neighbors (K)",
                min_value=1, max_value=21, step=2,
                value=st.session_state.k_value,
                help="Odd values reduce tie-breaking ambiguity."
            )
            st.session_state.k_value = k_val

            test_size = st.slider(
                "Test Set Size",
                min_value=0.1, max_value=0.4, step=0.05,
                value=st.session_state.test_size,
                format="%.0f%%",
                help="Fraction of data reserved for evaluation."
            )
            st.session_state.test_size = test_size

            metric_opt = st.selectbox(
                "Distance Metric",
                ["euclidean", "manhattan", "minkowski"],
                index=0,
            )

            weight_opt = st.selectbox(
                "Weight Function",
                ["uniform", "distance"],
                index=0,
            )

            # Feature selection
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if "Label" in num_cols:
                num_cols.remove("Label")
            st.markdown("<div style='margin-top:1rem;font-size:0.78rem;color:#6A5E4A;letter-spacing:0.04em;text-transform:uppercase;'>Feature Columns</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.82rem;color:#9A8E7A;margin-top:0.2rem;'>{len(num_cols)} numeric features available — all will be used for training.</div>", unsafe_allow_html=True)

        with col_run:
            st.markdown("<h2>Run Training</h2>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#0A0804;border:1px solid #1F150C;border-radius:10px;padding:1.4rem;margin-bottom:1.5rem;">
              <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:#5A5040;margin-bottom:0.8rem;">Summary</div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;">
                <div>
                  <div style="font-size:0.7rem;color:#4A4030;text-transform:uppercase;letter-spacing:0.06em;">Algorithm</div>
                  <div style="font-size:0.88rem;color:#E1DCC9;font-weight:500;">K-Nearest Neighbors</div>
                </div>
                <div>
                  <div style="font-size:0.7rem;color:#4A4030;text-transform:uppercase;letter-spacing:0.06em;">K Value</div>
                  <div style="font-size:0.88rem;color:#C8A86A;font-weight:600;">{k_val}</div>
                </div>
                <div>
                  <div style="font-size:0.7rem;color:#4A4030;text-transform:uppercase;letter-spacing:0.06em;">Train / Test</div>
                  <div style="font-size:0.88rem;color:#E1DCC9;font-weight:500;">{int((1-test_size)*100)}% / {int(test_size*100)}%</div>
                </div>
                <div>
                  <div style="font-size:0.7rem;color:#4A4030;text-transform:uppercase;letter-spacing:0.06em;">Features</div>
                  <div style="font-size:0.88rem;color:#E1DCC9;font-weight:500;">{len(num_cols)}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            run_btn = st.button("Run KNN Training", use_container_width=True)

            if run_btn:
                if "Label" not in df.columns:
                    st.error("Column 'Label' not found in dataset.")
                else:
                    with st.spinner("Training in progress..."):
                        X = df[num_cols].values
                        y = df["Label"].values

                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)

                        X_train, X_test, y_train, y_test = train_test_split(
                            X_scaled, y,
                            test_size=test_size,
                            random_state=42,
                            stratify=y,
                        )

                        knn = KNeighborsClassifier(
                            n_neighbors=k_val,
                            metric=metric_opt,
                            weights=weight_opt,
                            n_jobs=-1,
                        )
                        knn.fit(X_train, y_train)
                        y_pred = knn.predict(X_test)

                        acc = accuracy_score(y_test, y_pred)
                        prec = precision_score(y_test, y_pred, zero_division=0)
                        rec = recall_score(y_test, y_pred, zero_division=0)
                        cm = confusion_matrix(y_test, y_pred)

                        # Prediction results
                        df_result = df.iloc[len(X_train):].copy() if len(df) > len(X_train) else df.copy()
                        pred_series = pd.Series(y_pred, name="Prediction")
                        actual_series = pd.Series(y_test, name="Actual")

                        df_result_out = pd.DataFrame({
                            "Index": np.arange(len(y_pred)),
                            "Actual": ["Normal" if v == 0 else "DDoS" for v in y_test],
                            "Predicted": ["Normal" if v == 0 else "DDoS" for v in y_pred],
                            "Correct": y_test == y_pred,
                        })

                        st.session_state.model = knn
                        st.session_state.scaler = scaler
                        st.session_state.X_test = X_test
                        st.session_state.y_test = y_test
                        st.session_state.y_pred = y_pred
                        st.session_state.accuracy = acc
                        st.session_state.precision = prec
                        st.session_state.recall = rec
                        st.session_state.cm = cm
                        st.session_state.df_result = df_result_out
                        st.session_state.trained = True
                        st.session_state.feature_cols = num_cols

                    st.success("Training complete.")

        # Results after training
        if st.session_state.trained:
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            st.markdown("<h2>Training Results</h2>", unsafe_allow_html=True)

            r1, r2, r3 = st.columns(3)
            for col, (label, val, color) in zip(
                [r1, r2, r3],
                [
                    ("Accuracy", st.session_state.accuracy, ACCENT),
                    ("Precision", st.session_state.precision, "#7EC87E"),
                    ("Recall", st.session_state.recall, "#6AACE8"),
                ]
            ):
                with col:
                    col.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-label">{label}</div>
                      <div class="metric-value" style="color:{color};">{val*100:.2f}%</div>
                    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: DETECTION RESULT
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Detection Result":
    st.markdown("""
    <div class="page-header">
      <div class="page-title">Detection Result</div>
      <div class="page-subtitle">Analysis, visualizations, and prediction table</div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.trained:
        st.markdown("""
        <div style="background:#0A0804;border:1px solid #2A1E10;border-radius:10px;
                    padding:2.5rem;text-align:center;">
          <div style="font-size:0.9rem;color:#6A5E4A;">Model has not been trained yet. Go to Training Model to run the classifier.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        y_pred = st.session_state.y_pred
        y_test = st.session_state.y_test
        cm = st.session_state.cm
        acc = st.session_state.accuracy
        prec = st.session_state.precision
        rec = st.session_state.recall
        df_result = st.session_state.df_result

        # ── Detection status ──────────────────────────────────────────────
        n_ddos_detected = int((y_pred == 1).sum())
        pct_ddos = n_ddos_detected / len(y_pred) * 100

        if n_ddos_detected > 0:
            st.markdown(f"""
            <div class="status-attack">
              <div class="status-label">Overall Detection Status</div>
              DDoS Attack Detected &nbsp;—&nbsp; {n_ddos_detected:,} malicious flows identified ({pct_ddos:.1f}% of test set)
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-normal">
              <div class="status-label">Overall Detection Status</div>
              Normal Traffic — No DDoS attacks detected in the test set
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # ── Metrics ───────────────────────────────────────────────────────
        st.markdown("<h2>Performance Metrics</h2>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        for col, (label, val, color, sub) in zip(
            [m1, m2, m3, m4],
            [
                ("Accuracy", f"{acc*100:.2f}%", ACCENT, "Overall correctness"),
                ("Precision", f"{prec*100:.2f}%", "#7EC87E", "DDoS precision"),
                ("Recall", f"{rec*100:.2f}%", "#6AACE8", "DDoS recall"),
                ("Test Samples", f"{len(y_test):,}", "#B8A088", f"K = {st.session_state.k_value}"),
            ]
        ):
            with col:
                col.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value" style="color:{color};">{val}</div>
                  <div class="metric-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # ── Charts row ────────────────────────────────────────────────────
        st.markdown("<h2>Visualizations</h2>", unsafe_allow_html=True)
        ch1, ch2, ch3 = st.columns([1, 1, 1])

        with ch1:
            pred_counts = pd.Series(y_pred).map({0: "Normal", 1: "DDoS"}).value_counts()
            fig1 = go.Figure(go.Pie(
                labels=pred_counts.index.tolist(),
                values=pred_counts.values.tolist(),
                hole=0.55,
                marker=dict(colors=["#2D6A2D", "#8B2020"],
                            line=dict(color=PLOT_BGCOLOR, width=2)),
                textfont=dict(family="Inter", size=11, color=CREAM),
                hovertemplate="<b>%{label}</b><br>%{value:,}<br>%{percent}<extra></extra>",
            ))
            fig1.update_layout(
                **plotly_layout("Predicted Distribution", height=300),
                showlegend=True,
                legend=LEGEND_H,
            )
            st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

        with ch2:
            cats = ["Normal", "DDoS"]
            actual_c = [int((y_test == 0).sum()), int((y_test == 1).sum())]
            pred_c   = [int((y_pred == 0).sum()), int((y_pred == 1).sum())]

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                name="Actual", x=cats, y=actual_c,
                marker_color=["#2D6A2D", "#8B2020"],
                marker_line=dict(color=["#3D8A3D", "#B03030"], width=1),
                opacity=0.65,
                hovertemplate="%{x} Actual: %{y:,}<extra></extra>",
            ))
            fig2.add_trace(go.Bar(
                name="Predicted", x=cats, y=pred_c,
                marker_color=["#1E4A1E", "#5A1414"],
                marker_line=dict(color=["#2D6A2D", "#8B2020"], width=1),
                opacity=0.9,
                hovertemplate="%{x} Predicted: %{y:,}<extra></extra>",
            ))
            fig2.update_layout(
                **plotly_layout("Actual vs Predicted", height=300),
                barmode="group", bargap=0.25, bargroupgap=0.05,
                yaxis=dict(showgrid=True, gridcolor=GRID_COLOR, tickformat=",", color=TEXT_COLOR),
                xaxis=dict(color=CREAM, tickfont=dict(size=12, family="Inter", color=CREAM)),
                legend=LEGEND_H,
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        with ch3:
            z = cm[::-1]
            x_lbl = ["Predicted Normal", "Predicted DDoS"]
            y_lbl = ["Actual DDoS", "Actual Normal"]
            text_z = [[str(v) for v in row] for row in z]

            fig3 = go.Figure(go.Heatmap(
                z=z, x=x_lbl, y=y_lbl,
                text=text_z, texttemplate="%{text}",
                textfont=dict(family="Inter", size=14, color=CREAM),
                colorscale=[
                    [0.0, "#0A0804"],
                    [0.4, "#1F150C"],
                    [0.7, "#412D15"],
                    [1.0, "#7A5530"],
                ],
                showscale=False,
                hovertemplate="%{y}<br>%{x}<br>Count: %{z}<extra></extra>",
            ))
            fig3.update_layout(
                **plotly_layout("Confusion Matrix", height=300),
                xaxis=dict(side="bottom", color=CREAM, tickfont=dict(size=10, family="Inter", color=CREAM)),
                yaxis=dict(color=CREAM, tickfont=dict(size=10, family="Inter", color=CREAM)),
            )
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # ── Prediction table ──────────────────────────────────────────────
        st.markdown("<h2>Prediction Table</h2>", unsafe_allow_html=True)

        col_filter, col_info = st.columns([1, 2])
        with col_filter:
            filter_opt = st.selectbox(
                "Filter by prediction",
                ["All", "Normal", "DDoS", "Incorrect only"],
                label_visibility="collapsed",
            )

        tbl = df_result.copy()
        if filter_opt == "Normal":
            tbl = tbl[tbl["Predicted"] == "Normal"]
        elif filter_opt == "DDoS":
            tbl = tbl[tbl["Predicted"] == "DDoS"]
        elif filter_opt == "Incorrect only":
            tbl = tbl[~tbl["Correct"]]

        tbl["Correct"] = tbl["Correct"].map({True: "Yes", False: "No"})
        tbl = tbl.drop(columns=["Index"])

        with col_info:
            st.markdown(f"<div style='font-size:0.82rem;color:#6A5E4A;padding-top:0.5rem;'>Showing {len(tbl):,} of {len(df_result):,} records</div>",
                        unsafe_allow_html=True)

        st.dataframe(tbl, use_container_width=True, height=380)