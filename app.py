"""
app.py — Monte Carlo Insight Simulator (Streamlit)
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from simulation import (
    run_simulation,
    auto_convergence,
    calc_confidence_interval,
    calc_running_mean,
    sensitivity_analysis,
)
from scenarios import save_scenario, load_scenario, list_scenarios, delete_scenario

# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Monte Carlo Insight Simulator",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 전역 CSS  (밝은 라이트 테마)
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f4f6fb !important;
        color: #1a1d23 !important;
    }

    /* ── 사이드바 ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #ffffff 0%, #eef1f8 100%) !important;
        border-right: 1px solid #d8dce8;
    }
    [data-testid="stSidebar"] * { color: #1a1d23 !important; }
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stNumberInput input {
        background: #ffffff;
        border: 1px solid #c5cad8;
        border-radius: 8px;
        color: #1a1d23 !important;
    }
    [data-testid="stSidebar"] label { color: #3d4257 !important; font-weight: 600; }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background: #ffffff;
        border-radius: 8px;
    }

    /* ── 메인 배경 ── */
    .main .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }

    /* ── 요약 카드 ── */
    .card {
        background: #ffffff;
        border: 1px solid #dde1ed;
        border-radius: 16px;
        padding: 22px 20px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(60,70,120,.07);
        transition: transform .18s, box-shadow .18s;
    }
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 28px rgba(60,70,120,.13);
    }
    .card-label {
        font-size: 0.72rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: .09em;
        font-weight: 600;
    }
    .card-value { font-size: 2rem; font-weight: 800; margin: 8px 0 4px; }
    .card-sub   { font-size: 0.8rem; color: #6b7280; }

    .val-blue   { color: #2563eb; }
    .val-purple { color: #7c3aed; }
    .val-green  { color: #059669; }
    .val-orange { color: #d97706; }

    /* ── 탭 ── */
    [data-testid="stTab"] {
        font-weight: 600;
        color: #374151;
    }
    [data-testid="stTab"][aria-selected="true"] {
        color: #2563eb !important;
        border-bottom: 2px solid #2563eb;
    }

    /* ── 버튼 ── */
    div.stButton > button {
        background: linear-gradient(90deg, #2563eb, #3b82f6);
        color: #ffffff !important;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.55rem 1.4rem;
        font-size: 0.95rem;
        transition: opacity .15s, transform .12s;
        box-shadow: 0 2px 10px rgba(37,99,235,.3);
    }
    div.stButton > button:hover { opacity: .88; transform: translateY(-1px); }

    /* ── 구분선 ── */
    hr { border-color: #d8dce8 !important; }

    /* ── 익스팬더 ── */
    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #dde1ed;
        border-radius: 12px;
        margin-bottom: 8px;
    }

    /* ── 경고/정보 박스 ── */
    .stAlert { border-radius: 10px; }

    /* ── 페이지 제목 영역 ── */
    .page-header {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        border-radius: 18px;
        padding: 28px 36px;
        margin-bottom: 28px;
        color: #ffffff;
        box-shadow: 0 4px 24px rgba(37,99,235,.25);
    }
    .page-header h1 { color: #ffffff; margin: 0; font-size: 1.8rem; font-weight: 800; }
    .page-header p  { color: rgba(255,255,255,.85); margin: 6px 0 0; font-size: .95rem; }

    /* ── 차트 래퍼 ── */
    .chart-card {
        background: #ffffff;
        border-radius: 16px;
        border: 1px solid #dde1ed;
        padding: 8px 8px 0;
        box-shadow: 0 2px 12px rgba(60,70,120,.06);
    }

    /* ── 랜딩 ── */
    .landing-wrap {
        text-align: center;
        padding: 80px 20px;
    }
    .landing-wrap h1 { font-size: 2.2rem; font-weight: 800; color: #1a1d23; margin: 16px 0 10px; }
    .landing-wrap p  { font-size: 1.05rem; color: #6b7280; max-width: 500px; margin: 0 auto; }

    /* ── 결과 타이틀 ── */
    .result-title { font-size: 1.5rem; font-weight: 800; color: #1a1d23; margin: 0 0 4px; }
    .result-sub   { font-size: .88rem; color: #6b7280; margin: 0 0 20px; }
    .result-sub strong { color: #2563eb; }

    /* ── 슬라이더 트랙 ── */
    [data-testid="stSlider"] .st-ck { background: #2563eb; }

    /* ── 토글 ── */
    [data-testid="stToggle"] { accent-color: #2563eb; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# 세션 상태 초기화
# ─────────────────────────────────────────────
if "variables" not in st.session_state:
    st.session_state.variables = [
        {"name": "변수 A", "min": 0.0, "max": 100.0, "dist": "정규", "weight": 1.0},
    ]
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "running_means" not in st.session_state:
    st.session_state.running_means = None

DIST_OPTIONS = ["균등", "정규", "삼각"]

# ─────────────────────────────────────────────
# 사이드바 — 변수 설정
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 0 4px;">
            <span style="font-size:1.6rem;">🎲</span>
            <span style="font-size:1.25rem; font-weight:800; color:#1a1d23; margin-left:6px;">
                Monte Carlo
            </span><br>
            <span style="font-size:.85rem; color:#6b7280; margin-left:2px;">
                Insight Simulator
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ── 변수 목록 ──────────────────────────────
    st.markdown(
        "<p style='font-size:.9rem; font-weight:700; color:#374151; margin:0 0 10px;'>📊 변수 설정</p>",
        unsafe_allow_html=True,
    )

    delete_idx = None
    for i, var in enumerate(st.session_state.variables):
        with st.expander(f"**{var['name']}**", expanded=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                var["name"] = st.text_input("변수명", value=var["name"], key=f"name_{i}")
            with col2:
                st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
                if st.button("🗑️", key=f"del_{i}", help="삭제"):
                    delete_idx = i

            c1, c2 = st.columns(2)
            var["min"] = c1.number_input("최솟값", value=float(var["min"]), key=f"min_{i}", step=1.0)
            var["max"] = c2.number_input("최댓값", value=float(var["max"]), key=f"max_{i}", step=1.0)

            var["dist"] = st.selectbox(
                "분포 유형", DIST_OPTIONS,
                index=DIST_OPTIONS.index(var["dist"]),
                key=f"dist_{i}",
            )
            var["weight"] = st.slider(
                "가중치", 0.1, 5.0, float(var["weight"]), 0.1, key=f"w_{i}"
            )

    if delete_idx is not None:
        st.session_state.variables.pop(delete_idx)
        st.rerun()

    if st.button("➕ 변수 추가", use_container_width=True):
        n = len(st.session_state.variables) + 1
        st.session_state.variables.append(
            {"name": f"변수 {chr(64 + n)}", "min": 0.0, "max": 100.0, "dist": "균등", "weight": 1.0}
        )
        st.rerun()

    st.divider()

    # ── 시뮬레이션 설정 ─────────────────────────
    st.markdown(
        "<p style='font-size:.9rem; font-weight:700; color:#374151; margin:0 0 10px;'>⚙️ 시뮬레이션 설정</p>",
        unsafe_allow_html=True,
    )
    use_auto = st.toggle("🔁 자동 수렴 감지", value=False)
    if use_auto:
        tol = st.select_slider(
            "수렴 허용 오차", options=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
            value=1e-3, format_func=lambda x: f"{x:.0e}"
        )
        max_iter = st.number_input("최대 반복 횟수", 10_000, 500_000, 100_000, 10_000)
    else:
        n_iter = st.selectbox(
            "시뮬레이션 횟수",
            [1_000, 5_000, 10_000, 50_000, 100_000],
            index=2,
            format_func=lambda x: f"{x:,}회",
        )

    st.divider()

    # ── 시나리오 관리 ────────────────────────────
    st.markdown(
        "<p style='font-size:.9rem; font-weight:700; color:#374151; margin:0 0 10px;'>💾 시나리오 관리</p>",
        unsafe_allow_html=True,
    )
    scenario_name = st.text_input("시나리오 이름", value="낙관 시나리오")

    col_s, col_d = st.columns(2)
    if col_s.button("💾 저장", use_container_width=True):
        settings = {"use_auto": use_auto}
        if use_auto:
            settings["tol"] = tol
            settings["max_iter"] = max_iter
        else:
            settings["n_iter"] = n_iter
        save_scenario(scenario_name, st.session_state.variables, settings)
        st.success(f"'{scenario_name}' 저장 완료!")

    saved = list_scenarios()
    if saved:
        selected_sc = st.selectbox("저장된 시나리오", ["— 선택 —"] + saved)
        if selected_sc != "— 선택 —":
            btn_load, btn_del = st.columns(2)
            if btn_load.button("📂 불러오기", key="load_sc", use_container_width=True):
                sc = load_scenario(selected_sc)
                st.session_state.variables = sc["variables"]
                st.rerun()
            if btn_del.button("🗑️ 삭제", key="del_sc", use_container_width=True):
                delete_scenario(selected_sc)
                st.rerun()

    st.divider()

    # ── 실행 버튼 ────────────────────────────────
    run_btn = st.button("🚀 시뮬레이션 실행", use_container_width=True)


# ─────────────────────────────────────────────
# 시뮬레이션 실행
# ─────────────────────────────────────────────
if run_btn:
    if len(st.session_state.variables) == 0:
        st.error("변수를 최소 1개 이상 추가해주세요.")
        st.stop()

    valid = True
    for v in st.session_state.variables:
        if v["min"] >= v["max"]:
            st.error(f"'{v['name']}': 최솟값이 최댓값보다 크거나 같습니다.")
            valid = False
    if not valid:
        st.stop()

    with st.spinner("시뮬레이션 진행 중 ..."):
        if use_auto:
            df, rm = auto_convergence(
                st.session_state.variables, tol=tol, max_iter=max_iter
            )
            st.session_state.running_means = rm
        else:
            df = run_simulation(st.session_state.variables, n_iter)
            idxs, rm = calc_running_mean(df["result"].values)
            st.session_state.running_means = list(rm)
        st.session_state.results_df = df


# ─────────────────────────────────────────────
# Plotly 공통 레이아웃 (라이트)
# ─────────────────────────────────────────────
CHART_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8f9fd",
    font=dict(family="Inter, sans-serif", color="#1a1d23", size=13),
    title_font=dict(size=16, color="#1a1d23", family="Inter, sans-serif"),
    xaxis=dict(
        gridcolor="#e5e7f0",
        linecolor="#c5cad8",
        zerolinecolor="#c5cad8",
        tickfont=dict(color="#374151"),
        title_font=dict(color="#374151"),
    ),
    yaxis=dict(
        gridcolor="#e5e7f0",
        linecolor="#c5cad8",
        tickfont=dict(color="#374151"),
        title_font=dict(color="#374151"),
    ),
    height=460,
    margin=dict(t=55, l=60, r=40, b=55),
)


# ─────────────────────────────────────────────
# 결과 표시
# ─────────────────────────────────────────────
if st.session_state.results_df is None:
    st.markdown(
        """
        <div class="landing-wrap">
            <div style="font-size:3.8rem; line-height:1;">🎲</div>
            <h1>Monte Carlo Insight Simulator</h1>
            <p>
                사이드바에서 변수를 설정하고<br>
                <strong style="color:#2563eb;">🚀 시뮬레이션 실행</strong> 버튼을 눌러<br>
                확률 기반 의사결정 분석을 시작하세요.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

df = st.session_state.results_df
results = df["result"].values
ci = calc_confidence_interval(results)
total_runs = len(results)

# ── 페이지 헤더 ───────────────────────────────
st.markdown(
    f"""
    <div class="page-header">
        <h1>📈 시뮬레이션 결과</h1>
        <p>
            총 <strong>{total_runs:,}회</strong> 반복 &nbsp;·&nbsp;
            변수 <strong>{len(st.session_state.variables)}개</strong> &nbsp;·&nbsp;
            90% 신뢰 구간: <strong>[{ci['p5']:,.2f} → {ci['p95']:,.2f}]</strong>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 요약 카드 ─────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
cards = [
    (c1, "🔵 P5 — 하위 5%",  ci["p5"],   "90% 범위 최저치", "val-blue"),
    (c2, "🟣 P95 — 상위 5%", ci["p95"],  "90% 범위 최고치", "val-purple"),
    (c3, "🟢 평균",           ci["mean"], f"중앙값 {ci['median']:,.2f}", "val-green"),
    (c4, "🟠 표준편차",       ci["std"],  "결과 분산 정도",  "val-orange"),
]
for col, label, val, sub, cls in cards:
    col.markdown(
        f"""
        <div class="card">
            <div class="card-label">{label}</div>
            <div class="card-value {cls}">{val:,.2f}</div>
            <div class="card-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

# ── 탭 ───────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 확률 분포도", "📉 수렴 그래프", "🌪️ 민감도 분석"])

# ────────────────────────────
# 탭 1: 히스토그램
# ────────────────────────────
with tab1:
    p5, p95 = ci["p5"], ci["p95"]

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=results, nbinsx=80,
        name="시뮬레이션 결과",
        marker_color="rgba(37,99,235,0.45)",
        marker_line=dict(color="rgba(37,99,235,0.8)", width=0.5),
    ))
    # 90% CI 음영
    fig_hist.add_vrect(
        x0=p5, x1=p95,
        fillcolor="rgba(124,58,237,0.08)",
        layer="below", line_width=0,
        annotation_text="90% CI", annotation_position="top left",
        annotation_font_color="#7c3aed",
    )
    # 수직선
    for val, label, color in [
        (p5,        f"P5: {p5:,.2f}",         "#2563eb"),
        (p95,       f"P95: {p95:,.2f}",        "#7c3aed"),
        (ci["mean"],f"평균: {ci['mean']:,.2f}","#059669"),
    ]:
        fig_hist.add_vline(
            x=val, line_dash="dash", line_color=color, line_width=2,
            annotation_text=label, annotation_font_color=color,
            annotation_position="top right",
        )

    fig_hist.update_layout(
        title="결과값 확률 분포도",
        xaxis_title="결과값",
        yaxis_title="빈도 (횟수)",
        showlegend=False,
        **CHART_LAYOUT,
    )
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ────────────────────────────
# 탭 2: 수렴 그래프
# ────────────────────────────
with tab2:
    rm = st.session_state.running_means

    if use_auto:
        x_vals = [(i + 1) * 1_000 for i in range(len(rm))]
    else:
        idxs, _ = calc_running_mean(results)
        x_vals = list(idxs)

    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(
        x=x_vals, y=rm,
        mode="lines",
        line=dict(color="#2563eb", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.06)",
        name="Running Mean",
    ))
    fig_conv.add_hline(
        y=ci["mean"], line_dash="dot", line_color="#059669", line_width=2,
        annotation_text=f"최종 평균: {ci['mean']:,.4f}",
        annotation_font_color="#059669",
        annotation_position="right",
    )
    fig_conv.update_layout(
        title="평균값 수렴 그래프",
        xaxis_title="누적 시뮬레이션 횟수",
        yaxis_title="누적 평균",
        **{**CHART_LAYOUT, "margin": dict(t=55, l=60, r=110, b=55)},
    )
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_conv, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption(
        f"수렴 상태: 최종 평균 **{ci['mean']:,.4f}** | 총 **{total_runs:,}회** 반복 수행"
    )

# ────────────────────────────
# 탭 3: 토네이도 차트
# ────────────────────────────
with tab3:
    if len(st.session_state.variables) < 2:
        st.info("민감도 분석에는 변수가 2개 이상 필요합니다.")
    else:
        corr = sensitivity_analysis(df)
        colors = ["#2563eb" if v >= 0 else "#dc2626" for v in corr.values]

        fig_tornado = go.Figure(go.Bar(
            x=corr.values,
            y=corr.index.tolist(),
            orientation="h",
            marker_color=colors,
            marker_line_width=0,
            text=[f"{v:+.3f}" for v in corr.values],
            textposition="outside",
            textfont=dict(color="#1a1d23", size=12, family="Inter, sans-serif"),
        ))
        fig_tornado.add_vline(x=0, line_color="#9ca3af", line_width=1.5)
        fig_tornado.update_layout(
            title="변수별 영향도 — 토네이도 차트",
            xaxis_title="결과값과의 상관계수 (Pearson r)",
            yaxis_title="",
            xaxis=dict(range=[-1.1, 1.1], gridcolor="#e5e7f0",
                       linecolor="#c5cad8", tickfont=dict(color="#374151"),
                       title_font=dict(color="#374151")),
            yaxis=dict(categoryorder="array",
                       categoryarray=corr.index[::-1].tolist(),
                       gridcolor="#e5e7f0", linecolor="#c5cad8",
                       tickfont=dict(color="#374151", size=13),
                       title_font=dict(color="#374151")),
            **{**CHART_LAYOUT,
               "height": max(300, 80 * len(corr) + 120),
               "margin": dict(t=55, l=130, r=90, b=55)},
        )
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_tornado, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div style="background:#eff6ff; border-left:4px solid #2563eb;
                        border-radius:8px; padding:14px 18px; margin-top:14px;">
                <strong style="color:#1d4ed8;">📌 해석 가이드</strong><br>
                <span style="color:#374151; font-size:.9rem;">
                    상관계수 절댓값이 클수록 해당 변수가 결과에 더 큰 영향을 미칩니다.<br>
                    🔵 <strong>양수(+)</strong>: 값이 커지면 결과도 커짐 &nbsp;&nbsp;
                    🔴 <strong>음수(−)</strong>: 값이 커지면 결과가 작아짐
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── 데이터 내보내기 ────────────────────────
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### 📥 데이터 내보내기")
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇️  CSV 다운로드",
        data=csv,
        file_name="monte_carlo_results.csv",
        mime="text/csv",
    )
