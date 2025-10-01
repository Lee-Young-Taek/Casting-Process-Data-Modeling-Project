from shiny import ui, render, reactive
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import warnings


# 폰트 관련 경고 메시지 숨기기
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='findfont')

# Shiny 버전에 따라 ui.nav 또는 ui.nav_panel을 사용
NAV = getattr(ui, "nav", ui.nav_panel)

# --- 경로 설정 ---
BASE_DIR = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parents[1]
PDF_FILE  = BASE_DIR / "reports" / "preprocessing_report.pdf"
DATA_FILE = BASE_DIR / "data" / "raw" / "train.csv"

# ===== Matplotlib 한글 폰트 (실제 존재하는 폰트만 사용) =====
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
# --- 상세 컨텐츠(텍스트) ---
DETAILS = {
    "drop_heating_furnace": (
        "- 결측이 매우 많고 기본 모델에서 변수 중요도 낮음 → 학습에서 제거"
    ),
    "drop_molten_volume": (
        "- 결측이 매우 많고 기본 모델에서 변수 중요도 낮음 → 학습에서 제거"
    ),
    "drop_mold_temp3": (
        "- 이상치 1449.0을 센서 오류 코드로 가정 → 두 칼럼 모두 제거"
    ),
    "drop_etc": (
        "- registration_time: 'time'+'date' 결합 정보(중복 의미) → 제거"
    ),
    "impute_molten_temp": (
        "- molten_temp 결측이 연속되어 나오는 경우 거의 없음 → 앞뒤 행들과 이어지도록 두 행들의 평균으로 대치\n"
        "- train 데이터에서는 이전 행과 다음 행의 molten_temp 값들의 평균으로 대치\n"
        "- test 데이터에서는 바로 직전 행의 molten_temp 값으로 대치"
    ),
    "row_emergency_stop": (
        "- emergency_stop이 결측인 경우 1번 존재, 이때 이 행의 나머지 칼럼들 대부분 결측 → 학습 데이터에서 행 제거\n"
        "- 모델 예측이 끝난 뒤, emergency_stop 값을 확인해서 결측인 경우 불량으로 나오도록 함"
    ),
    "row_count_dup": (
        "- count, mold_code, time, molten_volume 등이 겹치는 경우 다른 모든 변수들도 같은 값을 가짐\n"
        "- 정보 중복을 피하기 위해 하나만 남기고 나머지 중복 행들 삭제"
    ),
}

# --- 카드 하단 설명(HTML) ---
EVIDENCE_DESC = {
    "drop_heating_furnace": """
<ul>
  <li>결측이 아닌 구간: <code>mold_code</code> 일정, <code>date</code>/<code>count</code> 연속 → 동일 furnace 연속 생산으로 해석</li>
  <li>결측 구간(예: index 73407, 73408): <code>mold_code</code> 8917/8722로 상이,
      <code>molten_volume</code> 61.0→84.0, <code>count</code> 222/219로 불연속 → 서로 다른 furnace로 보임</li>
  <li>결론: 동일 <code>mold_code</code>이면서 <code>molten_volume</code>/<code>count</code>가 이어지면 하나의 furnace에서 연속 생산.
      반대로 <b>결측(NaN) 구간</b>은 최소 2개 이상의 상이한 집단일 가능성이 큼</li>
  <li>모델 관점: 변수 중요도도 높지 않아 최종적으로 <b>heating_furnace 열 제외</b></li>
</ul>
""",
    "drop_molten_volume": """
<ul>
    <li>mold_code별로 나눠서 count에 따라 molten_volume 그래프를 그렸을 때 
    count에 따라 molten_volume이 채워지고 다시 줄어드는 양상이 보임<br></li>
    <li>그러나 결측치가 너무 많아서 정확한 값을 예측하기 어렵고 
    기본 모델에서 변수 중요도도 높지 않아 최종적으로 <b>heating_furnace 열 제외</b></li>
</ul>
""",
    "drop_mold_temp3":      ("<h4>…생략…</h4>"),
    "drop_etc":             ("<h4>…생략…</h4>"),
    "impute_molten_temp": "<h4>설명</h4><p>설명 추가하기</p>",
    "impute_etc": "<h4>설명</h4><p>설명 추가하기</p>",
    "outlier_etc": "<h4>설명</h4><p>설명 추가하기</p>",
    "row_emergency_stop": """
<h4>emergency_stop이 결측이 아닌 행 제외 근거</h4>
<ul>
    <li>emergency_stop이 결측인 경우 1번 존재, 이때 이 행의 <br>나머지 칼럼들 대부분 결측</br> -> 따라서 이 행을 모델 학습에서 제외</li>
    <li>모델 예측이 끝난 뒤, emergency_stop 값을 확인해서 결측인 경우 불량으로 나오도록 함</li>
</ul>
""",
    "row_count_dup": """
<h4>count값이 겹치는 행 삭제 근거</h4>
<ul>
    <li>count, mold_code, time, molten_volume 등이 겹치는 경우 다른 모든 변수들도 같은 값을 가짐</li>
    <li>정보 중복을 피하기 위해 하나만 남기고 나머지 중복 행들 삭제</li>
</ul>
""",
}


# =========================
# UI
# =========================
def panel_body():
    css = ui.head_content(
    ui.tags.style("""
        .card .nav { margin-bottom: 0.75rem; }
        .left-col { padding-right: 1rem; border-right: 1px solid #eee; }
        .right-col { padding-left: 1rem; }
        .muted { color: #6c757d; }
         /* 탭 메뉴 색상 변경 */
        .nav-tabs .nav-link {
            background-color: #2A2D30;
            color: white;
            border: none;
            margin-right: 4px;
        }
        .nav-tabs .nav-link:hover {
            background-color: #686f7d;
        }
        .nav-tabs .nav-link.active {
            background-color: #6c6a6a;
            color: white;
        }
        /* 상단 툴바 */
        .topbar{
            display:flex; justify-content:space-between; align-items:center;
            gap:.75rem; margin-bottom:.75rem;
        }
        .topbar .right{ display:flex; align-items:center; gap:.5rem; }
        
        /* PDF 버튼 호버 효과 */
        .topbar button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(220, 53, 69, 0.4) !important;
        }
        .topbar button:active {
            transform: translateY(0px);
        }
        
        /* 접기 박스 */
        details.details-box { margin-top: .75rem; }
        details.details-box > summary { cursor: pointer; font-weight: 600; }
        
        /* 아코디언 스타일 */
        .accordion-section { 
            background: white; 
            border-radius: 16px; 
            margin-bottom: 20px; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); 
            overflow: hidden;
            max-width: 1400px;
            margin-left: ;
            margin-right: 1000;
        }
        .accordion-header { 
            background: #2A2D30; 
            color: white; 
            padding: 20px 28px; 
            cursor: pointer; 
            display: flex; 
            justify-content: space-between; 
            border: none; 
            width: 100%; 
            text-align: left; 
            font-size: 16px; 
            font-weight: 600; 
            border-radius: 16px 16px 0 0;
        }
        .accordion-header:hover { 
            background-color: #1f2428;
        }
        .accordion-content { 
            padding: 24px 28px; 
            background: #ffffff; 
            border-radius: 0 0 16px 16px;
        }
    """),
        ui.tags.script("""
            function toggleAccordion(id) {
                var content = document.getElementById(id);
                if (content.style.display === "none") {
                    content.style.display = "block";
                } else {
                    content.style.display = "none";
                }
            }
        """)
    )

    # 공통 레이아웃: 각 카드마다 탭 + (왼쪽 텍스트 / 오른쪽 시각화)
    def two_col(left, right):
        return ui.row(
            ui.column(6, left, class_="left-col"),
            ui.column(6, right, class_="right-col"),
        )

    return ui.TagList(
        "데이터 전처리 요약",
        css,

        # 상단 검색창 스타일 제목 + PDF 버튼
        ui.div(
        ui.div(
            ui.HTML('<i class="fa-solid fa-magnifying-glass" style="color: #9ca3af; margin-right: 14px; font-size: 16px;"></i>'),
            ui.span("데이터 전처리 요약", style="color: #6b7280; font-size: 15px; font-weight: 400;"),
            style="""
                background: white;
                border: 1px solid #d1d5db;
                border-radius: 24px;
                padding: 14px 20px;
                display: flex;
                align-items: center;
                flex: 1;
                max-width: 400px;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
                margin-left: 20px;
                margin-top: 20px;
            """
        ),
        ui.div(
            ui.download_button(
                "download_pdf", 
                ui.HTML('<i class="fa-solid fa-file-pdf"></i> PDF 다운로드'),
                style="""
                    background: linear-gradient(135deg, #ec685f 0%, #eb6056 100%);
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    font-size: 15px;
                    font-weight: 600;
                    border-radius: 20px;    
                    box-shadow: 0 4px 12px rgba(220, 53, 69, 0.2);
                    transition: all 0.3s ease;
                    cursor: pointer;
                    margin-right: 40px;
                    display: inline-flex;
                    align-items: center;
                    height: 41px;
                """
            ),
            style="margin-top: 20px;"
        ),
        class_="topbar",
        style="margin-bottom: 1rem; padding: 8px 0; display: flex; justify-content: space-between; align-items: center;"
    ),

        # 전체 컨텐츠를 감싸는 컨테이너 추가
        ui.div(
            # ───────────────────────────────────────────────
            # 1) 단일 칼럼 제거
            # ───────────────────────────────────────────────
            ui.div(
                ui.tags.button(
                    ui.div(
                        ui.span("단일 칼럼 제거", style="font-size: 16px;"),
                        ui.span("▼", style="font-size: 12px;"),
                        style="display: flex; justify-content: space-between; width: 100%;"
                    ),
                    onclick="toggleAccordion('drop_columns_content')",
                    class_="accordion-header"
                ),
                ui.div(
                    ui.navset_tab(
                        NAV(
                            "heating_furnace 열",
                            two_col(
                                ui.div(
                                     ui.img(src="heating_furnace.png", style="width: 100%; height: auto;"),
                                    ui.div(
                                        ui.HTML('<i class="fa-solid fa-circle-xmark" style="color: #dc3545; margin-right: 8px;"></i>'),
                                        ui.span("제거 이유: ", style="font-weight: 700; color: #dc3545; font-size: 15px;"),
                                        ui.span(DETAILS["drop_heating_furnace"], style="color: #495057; font-size: 14px;"),
                                        style="background: #fff5f5; border-left: 4px solid #dc3545; padding: 12px 16px; border-radius: 6px; margin-top: 12px; display: flex; align-items: flex-start;"
                                    ),
                                    style="display:flex; flex-direction:column; gap:.5rem;"
                                ),
                                ui.div(
                                    ui.h5("특정 인덱스 구간 확인하기 (73406–73413)"),
                                    ui.output_ui("hf_slice_df"),
                                    ui.HTML(EVIDENCE_DESC["drop_heating_furnace"]),
                                    style="margin:.25rem 0 0 0;"
                                ),
                            ),
                        ),
                        NAV(
                            "molten_volume 열",
                            two_col(
                                ui.div(
                                    ui.markdown(DETAILS["drop_molten_volume"]),
                                    ui.output_plot("mv_pie", width="100%", height="360px"),
                                    style="display:flex; flex-direction:column; gap:.5rem;"
                                ),
                                ui.div(
                                    ui.h5("mold_code별 count - molten_volume 산점도"),
                                    ui.output_plot("mv_scatter_plot", width="100%", height="600px"),
                                    ui.HTML(EVIDENCE_DESC["drop_molten_volume"]),
                                    style="margin:.25rem 0 0 0;"
                                ),
                            ),
                        ),
                        NAV(
                            "upper/lower_mold_temp3 · registration_time열",
                            two_col(
                                ui.div(
                                    ui.markdown(DETAILS["drop_mold_temp3"]),
                                    ui.output_plot("mt3_hist", width="100%", height="420px"),
                                    style="display:flex; flex-direction:column; gap:.5rem;"
                                ),
                                ui.div(
                                    ui.markdown(DETAILS["drop_etc"]),
                                    ui.output_data_frame("reg_head_df"),
                                    style="display:flex; flex-direction:column; gap:.5rem;"
                                ),
                            ),
                        ),
                    ),
                    id="drop_columns_content",
                    class_="accordion-content",
                    style="display: none;"
                ),
                class_="accordion-section"
            ),
        
            # ───────────────────────────────────────────────
            # 2) 행 제거
            # ───────────────────────────────────────────────
            ui.div(
                ui.tags.button(
                    ui.div(
                        ui.span("행 제거", style="font-size: 16px;"),
                        ui.span("▼", style="font-size: 12px;"),
                        style="display: flex; justify-content: space-between; width: 100%;"
                    ),
                    onclick="toggleAccordion('row_removal_content')",
                    class_="accordion-header"
                ),
                ui.div(
                    ui.navset_tab(
                        NAV(
                            "emergency_stop",
                            ui.div(
                                ui.markdown(DETAILS["row_emergency_stop"]),
                                ui.output_data_frame("emergency_stop_df"),
                                style="display:flex; flex-direction:column; gap:.5rem;"
                            ),
                        ),
                        NAV(
                            "count 중복",
                            ui.div(
                                ui.markdown(DETAILS["row_count_dup"]),
                                ui.output_ui("count_dup_ui"),
                                style="display:flex; flex-direction:column; gap:.5rem;"
                            ),
                        ),
                    ),
                    id="row_removal_content",
                    class_="accordion-content",
                    style="display: none;"
                ),
                class_="accordion-section"
            ),

            # ───────────────────────────────────────────────
            # 3) 결측치 처리
            # ───────────────────────────────────────────────
            ui.div(
                ui.tags.button(
                    ui.div(
                        ui.span("결측치 처리", style="font-size: 16px;"),
                        ui.span("▼", style="font-size: 12px;"),
                        style="display: flex; justify-content: space-between; width: 100%;"
                    ),
                    onclick="toggleAccordion('missing_value_content')",
                    class_="accordion-header"
                ),
                ui.div(
                    ui.navset_tab(
                        NAV(
                            "molten_temp 열",
                            ui.div(
                                ui.markdown(DETAILS["impute_molten_temp"]),
                                ui.row(
                                    ui.column(
                                        6,
                                        ui.output_plot("mt_na_runs_plot", width="100%", height="380px"),
                                        class_="pe-2"
                                    ),
                                    ui.column(
                                        6,
                                        ui.output_plot("mt_na_sample_plot", width="100%", height="580px"),
                                        class_="ps-2"
                                    )
                                ),
                                style="display:flex; flex-direction:column; gap:.75rem;"
                            ),
                        ),
                    ),
                    id="missing_value_content",
                    class_="accordion-content",
                    style="display: block;"
                ),
                class_="accordion-section"
            ),
            
            # 컨테이너 닫기 + 너비 제한 스타일
            style="max-width: 1400px; margin: 0 auto; padding: 0 24px;"
        ),
    )



        ## PDF
        #ui.card(
        #    ui.card_header("전처리 PDF"),
        #    ui.p("아래 버튼을 눌러 전처리 상세 문서를 PDF로 다운로드하세요."),
        #    ui.download_button("download_pdf", "PDF 다운로드"),
        #    ui.div(ui.output_text("pdf_status"), class_="text-muted mt-2"),
        #),


# =========================
# SERVER
# =========================

def panel():
    return ui.nav_panel("데이터 전처리 요약", panel_body())

def server(input, output, session):

    # ── DataFrames ────────────────────────────────────────────────────────────
    @reactive.calc
    def _raw_df() -> pd.DataFrame:
        # 문자열 보존용(등록/시간 열 미리보기 등에 사용)
        return pd.read_csv(DATA_FILE, dtype=str, low_memory=False)

    @reactive.calc
    def _num_df() -> pd.DataFrame:
        # 수치 시각화용
        df = pd.read_csv(DATA_FILE, low_memory=False)
        return df
    
    @reactive.calc
    def _mv_df() -> pd.DataFrame:
        df = pd.read_csv(DATA_FILE, low_memory=False)
        need = {"mold_code", "molten_volume", "count", "passorfail"}
        if not need.issubset(df.columns):
            # 필요한 칼럼이 없으면 빈 DF 반환
            return pd.DataFrame(columns=list(need))
        # 숫자 변환
        df["molten_volume"] = pd.to_numeric(df["molten_volume"], errors="coerce")
        df["count"] = pd.to_numeric(df["count"], errors="coerce")
        # 품질 라벨은 문자열(0.0/1.0)로 고정
        df["passorfail"] = df["passorfail"].astype(str)
        # 필수 값 결측 제거
        return df.dropna(subset=["molten_volume", "count", "mold_code"])


    
    # --- heating_furnace: 특정 인덱스 구간(73406–73413) + 그룹 색상 표시 ---
    @output
    @render.ui
    def hf_slice_df():
        df = _raw_df()
        required = ["heating_furnace", "mold_code", "time", "date", "molten_volume", "count"]
        if not set(required).issubset(df.columns) or df.empty:
            return ui.HTML("<div class='text-muted'>표시할 데이터가 없습니다.</div>")

        # 73406~73413 범위(없으면 앞 8행)
        start, end = 73406, 73413
        if df.index.min() <= start and df.index.max() >= end:
            subset = df.loc[start:end, required].copy()
        else:
            subset = df.loc[:, required].head(8).copy()

        # ---- 유틸: 안전 정규화/표시/이스케이프 ----
        def norm(x):
            # 비교용(색결정): 결측 -> "", 그 외 strip한 문자열
            if pd.isna(x):
                return ""
            s = str(x).strip()
            return "" if s in ("<NA>", "None") else s

        def show_text(x):
            # 화면표시용: 결측/빈문자/특수표현 -> "Nan"
            if pd.isna(x):
                return "Nan"
            s = str(x).strip()
            return "Nan" if s in ("", "<NA>", "None", "nan", "NaN") else s

        def esc(s):
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # 색 규칙
        def row_color(hf, mold):
            h = norm(hf)
            m = norm(mold)
            if h.upper() == "B":
                return "#e6f4ea"   # 연한 녹색
            if (h.lower() in ("nan", "")) and m == "8917":
                return "#fff7cc"   # 연한 노랑
            if (h.lower() in ("nan", "")) and m == "8722":
                return "#e7f3ff"   # 연한 하늘
            return "#ffffff"

        cols = ["index","heating_furnace","mold_code","time","date","molten_volume","count"]
        thead = "<thead><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr></thead>"

        body_rows = []
        for idx, row in subset.iterrows():
            bg = row_color(row.get("heating_furnace"), row.get("mold_code"))
            cells = []
            for c in cols:
                val = idx if c == "index" else row.get(c)
                txt = esc(show_text(val))
                cells.append(f"<td style='background-color:{bg} !important;'>{txt}</td>")
            body_rows.append(f"<tr>{''.join(cells)}</tr>")

        table_html = f"""
        <div style="max-height:420px; overflow:auto; border:1px solid #eee; border-radius:6px;">
          <table class="table table-sm" style="width:100%; border-collapse:collapse;">
            {thead}
            <tbody>{''.join(body_rows)}</tbody>
          </table>
        </div>
        <div class="text-muted" style="font-size:.9rem; margin-top:.35rem;">
          <span style="background:#e6f4ea; padding:0 .35rem; border-radius:.25rem;">B 그룹</span>
          <span style="background:#fff7cc; padding:0 .35rem; border-radius:.25rem; margin-left:.35rem;">NaN &amp; mold=8917</span>
          <span style="background:#e7f3ff; padding:0 .35rem; border-radius:.25rem; margin-left:.35rem;">NaN &amp; mold=8722</span>
        </div>
        """
        return ui.HTML(table_html)





    # ── (단일 칼럼) molten_volume: 결측 vs 비결측 도넛 차트 ─────────────────────
    @output
    @render.plot
    def mv_pie():
        df = _raw_df()
        if "molten_volume" not in df.columns or df.empty:
            fig = plt.figure(figsize=(3, 2))
            plt.text(0.5, 0.5, "molten_volume 칼럼이 없거나 데이터가 비어있습니다.",
                     ha="center", va="center")
            plt.axis("off")
            return fig
    
        # 문자열로 통일 후 공백 제거
        s = df["molten_volume"].astype("string").str.strip()
    
        # 결측 판단: NaN + 빈문자 + 'nan'/'none' (대소문자 무시)
        miss_mask = s.isna() | s.eq("") | s.str.lower().isin(["nan", "none"])
        n_miss = int(miss_mask.sum())
        n_ok   = int((~miss_mask).sum())
    
        labels = ["결측", "비결측"]
        sizes  = [n_miss, n_ok]
        colors = ["#e15759", "#4c78a8"]
    
        fig, ax = plt.subplots(figsize=(9, 4.2))
        # 반환값 언팩하지 않으면 버전 차이에도 안전
        ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.38, edgecolor="white"),  # 도넛 형태
            pctdistance=0.80,
            textprops={"fontsize": 12},
            autopct=lambda p: f"{p:.1f}%\n({int(round(p/100*sum(sizes))):,})",
        )
        ax.set_title("molten_volume 결측/비결측 비율", pad=8)
        ax.axis("equal")
        return fig
    
    # ── (단일 칼럼) molten_volume 상세 설명: mold_code별 count & molten_volume 산점도 ─────────────────────
    @output
    @render.plot
    def mv_scatter_plot():
        df = _mv_df()
        if df.empty:
            fig = plt.figure(figsize=(4, 2))
            plt.text(0.5, 0.5, "필수 칼럼이 없거나 데이터가 비어 있습니다.", ha="center", va="center")
            plt.axis("off")
            return fig

        # (1) 이상치 제거: molten_volume > 2500 제외
        df = df[df["molten_volume"] <= 2500]

        # (2) 빈도 상위 4개 mold_code 선택
        top_molds = df["mold_code"].value_counts().index.tolist()[:4]
        n = len(top_molds)
        if n == 0:
            fig = plt.figure(figsize=(4, 2))
            plt.text(0.5, 0.5, "표시할 mold_code가 없습니다.", ha="center", va="center")
            plt.axis("off")
            return fig

        # (3) 큰 그림 + 세로 간격 여유
        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(18, 4.8 * n))
        fig.subplots_adjust(hspace=0.65, top=0.95, bottom=0.08)

        if n == 1:
            axes = [axes]

        # (4) 색상/라벨: 0.0=양품(파랑), 1.0=불량(빨강)
        hue_order = ["0.0", "1.0"]
        palette_map = {"0.0": "blue", "1.0": "red"}

        for i, mold in enumerate(top_molds):
            ax = axes[i]
            mold_df = df[df["mold_code"] == mold].head(300)  # mold별 최대 300개 샘플
            sns.scatterplot(
                data=mold_df,
                x="count", y="molten_volume",
                hue="passorfail", hue_order=hue_order,
                palette=palette_map, alpha=0.7, s=24, ax=ax, legend=False
            )
            ax.set_title(f"Mold Code: {mold}", pad=6, fontsize=15)
            ax.set_xlabel("Count")
            ax.set_ylabel("Molten Volume")
            ax.margins(x=0.04, y=0.12)
            ax.grid(True, alpha=0.25)

        # (5) 커스텀 범례(마지막 축에만)
        custom_legend = [
            Line2D([0], [0], marker='o', color='blue', linestyle='', label='양품 (0.0)', markersize=8),
            Line2D([0], [0], marker='o', color='red',  linestyle='', label='불량 (1.0)', markersize=8),
        ]
        axes[-1].legend(handles=custom_legend, title="품질", loc="upper right")

        return fig



    # ── (단일 칼럼) upper/lower_mold_temp3: 0~1500 히스토그램 + 1449 라인 ─────
    @output
    @render.plot
    def mt3_hist():
        df = _num_df()
        cols = ["upper_mold_temp3", "lower_mold_temp3"]
        if not set(cols).issubset(df.columns):
            fig = plt.figure(figsize=(4, 2))
            plt.text(0.5, 0.5, "upper/lower_mold_temp3 열이 없습니다.", ha="center", va="center")
            plt.axis("off")
            return fig

        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        fig, axes = plt.subplots(2, 1, figsize=(10, 6.5))
        fig.subplots_adjust(hspace=0.4, top=0.92, bottom=0.12, left=0.08, right=0.98)

        name_map = {"upper_mold_temp3": "Upper Mold Temp3", "lower_mold_temp3": "Lower Mold Temp3"}
        for i, c in enumerate(cols):
            ax = axes[i]
            s_all  = df[c].dropna()
            s_plot = s_all[(s_all >= 0) & (s_all <= 1500)]
            ax.hist(s_plot, bins=120, color="#4c78a8", alpha=0.85, edgecolor="white")
            ax.axvline(1449.0, color="red", linestyle="--", linewidth=1.8)
            cnt_1449 = int((s_all == 1449.0).sum())
            ax.set_xlim(0, 1500)
            ax.set_xlabel("Temperature")
            ax.set_ylabel("Count")
            ax.set_title(f"{name_map[c]} — n={len(s_plot):,} (x≤1500), 1449.0 개수={cnt_1449:,}")
            ymax = ax.get_ylim()[1]
            ax.text(1449.0, ymax * 0.9, f"{cnt_1449:,}", color="red",
                    ha="left", va="center", fontsize=10, rotation=90)

        return fig

    # ── (단일 칼럼) registration/time/date 5행 미리보기 ────────────────────────
    @output
    @render.data_frame
    def reg_head_df():
        df = _raw_df()
        reg_col = "registration" if "registration" in df.columns else (
            "registration_time" if "registration_time" in df.columns else None
        )
        cols = []
        if reg_col: cols.append(reg_col)
        if "time" in df.columns: cols.append("time")
        if "date" in df.columns: cols.append("date")
        if not cols:
            return pd.DataFrame()

        view = df[cols].head(5).copy()
        if reg_col and reg_col != "registration":
            view = view.rename(columns={reg_col: "registration"})
        view = view.replace({pd.NA: "Nan", None: "Nan", "": "Nan"}).fillna("Nan")
        return view
    
    # --- emergency_stop 결측(비상정지) 행 표출 ---
    @output
    @render.data_frame
    def emergency_stop_df():
        df = _raw_df()
        if "emergency_stop" not in df.columns:
            return pd.DataFrame()

        # 1) 비상정지: emergency_stop이 결측/빈문자/'nan'/'none' 인 행
        s = df["emergency_stop"].astype("string").str.strip()
        mask_emergency = s.isna() | s.eq("") | s.str.lower().isin(["nan", "none"])
        out = df[mask_emergency].copy()

        # 2) 보여줄 칼럼(요청하신 순서)
        desired_cols = [
            "id", "time", "date", "count", "working", "emergency_stop",
            "facility_operation_cycleTime", "production_cycletime",
            "low_section_speed", "high_section_speed", "cast_pressure",
            "biscuit_thickness",
            "upper_mold_temp1", "upper_mold_temp2", "upper_mold_temp3",
            "lower_mold_temp1", "lower_mold_temp2", "lower_mold_temp3",
            "sleeve_temperature", "physical_strength", "Coolant_temperature",
            "EMS_operation_time", "tryshot_siganl", "heating_furnace",
        ]

        # 3) 없는 칼럼은 생성해서 결측으로 채운 뒤, 순서 정렬
        for c in desired_cols:
            if c not in out.columns:
                out[c] = pd.NA
        out = out[desired_cols]

        # 4) 표시용으로 결측을 "Nan" 문자열로 통일
        #    (숫자/문자 구분 없이 모두 문자열로 보여주고, 결측은 Nan으로 표기)
        for c in out.columns:
            out[c] = out[c].astype(str)
        out = out.replace({pd.NA: "Nan", "None": "Nan", "nan": "Nan", "NaN": "Nan", "": "Nan"}).fillna("Nan")

        return out
    
    # --- count 중복 - count==32 & mold_code==8412 예시 표출 ---
    @output
    @render.data_frame
    def count_dup_df():
        df = _raw_df()
        if ("count" not in df.columns) or ("mold_code" not in df.columns):
            return pd.DataFrame()

        # count 숫자 비교(문자 '32', '32.0' 등도 안전하게 잡도록 수치 변환)
        cnt = pd.to_numeric(df["count"], errors="coerce")
        mold = df["mold_code"].astype("string").str.strip()
        time = df["time"].astype("string").str.strip()

        out = df[(cnt == 32) & (mold == "8412") & (time == '2019-01-07')].copy()

        drop_cols = [c for c in ["line", "name", "mold_name"] if c in out.columns]
        if drop_cols:
            out = out.drop(columns=drop_cols)

        # 표시 가독성을 위해 결측값을 "Nan"으로 통일
        out = out.replace({pd.NA: "Nan", None: "Nan", "": "Nan"}).fillna("Nan")
        return out
    
    # --- count 중복 - count==32 & mold_code==8412 예시 표출 + 맨 윗 행 하이라이트 ---    
    @output
    @render.ui
    def count_dup_ui():
        df = _raw_df()
        need = {"count", "mold_code", "time"}
        if (not need.issubset(df.columns)) or df.empty:
            return ui.HTML("<div class='text-muted'>표시할 데이터가 없습니다.</div>")

        # 안전 비교(NA 무해화)
        cnt  = pd.to_numeric(df["count"], errors="coerce")
        mold = df["mold_code"].astype("string").str.strip()
        time = df["time"].astype("string").str.strip()

        mask = cnt.eq(32) & mold.eq("8412").fillna(False) & time.eq("2019-01-07").fillna(False)
        out = df[mask].copy()

        # 숨길 칼럼 제거
        drop_cols = [c for c in ["line", "name", "mold_name"] if c in out.columns]
        if drop_cols:
            out = out.drop(columns=drop_cols)

        if out.empty:
            return ui.HTML("<div class='text-muted'>조건(count=32, mold_code=8412, time=2019-01-07)에 맞는 행이 없습니다.</div>")

        # index==2953을 최상단으로 이동(있을 때만)
        if 2953 in out.index:
            out = pd.concat([out.loc[[2953]], out.drop(index=2953)], axis=0)

        # 표시용 포맷터
        def show_text(x):
            if pd.isna(x): return "Nan"
            s = str(x).strip()
            return "Nan" if s in ("", "<NA>", "None", "nan", "NaN") else s

        def esc(s):
            return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

        # HTML 테이블 만들기(2953만 하이라이트)
        cols = ["index"] + list(out.columns)
        thead = "<thead><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr></thead>"

        rows = []
        for idx, row in out.iterrows():
            bg = "#fff7cc" if idx == 2953 else "#ffffff"   # ← 연한 노란색
            tds = []
            for c in cols:
                val = idx if c == "index" else row.get(c)
                tds.append(f"<td style='background-color:{bg} !important;'>{esc(show_text(val))}</td>")
            rows.append(f"<tr>{''.join(tds)}</tr>")

        html = f"""
        <div style="max-height:420px; overflow:auto; border:1px solid #eee; border-radius:6px;">
          <table class="table table-sm" style="width:100%; border-collapse:collapse;">
            {thead}
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>
        """
        return ui.HTML(html)

    # --- (결측치 처리) molten_volume열 - 연속 결측 길이 분포 ------------------------
    # --- (결측치 처리) molten_temp열 - 연속 결측 길이 분포 (0~15 구간, 작은 막대도 보이게) ------------------------
    @output
    @render.plot
    def mt_na_runs_plot():
        df = _num_df()
        if "molten_temp" not in df.columns or df.empty:
            fig = plt.figure(figsize=(4, 2))
            plt.text(0.5, 0.5, "molten_temp 칼럼이 없거나 데이터가 없습니다.",
                     ha="center", va="center")
            plt.axis("off")
            return fig

        s = pd.to_numeric(df["molten_temp"], errors="coerce")
        mask = s.isna().to_numpy()
        idx = np.where(mask)[0]

        fig, ax = plt.subplots(figsize=(8.5, 3.2))
        if idx.size == 0:
            ax.text(0.5, 0.5, "molten_temp 결측이 없습니다.", ha="center", va="center")
            ax.axis("off")
            return fig

        # 연속 결측 구간 길이 계산
        splits = np.where(np.diff(idx) != 1)[0] + 1
        groups = np.split(idx, splits)
        lengths = [len(g) for g in groups]

        vc_all = pd.Series(lengths).value_counts().sort_index()
        # 0~15 구간만 표시
        vc = vc_all[vc_all.index <= 15]
        n_over = int(vc_all[vc_all.index > 15].sum())
        max_len = int(max(lengths)) if lengths else 0

        bars = ax.bar(vc.index, vc.values, color="#4c78a8", edgecolor="#3b5a85")

        # 작은 빈도도 보이도록 symlog(0~10은 선형, 그 이상은 로그) 적용
        ax.set_yscale("symlog", linthresh=10, linscale=1.0, base=10)

        # 값 라벨은 실제 개수로 모두 표시
        for rect, (x, v) in zip(bars, zip(vc.index, vc.values)):
            ax.text(rect.get_x() + rect.get_width()/2.0,
                    max(v, 1),               # 로그 스케일에서도 보이게 최소 위치 1 보정
                    f"{int(v)}",
                    ha="center", va="bottom", fontsize=9)

        ax.set_xlim(-0.5, 15.5)
        ax.set_xticks(np.arange(0, 16, 1))
        ax.set_xlabel("연속 결측 길이 (0–15만 표시)")
        ax.set_ylabel("개수 (symlog 스케일)")

        title = f"molten_temp 연속 결측 길이 분포 (총 결측 구간={len(lengths):,})"
        ax.set_title(title)

        ax.grid(axis="y", alpha=.25)
        # 상단 여유
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.08)
        return fig



    # --- (결측치 처리) molten_temp열 - 샘플 결측 구간 분포 (선형보간 전/후 비교) ------------------------
    @output
    @render.plot
    def mt_na_sample_plot():
        df = _num_df()
        if "molten_temp" not in df.columns or df.empty:
            fig = plt.figure(figsize=(4, 2))
            plt.text(0.5, 0.5, "molten_temp 칼럼이 없거나 데이터가 없습니다.",
                     ha="center", va="center")
            plt.axis("off")
            return fig

        s = pd.to_numeric(df["molten_temp"], errors="coerce")
        mask = s.isna().to_numpy()
        idx = np.where(mask)[0]

        if idx.size == 0:
            fig = plt.figure(figsize=(4, 2))
            plt.text(0.5, 0.5, "molten_temp 결측이 없습니다.", ha="center", va="center")
            plt.axis("off")
            return fig

        # 연속 결측 구간(start, length) 계산
        splits = np.where(np.diff(idx) != 1)[0] + 1
        groups = np.split(idx, splits)
        runs = [(g[0], len(g)) for g in groups]  # (시작인덱스, 길이)

        # 샘플 선택: 길이=1 + 2~10 사이의 짧은 구간(없으면 2 이상 중 최단)
        runs_sorted = sorted(runs, key=lambda x: (x[1], x[0]))
        typical = next((r for r in runs_sorted if r[1] == 1), runs_sorted[0])
        short = next((r for r in runs_sorted if 2 <= r[1] <= 10), None)
        if short is None:
            short = next((r for r in runs_sorted if r[1] > 1), None)

        samples = [typical] + ([short] if short and short != typical else [])
        n = len(samples)

        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10.2, 3.9 * n), sharey=False)
        if n == 1:
            axes = [axes]

        s_interp = s.interpolate("linear", limit_direction="both")

        for ax, (start, length) in zip(axes, samples):
            end = start + length - 1
            pad = 12
            a = max(0, start - pad)
            b = min(len(s) - 1, end + pad)

            xs = np.arange(a, b + 1)
            y_obs = s.iloc[a:b+1]
            y_int = s_interp.iloc[a:b+1]

            # ✅ 관측: 파란 '원형 마커'만 (선 제거)
            ax.plot(xs, y_obs, marker='o', linestyle='None', ms=3.5,
                    color='#1f77b4', label="원시값(관측)")

            # ✅ 대치 후(선형): 주황 '점선'
            ax.plot(xs, y_int, linestyle='--', lw=1.2,
                    color='#ff7f0e', label="대치 후(선형)")

            # ✅ 대치값(결측 위치): 초록 '사각형 마커'
            ax.plot(np.arange(start, end + 1), s_interp.iloc[start:end + 1],
                    marker='s', linestyle='None', ms=5,
                    color='#2ca02c', label="대치값")

            # 결측 구간 하이라이트
            ax.axvspan(start - 0.5, end + 0.5, color="#fde68a", alpha=0.35)
            ax.set_title(f"샘플 결측 구간 (index≈{start}, 길이={length})", fontsize=11)
            ax.set_xlabel("Row index (시간 순)")
            ax.set_ylabel("molten_temp")
            ax.grid(True, alpha=.25)
            ax.margins(x=0.02)

        fig.subplots_adjust(top=0.92, bottom=0.16, left=0.08, right=0.98, hspace=0.65)
        axes[0].legend(loc="best", fontsize=9)
        return fig








    # ── PDF 상태/다운로드 ──────────────────────────────────────────────────────
    @output
    @render.text
    def pdf_status():
        return f"파일 위치: {PDF_FILE}" if PDF_FILE.exists() \
            else "주의: PDF 파일이 아직 없습니다. reports/preprocessing_report.pdf 경로에 파일을 생성해 주세요."

    @render.download(filename="preprocessing_report.pdf")
    def download_pdf():
        if PDF_FILE.exists():
            with open(PDF_FILE, "rb") as f:
                chunk = f.read(8192)
                while chunk:
                    yield chunk
                    chunk = f.read(8192)
        else:
            minimal_pdf = (
                b"%PDF-1.4\n"
                b"1 0 obj<<>>endobj\n"
                b"2 0 obj<< /Type /Catalog /Pages 3 0 R >>endobj\n"
                b"3 0 obj<< /Type /Pages /Kids [4 0 R] /Count 1 >>endobj\n"
                b"4 0 obj<< /Type /Page /Parent 3 0 R /MediaBox [0 0 300 144] /Contents 5 0 R >>endobj\n"
                b"5 0 obj<< /Length 62 >>stream\n"
                b"BT /F1 12 Tf 24 100 Td (Preprocessing report missing) Tj ET\n"
                b"endstream endobj\n"
                b"6 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n"
                b"xref\n0 7\n0000000000 65535 f \n0000000010 00000 n \n0000000050 00000 n \n0000000102 00000 n \n0000000162 00000 n \n0000000258 00000 n \n0000000358 00000 n \n"
                b"trailer<< /Size 7 /Root 2 0 R >>\nstartxref\n408\n%%EOF\n"
            )
            yield minimal_pdf

