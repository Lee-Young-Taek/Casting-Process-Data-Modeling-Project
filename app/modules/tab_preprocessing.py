# modules/tab_preprocessing.py
from shiny import ui, render, reactive
from pathlib import Path
import pandas as pd

# --- 경로 설정 ---
BASE_DIR = Path(__file__).resolve().parents[2]
PDF_FILE  = BASE_DIR / "reports" / "preprocessing_report.pdf"
DATA_FILE = BASE_DIR / "data" / "raw" / "train.csv"

# --- 상세 컨텐츠(텍스트) ---
DETAILS = {
    # 단일 칼럼 제거
    "drop_heating_furnace": (
        "### 단일 칼럼 제거 — heating_furnace 열\n"
        "- A: 16,413개, B: 16,318개, 결측치: 40,880개\n"
        "- -> 결측치가 A와 B의 2배 정도\n"
    ),
    "drop_molten_volume": (
        "### 단일 칼럼 제거 — molten_volume 열\n"
        "- 결측치 **총 34,992개**. count에 따라 약한 추세는 있으나 **명확한 대치 기준 불분명**\n"
        "- 대치 시 오히려 **노이즈/누수 위험** → **학습에서는 제거**"
    ),
    "drop_mold_temp3": (
        "### 단일 칼럼 제거 — upper/lower_mold_temp3 열\n"
        "- 두 칼럼 모두 이상치가 대량 발생하며 값이 **1449.0으로 고정**되는 패턴\n"
        "- 센서 오류 코드로 가정 가능 → 신호 왜곡 방지를 위해 **두 칼럼 모두 제거**"
    ),
    "drop_etc": (
        "### 단일 칼럼 제거 — 기타\n"
        "- **registration_time**: `time` + `date` 결합 정보로 **중복 의미** → 제거\n"
        "- (추가) 강한 다중공선성/중복 정보 칼럼은 **피처 중요도**와 **상관 분석** 기반으로 정리"
    ),

    # 결측치 처리
    "impute_molten_temp": (
        "### 결측치 처리 — molten_temp 열\n"
        "- 같은 **mold_code**로 그룹화 후, **인접 count(앞·뒤)**의 `molten_temp` 평균으로 대치\n"
        "- 급격한 상태 전이를 막기 위해 **단측/양측 이웃 평균**을 상황에 맞게 선택\n"
        "- (의사코드)\n"
        "    g = df.sort_values([\"mold_code\",\"count\"]).groupby(\"mold_code\")\n"
        "    df[\"molten_temp_filled\"] = g[\"molten_temp\"].apply(\n"
        "        lambda s: s.fillna((s.shift(1) + s.shift(-1))/2)\n"
        "    )"
    ),
    "impute_etc": (
        "### 결측치 처리 — 기타\n"
        "- 미미한 결측(무작위성 가정 가능)에는 **중앙값 대치** 우선\n"
        "- 모델 파이프라인에서는 **학습/검증 분리 이후** SimpleImputer/IterativeImputer 적용으로 **누수 방지**"
    ),

    # 이상치 처리
    "outlier_etc": (
        "### 이상치 처리 — 기타\n"
        "- 분포 기반(IQR/z-score) + 공정 지식 병행\n"
        "- **윈저라이징(상·하위 백분위수 클리핑)** 또는 **로부스트 스케일링** 적용\n"
        "- **센서 오류 코드(예: 1449.0)** 패턴은 별도 라벨링 후 **칼럼 제거 or 대체**"
    ),

    # 행 제거
    "row_emergency_stop": (
        "### 행 제거 — emergency_stop\n"
        "- **index 19327**: `emergency_stop` 결측 + **다른 다수 측정값도 결측**\n"
        "- 학습 데이터에서는 **행 제거**로 이상치 영향 차단\n"
        "- 예측 단계에서는 **`emergency_stop` 결측 → 불량 판정** 규칙을 **후처리 로직**에 추가"
    ),
    "row_count_dup": (
        "### 행 제거 — count 중복\n"
        "- 동일 **(mold_code, date, count)** 조합의 **완전 중복** 레코드 존재\n"
        "- 모든 파라미터 값 동일 시 **중복 제거(첫 레코드만 유지)**로 편향 방지\n"
        "    df = df.sort_values([\"mold_code\",\"date\",\"count\"])\n"
        "    df = df.drop_duplicates(subset=[\"mold_code\",\"date\",\"count\"], keep=\"first\")"
    ),
}

# --- 표 아래에 고정으로 보여줄 설명 텍스트(코드에서 직접 수정) ---
EVIDENCE_DESC = {
    "drop_heating_furnace": (
        "#### 설명\n"
        "설명 추가하기"
    ),
}

def panel():
    return ui.nav_panel(
        "데이터 전처리 요약",
        ui.page_sidebar(
            # --------- 왼쪽 사이드바 ---------
            ui.sidebar(
                ui.h4("전처리 항목"),
                ui.p("• 단일 칼럼 제거"),
                ui.input_action_button("btn_drop_heating_furnace", "heating_furnace 열"),
                ui.input_action_button("btn_drop_molten_volume", "molten_volume 열"),
                ui.input_action_button("btn_drop_mold_temp3", "upper/lower_mold_temp3 열"),
                ui.input_action_button("btn_drop_etc", "기타"),
                ui.hr(),

                ui.p("• 결측치 처리"),
                ui.input_action_button("btn_impute_molten_temp", "molten_temp 열"),
                ui.input_action_button("btn_impute_etc", "기타"),
                ui.hr(),

                ui.p("• 이상치 처리"),
                ui.input_action_button("btn_outlier_etc", "기타"),
                ui.hr(),

                ui.p("• 행 제거"),
                ui.input_action_button("btn_row_emergency_stop", "emergency_stop"),
                ui.input_action_button("btn_row_count_dup", "count 중복"),
            ),

            # --------- 메인: 설명 + 표 + PDF ---------
            ui.card(
                ui.h3("상세 내용"),
                ui.output_ui("detail_panel"),
            ),
            ui.card(
                ui.h3("증거 데이터 미리보기"),
                ui.output_ui("evidence_panel"),
            ),
            ui.card(
                ui.h3("전처리 PDF"),
                ui.p("아래 버튼을 눌러 전처리 상세 문서를 PDF로 다운로드하세요."),
                ui.download_button("download_pdf", "PDF 다운로드"),
                ui.div(ui.output_text("pdf_status"), class_="text-muted mt-2"),
            ),
        ),
    )

def server(input, output, session):
    selected = reactive.Value(None)

    # --- 버튼 이벤트 (세미콜론 X, 한 줄에 한 데코레이터) ---
    @reactive.effect
    @reactive.event(input.btn_drop_heating_furnace)
    def _sel1():
        selected.set("drop_heating_furnace")

    @reactive.effect
    @reactive.event(input.btn_drop_molten_volume)
    def _sel2():
        selected.set("drop_molten_volume")

    @reactive.effect
    @reactive.event(input.btn_drop_mold_temp3)
    def _sel3():
        selected.set("drop_mold_temp3")

    @reactive.effect
    @reactive.event(input.btn_drop_etc)
    def _sel4():
        selected.set("drop_etc")

    @reactive.effect
    @reactive.event(input.btn_impute_molten_temp)
    def _sel5():
        selected.set("impute_molten_temp")

    @reactive.effect
    @reactive.event(input.btn_impute_etc)
    def _sel6():
        selected.set("impute_etc")

    @reactive.effect
    @reactive.event(input.btn_outlier_etc)
    def _sel7():
        selected.set("outlier_etc")

    @reactive.effect
    @reactive.event(input.btn_row_emergency_stop)
    def _sel8():
        selected.set("row_emergency_stop")

    @reactive.effect
    @reactive.event(input.btn_row_count_dup)
    def _sel9():
        selected.set("row_count_dup")

    # --- 데이터 로드: 문자열 그대로 보존 (파싱 X) ---
    @reactive.calc
    def _raw_df() -> pd.DataFrame:
        # dtype=str 로 읽어 날짜/시간/숫자 모두 "원본 문자열" 유지
        df = pd.read_csv(DATA_FILE, dtype=str, low_memory=False)
        return df

    # --- 상세 설명 ---
    @output
    @render.ui
    def detail_panel():
        key = selected.get()
        if not key:
            return ui.markdown("왼쪽 사이드바에서 항목을 선택하세요.")
        return ui.markdown(DETAILS.get(key, "내용이 없습니다."))

    # --- 증거 패널 ---
    @output
    @render.ui
    def evidence_panel():
        key = selected.get()
        if key == "drop_heating_furnace":
            return ui.TagList(
                ui.h4("특정 인덱스 구간 (73406–73418)"),
                ui.output_data_frame("hf_slice_df"),
                ui.hr(),
                ui.markdown(EVIDENCE_DESC.get(key, "설명 추가하기")),
            )
        return ui.markdown("해당 항목은 간단 설명만 제공합니다.")

    # --- heating_furnace: 특정 인덱스 구간 미리보기 (원본 문자열 그대로 표시) ---
    @output
    @render.data_frame
    def hf_slice_df():
        if selected.get() != "drop_heating_furnace":
            return pd.DataFrame()
    
        df = _raw_df()
        required = ["heating_furnace", "mold_code", "time", "date", "molten_volume", "count"]
        if not set(required).issubset(df.columns):
            return pd.DataFrame()
    
        start, end = 73406, 73418
        subset = df.loc[start:end, required].copy() if df.index.max() >= end else df.loc[:, required].head(12).copy()
    
        # ★ 새 DataFrame으로 명시적 재구성: time ← 원래 'date', date ← 원래 'time'
        display = pd.DataFrame({
            "index":         subset.index,                 # 인덱스도 보이고 싶다면 유지
            "heating_furnace": subset["heating_furnace"].astype(str),
            "mold_code":       subset["mold_code"].astype(str),
            "time":            subset["date"].astype(str),   # 날짜를 time 열에
            "date":            subset["time"].astype(str),   # 시:분:초를 date 열에
            "molten_volume":   subset["molten_volume"].astype(str),
            "count":           subset["count"].astype(str),
        })
    
        # 결측값 표기 통일
        display = display.replace({pd.NA: "Nan", None: "Nan", "": "Nan"}).fillna("Nan")
        return display



    
    # --- PDF 상태 안내 ---
    @output
    @render.text
    def pdf_status():
        return f"파일 위치: {PDF_FILE}" if PDF_FILE.exists() \
            else "주의: PDF 파일이 아직 없습니다. reports/preprocessing_report.pdf 경로에 파일을 생성해 주세요."

    # --- PDF 다운로드 ---
    @render.download(filename="preprocessing_report.pdf")
    def download_pdf():
        if PDF_FILE.exists():
            with open(PDF_FILE, "rb") as f:
                chunk = f.read(8192)
                while chunk:
                    yield chunk
                    chunk = f.read(8192)
        else:
            # 아주 작은 placeholder PDF
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
