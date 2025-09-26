# app.py
from shiny import App, ui
from modules import tab_analysis_copy, tab_model_performance, tab_process_explanation, tab_preprocessing

# 앱 UI 정의
app_ui = ui.page_navbar(
    tab_analysis_copy.panel(),          # 분석 탭
    tab_model_performance.panel(),      # 모델 성능 설명 탭
    tab_process_explanation.panel(),    # 공정 설명 탭
    tab_preprocessing.panel(),          # 전처리 요약 + PDF 다운로드 탭 (신규)
    title="공정 불량 분류 대시보드",
)

# 서버 정의
def server(input, output, session):
    tab_analysis_copy.server(input, output, session)
    tab_model_performance.server(input, output, session)
    tab_process_explanation.server(input, output, session)
    tab_preprocessing.server(input, output, session)  # 신규 탭 서버 연결

# 앱 실행 객체
app = App(app_ui, server)