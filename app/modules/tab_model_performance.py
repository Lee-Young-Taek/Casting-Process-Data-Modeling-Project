# modules/tab_process_explanation.py
from shiny import ui

def panel():
    return ui.nav_panel(
        "모델 성능 평가",
        ui.layout_column_wrap(
            ui.card(
                ui.h3("모델별 중요 변수"),
                ui.markdown(
                    """
                    """
                ),
            ),
            ui.card(
                ui.h3("모델별 성능 지표"),
                ui.markdown(
                    """
                    """
                ),
            ),
            fill=False,  # 카드 폭이 너무 넓어지지 않게
        ),
    )

def server(input, output, session):
    # 이 탭은 출력/반응 없음 (텍스트 전용)
    pass
