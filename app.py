"""MNIST ONNX 손글씨 숫자 인식 — 공개 데모판 (2026-11 공개용 정리).

원래 미션 17 앱에서 입력 이미지 저장·공용 갤러리를 뺀 판이다.
방문자가 그린 그림은 이 실행(세션) 안에서 예측에만 쓰고 파일로 남기지 않는다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import altair as alt
import numpy as np
import onnxruntime as ort
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "mnist-12.onnx"
# ONNX Model Zoo(onnx/models)의 validated/vision/classification/mnist/model/mnist-12.onnx와 같은 파일
MODEL_SHA256 = "5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd"
LABELS = list(range(10))


@st.cache_resource(show_spinner="모델을 불러오는 중…")
def load_model() -> ort.InferenceSession:
    """저장소에 포함된 모델만 쓴다. 실행 중에 외부에서 내려받지 않고, 해시가 다르면 멈춘다."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH.name}")
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError(f"모델 해시가 기대값과 다릅니다: {digest[:16]}…")
    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def preprocess_canvas_image(image_data: np.ndarray) -> tuple[np.ndarray, Image.Image]:
    """Convert RGBA canvas output to a centered 1x1x28x28 float32 tensor."""
    rgba = Image.fromarray(image_data.astype(np.uint8), mode="RGBA")
    black_background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    merged = Image.alpha_composite(black_background, rgba).convert("L")

    image_array = np.array(merged)
    coords = np.argwhere(image_array > 20)
    if coords.size == 0:
        empty = Image.new("L", (28, 28), 0)
        return np.zeros((1, 1, 28, 28), dtype=np.float32), empty

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    digit = merged.crop((x_min, y_min, x_max, y_max))

    # Keep the digit centered and close to the MNIST training distribution.
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    left = (28 - digit.width) // 2
    top = (28 - digit.height) // 2
    canvas.paste(digit, (left, top))

    tensor = np.array(canvas, dtype=np.float32) / 255.0
    tensor = tensor.reshape(1, 1, 28, 28)
    return tensor, canvas


def predict(session: ort.InferenceSession, input_tensor: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logits = session.run([output_name], {input_name: input_tensor})[0][0]
    return softmax(logits)


def probability_chart(probabilities: np.ndarray) -> alt.Chart:
    chart_data = pd.DataFrame({"label": [str(label) for label in LABELS], "probability": probabilities})
    return (
        alt.Chart(chart_data)
        .mark_bar(color="#3B82F6")
        .encode(
            x=alt.X(
                "label:N",
                title="숫자",
                sort=None,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "probability:Q",
                title="확률",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(titleAngle=0, titleAlign="left", titleX=-42, titleY=-8),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="숫자"),
                alt.Tooltip("probability:Q", title="확률", format=".2%"),
            ],
        )
        .properties(height=240)
    )


def clear_canvas() -> None:
    # 캔버스 key를 바꾸면 새 빈 캔버스가 그려진다 (세션 안의 상태만 바뀐다)
    st.session_state.canvas_round = st.session_state.get("canvas_round", 0) + 1


def main() -> None:
    st.set_page_config(page_title="MNIST ONNX 손글씨 숫자 인식", layout="wide")

    st.title("MNIST ONNX 손글씨 숫자 인식")
    st.caption(
        "검은 칸에 0~9 숫자 하나를 크게 그려 보세요. "
        "사전학습 MNIST ONNX 모델이 ONNX Runtime으로 0부터 9까지의 확률을 계산합니다. "
        "그린 그림은 예측에만 쓰고 파일로 저장하지 않으며, 다른 방문자에게 보이지 않습니다."
    )

    try:
        session = load_model()
    except (FileNotFoundError, ValueError) as error:
        st.error(f"모델을 불러오지 못했습니다. {error}")
        st.stop()

    canvas_col, processed_col, result_col = st.columns([1.05, 0.85, 1.25])

    with canvas_col:
        st.subheader("입력 캔버스")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=18,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            display_toolbar=False,
            key=f"digit_canvas_{st.session_state.get('canvas_round', 0)}",
        )

        if canvas_result.image_data is not None:
            input_tensor, processed_image = preprocess_canvas_image(canvas_result.image_data)
        else:
            input_tensor = np.zeros((1, 1, 28, 28), dtype=np.float32)
            processed_image = Image.new("L", (28, 28), 0)
        is_empty = processed_image.getbbox() is None
        probabilities = np.zeros(10, dtype=np.float64) if is_empty else predict(session, input_tensor)
        prediction = int(np.argmax(probabilities))

        # 휴대폰에서는 세 칸이 위아래로 쌓이므로, 스크롤하지 않아도 결과가 보이게 캔버스 바로 아래에 한 줄로 적는다
        button_col, quick_col = st.columns([1, 2], vertical_alignment="center")
        with button_col:
            st.button("지우기", on_click=clear_canvas)
        with quick_col:
            st.markdown("숫자를 그려 주세요" if is_empty else f"예측 숫자 **{prediction}**")

    with processed_col:
        st.subheader("전처리 이미지")
        st.image(ImageOps.colorize(processed_image.resize((168, 168)), black="black", white="white"))
        st.caption(f"모델 입력: 28×28 흑백, shape {tuple(input_tensor.shape)}")

    with result_col:
        st.subheader("모델 추론 결과")
        if is_empty:
            st.info("캔버스에 숫자를 그려 주세요.")
        else:
            st.metric("예측 숫자", prediction)
            st.caption(f"화면 표시 확률 {probabilities[prediction]:.2%} — 이 입력 한 장에 대한 모델 출력이며 정확도가 아닙니다.")
        st.altair_chart(probability_chart(probabilities), width="stretch")

    st.divider()
    st.caption(
        f"모델: ONNX Model Zoo `mnist-12.onnx` (MIT, SHA-256 앞 16자 `{MODEL_SHA256[:16]}`) · "
        "원래 미션 17 앱에서 입력 저장·공용 갤러리를 뺀 2026-11 공개용 정리판"
    )


if __name__ == "__main__":
    main()
