# MNIST ONNX 손글씨 숫자 인식 — 공개 데모

캔버스에 숫자 하나를 그리면 사전학습 MNIST ONNX 모델이 ONNX Runtime으로 0~9 확률을 계산합니다.
입력 캔버스 → 28×28 전처리 이미지 → 예측 숫자와 클래스별 확률을 한 화면에서 보여 줍니다.

- 원래 미션 코드: [onnx-ai-streamlit-mnist-service-mission-17](https://github.com/malit9999/onnx-ai-streamlit-mnist-service-mission-17) (부트캠프 미션 17)
- 이 저장소: 위 미션 앱을 방문자가 직접 써 볼 수 있게 **2026-11에 공개용으로 정리한 판**입니다. 새로 학습한 모델이 아닙니다.

## 원래 미션에서 바꾼 점 (2026-11 공개용 정리)

| 항목 | 원래 미션 | 공개판 |
|---|---|---|
| 입력 저장 | "현재 이미지 저장" 버튼으로 `saved_images/`에 원본·전처리 PNG와 `metadata.json` 저장 | 저장 버튼·저장 폴더·메타데이터 파일 제거. 입력은 그 세션의 예측에만 쓰고 파일로 남기지 않음 |
| 갤러리 | 저장된 그림을 모든 방문자에게 보여 주는 "이미지 저장소" | 제거 |
| 모델 파일 | 없으면 실행 중 외부에서 내려받음 | 저장소에 포함된 파일만 사용, SHA-256이 다르면 실행을 멈춤 |
| 지우기 | 캔버스 도구 막대의 아이콘 | "지우기" 버튼 |
| 빈 입력 | 경고 표시 | "숫자를 그려 주세요" 안내 |
| 의존성 | 버전 고정 없음 | Python 3.12에서 확인한 버전으로 고정 (`requirements.txt`) |
| 기타 | Dockerfile·보고서 포함 | 공개 실행에 필요한 파일만 포함 |

전처리(`preprocess_canvas_image`)와 추론·softmax 코드는 원래 미션과 같습니다.

## 실행

Python 3.12 기준입니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 모델

- 파일: `models/mnist-12.onnx` (26,143바이트, SHA-256 `5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd`)
- 출처: [ONNX Model Zoo](https://github.com/onnx/models) `validated/vision/classification/mnist/model/mnist-12.onnx` — 위 해시는 Model Zoo에 등록된 파일의 해시와 같습니다.
- 라이선스: 모델 카드에 MIT로 표시되어 있습니다. 자세한 내용은 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)를 보세요.
- 화면에 나오는 확률은 그 입력 한 장에 대한 모델 출력이며, 정확도 지표가 아닙니다.

## 입력 처리

- 앱 코드에는 입력 이미지를 파일로 쓰거나 로그로 남기는 부분이 없습니다.
- Streamlit은 화면 표시를 위해 전처리 이미지를 서버 메모리에 잠시 두고, 그 세션 화면에만 보여 줍니다.
- 다른 방문자의 그림은 보이지 않습니다. 페이지를 새로 열면 빈 캔버스로 시작합니다.
