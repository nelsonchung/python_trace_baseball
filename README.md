# python_trace_baseball

用 Python 產生一組「本壘裁判視角」的棒球投捕範例圖，並使用可插拔的
YOLO/OpenCV 偵測流程輸出棒球框選結果。

目前流程會：

- 產生 10 張投手出手到捕手接球的合成範例圖。
- 同步產生 YOLO 格式標註檔，方便後續訓練自訂 baseball detector。
- 優先使用 `ultralytics` YOLO 模型偵測；沒有模型時，改用 OpenCV 圓形/顏色
  偵測 fallback，仍會輸出框選後的結果圖。
- 輸出每張圖的 detection CSV 與 trajectory JSON。

## 快速開始

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m baseball_trace.pipeline
```

產出位置：

- `data/samples/frame_01.png` 到 `frame_10.png`
- `data/labels/frame_01.txt` 到 `frame_10.txt`
- `data/results/frame_01_detected.png` 到 `frame_10_detected.png`
- `data/results/detections.csv`
- `data/results/trajectory.json`

## 使用 YOLO 模型

如果已有訓練好的棒球模型：

```bash
python -m pip install -r requirements-yolo.txt
python -m baseball_trace.pipeline --model path/to/baseball.pt
```

YOLO 類別名稱只要包含 `baseball`、`sports ball` 或 `ball` 都會被接受。若未指定
`--model`，流程會使用 OpenCV fallback，適合先驗證影像產生與框選輸出。

## 測試

```bash
python -m unittest discover -s tests
```
