# OpenCV Haar Cascade 即時人臉偵測專案 (VS Code)

本專案使用 Python 與 OpenCV 函式庫，採用 **Haar Cascade 特徵分類器** 進行網路攝影機（Webcam）即時人臉偵測與標記。

---

## 專案架構 (Project Structure)

```text
opencv-face-detection/
│
├── haarcascade_frontalface_alt.xml  # OpenCV 官方 Haar Cascade 特徵模型檔
├── main.py                          # 主要執行程式碼 (Webcam 即時偵測)
├── requirements.txt                 # 依賴套件清單
└── README.md                        # 專案說明文件