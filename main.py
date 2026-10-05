import cv2
import os
import urllib.request

def download_cascade_if_missing(filename):
    """如果模型檔案不存在，自動從 OpenCV GitHub 下載"""
    if not os.path.exists(filename):
        print(f"提示：找不到 '{filename}'，正在自動下載模型檔案...")
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{filename}"
        try:
            urllib.request.urlretrieve(url, filename)
            print("下載完成！")
        except Exception as e:
            print(f"錯誤：模型檔案下載失敗，請檢查網路連線。({e})")

def main():
    cascade_path = 'haarcascade_frontalface_alt.xml'
    
    # 1. 檢查並自動下載模型檔
    download_cascade_if_missing(cascade_path)

    # 2. 📋 載入 Haar Cascade 分類器
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"錯誤：無法載入模型檔案 '{cascade_path}'")
        return

    # 3. 🎥 自動尋找可用的攝影機連接埠 (嘗試 0, 1, 2)
    cap = None
    for cam_index in [0, 1, 2]:
        temp_cap = cv2.VideoCapture(cam_index)
        if temp_cap.isOpened():
            ret, frame = temp_cap.read()
            if ret:
                cap = temp_cap
                print(f"成功連接攝影機 (連接埠索引: {cam_index})")
                break
            temp_cap.release()

    if cap is None:
        print("錯誤：無法開啟任何網路攝影機！請檢查攝影機是否已插好或已被其他應用程式佔用。")
        return

    print("正在啟動攝影機... 請在影像視窗上按下 'q' 鍵以結束程式。")

    # 4. 🔄 無窮迴圈：即時影像讀取與人臉偵測
    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法接收攝影機串流訊號，正在關閉程式...")
            break

        # 將影像轉換為灰階
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 進行多尺度人臉偵測
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )

        # 在偵測到的人臉位置繪製綠色矩形框
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 在左上角顯示即時偵測到的人臉數量
        cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 顯示即時影像視窗
        cv2.imshow('Real-time Face Detection', frame)

        # 5. 🛑 按下 'q' 鍵跳出迴圈
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. 🧹 釋放資源並關閉視窗
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()