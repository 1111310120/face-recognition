import cv2
import json
import os

def recognize_faces():
    model_path = os.path.join('models', 'lbph_model.yml')
    labels_path = os.path.join('models', 'labels.json')
    cascade_path = 'haarcascade_frontalface_alt.xml'

    if not (os.path.exists(model_path) and os.path.exists(labels_path)):
        print("找不到模型或標籤檔，請先執行 train_lbph.py 進行訓練！")
        return

    # 載入 LBPH 模型與標籤映射
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    with open(labels_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
        # JSON 讀取的 key 是 string，轉為 int
        label_map = {int(k): v for k, v in label_map.items()}

    face_cascade = cv2.CascadeClassifier(cascade_path)
    cap = cv2.VideoCapture(0)

    # 距離閾值設定（距離越小越相似，可依實際狀況調整，通常 70~80 為界線）
    THRESHOLD = 75.0

    print("啟動即時人臉辨識中... (按 'q' 退出)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))

            # 進行 LBPH 辨識 (返回 label 與 distance 距離)
            label, distance = recognizer.predict(face_roi)

            # 判斷距離是否小於閾值
            if distance <= THRESHOLD:
                name = label_map.get(label, "Unknown")
            else:
                name = "Unknown"

            # 標示結果與距離值
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} ({round(distance, 1)})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('LBPH Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    recognize_faces()