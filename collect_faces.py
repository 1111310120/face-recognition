import cv2
import os

def collect_faces():
    person_name = input("請輸入人員姓名或ID (例如: student01): ").strip()
    if not person_name:
        print("姓名不能為空！")
        return

    save_dir = os.path.join('data', person_name)
    os.makedirs(save_dir, exist_ok=True)

    cascade_path = 'haarcascade_frontalface_alt.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print(f"錯誤：無法載入分類器檔案 {cascade_path}")
        return

    cap = cv2.VideoCapture(0)
    count = 0
    max_samples = 30

    print("開始採集人臉，請面對攝影機... (按下 'q' 可提前退出)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法取得 Webcam 畫面")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            
            file_path = os.path.join(save_dir, f"{count}.jpg")
            cv2.imwrite(file_path, face_img)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {count}/{max_samples}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Collect Faces', frame)

        if cv2.waitKey(100) & 0xFF == ord('q') or count >= max_samples:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"完成採集！照片已存入: {save_dir}")

if __name__ == '__main__':
    collect_faces()