import cv2
import os
import json
import numpy as np

def train_model():
    data_dir = 'data'
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    cascade_path = 'haarcascade_frontalface_alt.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = []
    labels = []
    label_map = {}
    current_label = 0

    if not os.path.exists(data_dir):
        print("未找到 data 資料夾，請先執行 collect_faces.py 收集資料。")
        return

    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        label_map[current_label] = person_name
        
        for file_name in os.listdir(person_dir):
            if file_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(person_dir, file_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    continue

                faces.append(img)
                labels.append(current_label)

        current_label += 1

    if len(faces) == 0:
        print("沒有找到有效的訓練影像！")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    model_path = os.path.join(models_dir, 'lbph_model.yml')
    labels_path = os.path.join(models_dir, 'labels.json')

    recognizer.write(model_path)
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=4)

    print(f"模型訓練完成！\n模型已儲存至: {model_path}\n標籤映射檔: {labels_path}")

if __name__ == '__main__':
    train_model()