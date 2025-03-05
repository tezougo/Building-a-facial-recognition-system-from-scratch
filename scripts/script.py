import cv2
import numpy as np
import os
import onnxruntime as ort
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#Encontrar o índice correto da webcam
indice_webcam = None
for i in range(10):  # Testa até 10 dispositivos diferentes
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Câmera encontrada no índice {i}")
        indice_webcam = i
        cap.release()
        break

if indice_webcam is None:
    print("Nenhuma câmera encontrada. Encerrando...")
    exit()

#Carregar YOLO para detecção facial
face_cfg = "data/face.cfg"
face_weights = "data/face.weights"
net = cv2.dnn.readNet(face_weights, face_cfg)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

#Carregar modelo FaceNet ONNX
facenet_model_path = "data/facenet.onnx"
facenet_session = ort.InferenceSession(facenet_model_path)

#Função para obter embeddings faciais
def get_embedding(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Converter para tons de cinza
    face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)  # Expandir para 3 canais
    face = cv2.resize(face, (160, 160))  # Redimensiona para entrada do FaceNet
    face = face.astype("float32") / 255.0  # Normalização para [0,1]
    face = np.expand_dims(face, axis=0)  # Adiciona dimensão batch
    
    ort_inputs = {facenet_session.get_inputs()[0].name: face}
    embeddings = facenet_session.run(None, ort_inputs)[0]

    return embeddings[0] / np.linalg.norm(embeddings[0], ord=2)  # 🔹 Normalização L2

#Treinamento com KNN
known_faces_encodings = []
known_faces_names = []
known_faces_dir = "known_faces"

#Verifica se há imagens novas e as adiciona
for filename in os.listdir(known_faces_dir):
    image_path = os.path.join(known_faces_dir, filename)
    name = os.path.splitext(filename)[0]  # Nome sem extensão
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao carregar imagem: {filename}")
        continue

    embeddings = get_embedding(image)
    known_faces_encodings.append(embeddings)
    known_faces_names.append(name)

if len(known_faces_encodings) > 0:
    global knn
    knn = KNeighborsClassifier(n_neighbors=7, metric="euclidean", algorithm="auto", weights="distance")
    knn.fit(known_faces_encodings, known_faces_names)
    print(f"Modelo atualizado com {len(known_faces_encodings)} rostos conhecidos!")
else:
    print("Nenhuma nova imagem encontrada para treinar.")

#Treinar SVM inicialmente
known_faces_encodings = []
known_faces_names = []
known_faces_dir = "known_faces"

for filename in os.listdir(known_faces_dir):
    image_path = os.path.join(known_faces_dir, filename)
    name = os.path.splitext(filename)[0]  # Nome sem extensão
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao carregar imagem: {filename}")
        continue

    embeddings = get_embedding(image)
    known_faces_encodings.append(embeddings)
    known_faces_names.append(name)

if len(known_faces_encodings) > 0:
    global svm
    svm = SVC(kernel="sigmoid", probability=True)
    svm.fit(known_faces_encodings, known_faces_names)
    print(f"Modelo SVM atualizado com {len(known_faces_encodings)} rostos conhecidos!")
else:
    print("Nenhuma nova imagem encontrada para treinar.")

#Captura de vídeo ao vivo
cap = cv2.VideoCapture(indice_webcam, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, 60)

#Aguarda a câmera estar disponível antes de capturar os frames
while not cap.isOpened():
    print("Aguardando inicialização da câmera...")

print("Câmera inicializada com sucesso!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar frame")
        break

    height, width, _ = frame.shape

    #Processar imagem com YOLO
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    faces = []
    confidences = []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.99:
                center_x, center_y, w, h = (obj[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                faces.append([x, y, w, h])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(faces, confidences, 0.5, 0.4)

    if indices is not None and len(indices) > 0:
        for i in indices.flatten():  # 🔹 Ajustado para garantir compatibilidade
            x, y, w, h = faces[i]

            #Recorte do rosto detectado
            face_crop = frame[y:y+h, x:x+w]

            #Obter embeddings faciais com FaceNet ONNX
            face_embedding = get_embedding(face_crop)

            name = "Desconhecido"  # Nome padrão

            #Comparação com KNN
            if len(known_faces_encodings) > 0:
                predicted_name = knn.predict([face_embedding])[0]
                name_KNN = predicted_name

            #Comparação com SVM
            if len(known_faces_encodings) > 0:
                predicted_name = svm.predict([face_embedding])[0]
                name_SVM = predicted_name

            if name_KNN == name_SVM:
                name = name_KNN

            #Desenhar retângulo e exibir nome e coordenadas
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} (X:{x}, Y:{y})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #Exibir imagem
    cv2.imshow("Reconhecimento Facial com YOLO e FaceNet ONNX", frame)

    #Finaliza a execução do script ao fechar a janela
    if cv2.waitKey(1) & (cv2.getWindowProperty("Reconhecimento Facial com YOLO e FaceNet ONNX", cv2.WND_PROP_VISIBLE) < 1):
        break

cap.release()
cv2.destroyAllWindows()
print("Câmera desligada corretamente.")
