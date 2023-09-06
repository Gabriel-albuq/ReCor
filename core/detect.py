import cv2
from ultralytics import YOLO
import time
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2

last_frame = None
detected_class = "Esperando detecção"
model_detect = YOLO("core/best_detect.pt")
model_classify = YOLO("core/best_classify.pt")

def stream():
    global last_frame

    cap = cv2.VideoCapture(0)
    cap.set(3, 320)  # Largura
    cap.set(4, 240)  # Altura
    cap.set(5, 10)  # Define a taxa de quadros para 30 FPS
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar imagem")
            break
        
        last_frame = frame
        image_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n\r\n')

def detect():
    global last_frame 
    global detected_class

    last_capture_time = time.time()
    intervalo = 3

    while True:
        current_time = time.time()
        #print("Tempo:", (current_time - last_capture_time))

        if current_time - last_capture_time >= intervalo:
            last_capture_time = current_time

            if last_frame is not None:
                frame = last_frame  # Use o último frame capturado
                results = model_detect(frame)  # Execute o modelo
        
                for det in results[0]:
                    bbox = det.boxes.xyxy[:4].cpu().numpy()  # Coordenadas da caixa delimitadora (xmin, ymin, xmax, ymax)
                    conf = float(det.boxes.conf.cpu().numpy())   # Confiança da detecção
                    class_id = det.names[int(det.boxes.cls)]  # ID da classe

                    if conf > 0.5:  # Considerar apenas detecções com confiança acima de 0.# Desenhar a caixa delimitadora na imagem
                        xmin, ymin, xmax, ymax = map(int,(bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3]))
                        imagem_print = frame.copy()
                        cropped_image = imagem_print[int(ymin):int(ymax), int(xmin):int(xmax)]

                        #Aplicar transformações usando o Albumentations
                        transform = A.Compose([
                            A.Resize(416, 416),  # Redimensionar para o tamanho do modelo YOLO
                            ToTensorV2(),
                        ])

                        imagem_transformed = transform(image=cropped_image)
                        imagem_transformed = imagem_transformed["image"].numpy().transpose(1, 2, 0)
                        classify = model_classify.predict(imagem_transformed, show = False)
                        classify_imagem_convert = cv2.cvtColor(classify[0].orig_img, cv2.COLOR_BGRA2RGBA)

                        conf = float(det[0].boxes.conf.cpu().numpy())   # Confiança da detecção
                        class_id = det[0].names[int(det[0].boxes.cls)]  # ID da classe
                        detected_class = class_id
                        label = f"{class_id} - {conf*100:.2f}%"
                        
                        classif = classify[0].names[classify[0].probs.top5[0]]
                        conf_classify = classify[0].probs.top5conf[0].item()
                        label_classify = f"{classif} - {conf_classify*100:.2f}%"

                        cv2.putText(classify_imagem_convert, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.putText(classify_imagem_convert, label_classify, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        classify_rgb = classify_imagem_convert[...,:3][...,::-1]

                #dt_time= datetime.now()
                #formato_texto = "%Y-%m-%d %H:%M:%S"  # Por exemplo, "Ano-Mês-Dia Hora:Minuto:Segundo"
                #dt_time_txt = dt_time.strftime(formato_texto)
                #up = TesteImagem(nome = dt_time_txt, foto = frame) #Salva a imagem
                #up.save()

                        image_bytes = cv2.imencode('.jpg', classify_rgb )[1].tobytes()
                        yield (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n\r\n')
                        
def detect_class():
    global detected_class
    return detected_class
