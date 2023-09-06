import cv2
from ultralytics import YOLO
import time
from datetime import datetime

last_frame = None
model = YOLO("core/best_detect.pt")

def stream():
    global last_frame

    cap = cv2.VideoCapture('https://10.0.0.101:8080/video')
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

    last_capture_time = time.time()
    intervalo = 3

    while True:
        current_time = time.time()
        #print("Tempo:", (current_time - last_capture_time))

        if current_time - last_capture_time >= intervalo:
            last_capture_time = current_time

            if last_frame is not None:
                frame = last_frame  # Use o último frame capturado
                results = model(frame)  # Execute o modelo
        
                for det in results[0]:
                    bbox = det.boxes.xyxy[:4].cpu().numpy()  # Coordenadas da caixa delimitadora (xmin, ymin, xmax, ymax)
                    conf = float(det.boxes.conf.cpu().numpy())   # Confiança da detecção
                    class_id = det.names[int(det.boxes.cls)]  # ID da classe

                    if conf > 0.5:  # Considerar apenas detecções com confiança acima de 0.# Desenhar a caixa delimitadora na imagem
                        xmin, ymin, xmax, ymax = map(int,(bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3]))
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Cor verde, espessura Escrever o nome da classe e  a confiança
                        label = f"{class_id} - {conf*100:.2f}%"
                        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                #dt_time= datetime.now()
                #formato_texto = "%Y-%m-%d %H:%M:%S"  # Por exemplo, "Ano-Mês-Dia Hora:Minuto:Segundo"
                #dt_time_txt = dt_time.strftime(formato_texto)
                #up = TesteImagem(nome = dt_time_txt, foto = frame) #Salva a imagem
                #up.save()

                image_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n\r\n')