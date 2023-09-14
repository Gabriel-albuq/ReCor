import cv2
from ultralytics import YOLO
import time
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
import csv

save_directory_image = os.getcwd() + "\\core\\detect\\image\\"
save_directory_csv = os.getcwd() + "\\core\\detect\\files_csv\\"
last_frame = None
classify_rgb = None
contador_padrao = 0
contador_claro = 0
contador_escuro = 0
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
    global contador_padrao
    global contador_claro
    global contador_escuro
    global last_frame 
    global detected_class
    global save_directory_image
    global save_directory_csv
    global classify_rgb

    escala = 6
    espessura = 4
    cor_texto = (255, 255, 255)  # Cor do texto em branco (BGR)

    last_capture_time = time.time()
    intervalo = 1

    while True:
        current_time = time.time()

        if classify_rgb is not None: 
            image_bytes = cv2.imencode('.jpg', classify_rgb )[1].tobytes()
            yield (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n\r\n')

        if current_time - last_capture_time >= intervalo:
            last_capture_time = current_time

            if last_frame is not None:
                frame = last_frame  # Use o último frame capturado
                results = model_detect(frame)  # Execute o modelo

                if len(results[0]) > 0:
                    det = results[0][0]
                    bbox = det.boxes.xyxy[:4].cpu().numpy()  # Coordenadas da caixa delimitadora (xmin, ymin, xmax, ymax)
                    conf = float(det.boxes.conf.cpu().numpy())   # Confiança da detecção
                    class_id = "Biscoito"
                    #class_id = det.names[int(det.boxes.cls)]  # ID da classe

                    if conf > 0.2:  # Considerar apenas detecções com confiança acima de 0.# Desenhar a caixa delimitadora na imagem
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

                        if classif == "Padrao":
                            contador_padrao = contador_padrao + 1

                        if classif == "Claro":
                            contador_claro = contador_claro + 1

                        if classif == "Escuro":
                            contador_escuro = contador_escuro + 1

                        cv2.putText(classify_imagem_convert, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.putText(classify_imagem_convert, label_classify, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        classify_rgb = classify_imagem_convert[...,:3][...,::-1]

                        largura_retangulo = 1000
                        if classif == "Padrao":
                            cor_fundo = (53,130,84)
                        else:
                            cor_fundo = (51,0,153)
                        altura, largura, _ = classify_rgb.shape
                        imagem_com_retangulo = np.zeros((altura, largura + largura_retangulo, 3), dtype=np.uint8)
                        imagem_com_retangulo[:, :largura, :] = classify_rgb
                        cv2.rectangle(imagem_com_retangulo, (largura, 0), (largura + largura_retangulo, altura), cor_fundo, thickness=cv2.FILLED)

                        cv2.putText(imagem_com_retangulo, classif, (int(largura + 20), int((altura/2)+30)), cv2.FONT_HERSHEY_SIMPLEX, escala, cor_texto, espessura)

                        classify_rgb = imagem_com_retangulo

                        #Salvar imagem da classificação
                        dt_time = ' '.join([datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%H:%M:%S')])
                        name_file = dt_time.replace(' ', '').replace('-', '').replace(':', '')
                        save_path = os.path.join(save_directory_image, f'{name_file}.jpg')
                        cv2.imwrite(save_path, classify_rgb)

                        #Salvar Arquivo da classificação
                        save_path = os.path.join(save_directory_csv, f'{name_file}.csv')
                        with open(save_path, mode='w', newline='') as file_csv:
                            escritor_csv = csv.writer(file_csv)
                            cabecalho = ["datahora","classe","confclasse","principal_cor","principal_confcor",f"conf_{classify[0].names[0]}",f"conf_{classify[0].names[1]}",f"conf_{classify[0].names[2]}"]
                            texto = [dt_time,class_id,conf,classif,conf_classify,classify[0].probs.data[0].item(),classify[0].probs.data[1].item(),classify[0].probs.data[2].item()]
                            escritor_csv.writerow(cabecalho)
                            escritor_csv.writerow(texto)
                            file_csv.close()
                        #print(save_path)

def opencv_contador_padrao():
    global contador_padrao
    global contador_claro
    global contador_escuro

    altura, largura = 140, 500  # Altura e largura da imagem
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala = 1
    espessura = 2
    cor_texto = (255, 255, 255)  # Cor do texto em branco (BGR)
    y_positions = [30, 80, 130]
    labels = ["Padrao", "Claro", "Escuro"]
    cores_fundo = [(53,130,84),(51,0,153),(51,0,153)]

    while True:
        fundo_azul = np.zeros((altura, largura, 3), dtype=np.uint8)
        fundo_azul[:] = (173, 125, 97)

        def adicionar_texto(imagem, texto, label, y_pos, cor_fundo):
            cv2.rectangle(fundo_azul, (5,(y_pos-25)), (200,(y_pos+5)), cor_fundo, thickness=cv2.FILLED)
            largura_texto, altura_texto = cv2.getTextSize(texto, fonte, escala, espessura)[0]
            posicao_texto = ((largura - 300) - largura_texto, y_pos)
            cv2.putText(imagem, texto, posicao_texto, fonte, escala, cor_texto, espessura)
            cv2.putText(imagem, label, (210, y_pos), fonte, escala, cor_texto, espessura)

        # Textos que você deseja adicionar
        textos = [str(contador_padrao), str(contador_claro), str(contador_escuro)]
        
        # Adicione os textos à imagem usando a função adicionar_texto
        for texto, label, y_pos, cor_fundo in zip(textos, labels, y_positions, cores_fundo):
            adicionar_texto(fundo_azul, texto, label, y_pos, cor_fundo)

        # Converter a imagem OpenCV para JPEG
        _, buffer = cv2.imencode('.jpg', fundo_azul)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')# Mostrar a imagem resultante
