import cv2


video = cv2.VideoCapture()
#ip = 'https://10.107.242.45:8080/video'
video.open(0)

while True:
    check, img = video.read()

    cv2.imshow("Imagem Original", img)  #Saida
    key = cv2.waitKey(1)

    #Apartar algum botão para sair
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()