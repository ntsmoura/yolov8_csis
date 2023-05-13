from ultralytics import YOLO
import cv2

# Carrega o modelo yolov8x
model = YOLO("yolov8x.pt")

# Lê a imagem de fogo previamente selecionada
image = cv2.imread("./images/4400060234.jpg")

# Prediz a classe 0 (person) na imagem anterior, salvando tanto imagem anotada quanto posição
results = model.predict(source=image, classes=0)

predicitons_list = []
for box in results[0].boxes:
    positions = box.xywhn.tolist()[0]
    predicitons_list.append(
        f"8 {' '.join(str('%.7f' % p) for p in positions)}"
    )

with open("./predicted_labels/4400060234.txt", "w") as file:
    file.write("\n".join(predicitons_list))
