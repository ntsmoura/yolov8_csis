import cv2
import glob
import os

from ultralytics import YOLO


def detect_class_inside_dataset(source_path: str, dest_path: str, original_index: int, final_index: int):
    """
    Detecta uma classe específica num conjunto de imagens definidas em source_path, salvando suas labels referentes
    em dest_path, as labels terão o mesmo nome que as imagens. (Evita detectar imagens que já possuem lables em
    dest_path)

    :param source_path: Path de origem das imagens.
    :param dest_path: Path de destino da label das imagens.
    :param original_index: Index original da classe.
    :param final_index: Index final da classe (que será salvo nas labels).
    """
    model = YOLO("yolov8x.pt")

    os.chdir(dest_path)

    already_done = [os.path.basename(txt) for txt in glob.glob("*.txt")]

    os.chdir(source_path)

    for image_path in glob.glob("*.jpe?g"):

        image = cv2.imread(image_path)

        results = model.predict(source=image, classes=original_index)

        predicitons_list = []
        for box in results[0].boxes:
            positions = box.xywhn.tolist()[0]
            predicitons_list.append(f"{final_index} {' '.join(str('%.7f' % p) for p in positions)}")

        dest_path = dest_path + "/" if dest_path[-1] != "/" else dest_path
        with open(dest_path + os.path.basename(image_path), "w") as file:
            file.write("\n".join(predicitons_list))


detect_class_inside_dataset("./images", "./predicted_labels", 0, 8)