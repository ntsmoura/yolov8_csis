import logging

from ultralytics import YOLO

import cv2
import glob
import os


def detect_class_inside_dataset(source_path: str, dest_path: str, original_index: int, final_index: int):
    """
    Detecta uma classe específica num conjunto de imagens definidas em source_path, salvando suas labels referentes
    em dest_path, as labels terão o mesmo nome que as imagens. (Evita detectar imagens que já possuem lables em
    dest_path). Suporta imagens em jpg, jpeg e png.

    :param source_path: Path de origem das imagens.
    :param dest_path: Path de destino da label das imagens.
    :param original_index: Index original da classe.
    :param final_index: Index final da classe (que será salvo nas labels).
    """
    model = YOLO("yolov8x.pt")

    os.chdir(dest_path)

    already_done = [os.path.splitext(txt)[0] for txt in glob.glob("*.txt")]

    os.chdir(source_path)

    image_paths = glob.glob("*.jpeg") + glob.glob("*.jpg") + glob.glob("*.png")

    for image_path in image_paths:

        image_name = os.path.splitext(image_path)[0]
        if image_name in already_done:
            print(f"Imagem {image_path}: Já realizamos a predição, não iremos fazer novamente.")
            continue

        image = cv2.imread(image_path)

        print(f"Imagem {image_path}: Iniciando predição...")

        results = model.predict(source=image, classes=original_index)

        print(f"Imagem {image_path}: Predição encerrada. {len(results[0].boxes)} objetos encontrados!")

        predicitons_list = []
        for box in results[0].boxes:
            positions = box.xywhn.tolist()[0]
            predicitons_list.append(f"{final_index} {' '.join(str('%.7f' % p) for p in positions)}")

        print(f"Imagem {image_path}: Salvando label...")
        dest_path = dest_path + "/" if dest_path[-1] != "/" else dest_path
        with open(dest_path + image_name + ".txt", "w") as file:
            file.write("\n".join(predicitons_list))
        print(f"Imagem {image_path}: Label salva!")


detect_class_inside_dataset("C:/yolov8_csis/test_dir", "C:/yolov8_csis/predicted_labels", 0, 8)
