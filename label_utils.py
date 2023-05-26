import glob, os
import random
from copy import copy
from math import floor, ceil
from typing import List


def count_yolo_classes(class_list: List[str], folder_path: str):
    """
    Conta a quantidade de labels de cada classe nos arquivos de texto do path especificado.

    :param class_list: Lista de classes do dataset
    :param folder_path: Path para a pasta com as labels
    :return: Dicionário no seguinte esquema: {'class_name': {'count': int, 'images': set()}* ...,'images_total': int,
    'images_wlabel': int}
    """
    infos_dict = {key: {"count": 0, "images": set()} for key in class_list}
    infos_dict["images_total"] = 0
    infos_dict["images_wlabel"] = 0
    os.chdir(folder_path)

    for file in glob.glob("*.txt"):
        infos_dict["images_total"] += 1
        with open(file, "r") as open_file:
            found_line = False
            for line in open_file:
                found_line = True
                key = class_list[int(line[0])]
                infos_dict[key]["count"] += 1
                infos_dict[key]["images"].add(os.path.basename(file))
            if not found_line:
                infos_dict["images_wlabel"] += 1

    return infos_dict


def print_infos(infos_dict: dict):
    """
    Imprime as informações do dicionário de informações de classes do dataset.

    :param infos_dict: Dicionário com informações sobre as classes no dataset.
    """

    dict_view = copy(infos_dict)
    print("----------------------- Informações Gerais -----------------------")
    print(f"Total de imagens do dataset: {dict_view['images_total']}")
    print(f"Total de imagens com label vazia: {dict_view['images_wlabel']}")
    del dict_view["images_total"]
    del dict_view["images_wlabel"]
    print("--------------------- Informações por Classe ---------------------")
    for key in dict_view:
        print(
            f"Classe: {key} - quantidade de imagens: {len(dict_view[key]['images'])} - "
            f"quantidade de labels: {dict_view[key]['count']}"
        )


def merge_yolo_files(source_path: str, dest_path: str):
    """
    Adiciona o conteúdo das labels dos arquivos do source_path aos do dest_path de mesmo nome.

    :param source_path: Path de origem das labels que serão adicionadas.
    :param dest_path: Path de destino das labels que serão modificadas.
    """

    print("Iniciando operação de merge...")
    dest_path = dest_path + "/" if dest_path[-1] != "/" else dest_path
    os.chdir(source_path)
    for file_path in glob.glob("*.txt"):
        with open(file_path, "r") as source_file:
            content = source_file.read().strip("\n")
        with open(dest_path + file_path, "r") as dest_file:
            content_dest = dest_file.read().strip("\n")
        with open(dest_path + file_path, "w") as dest_file:
            dest_file.write(f"{content_dest}\n{content}".strip("\n"))
    print("Operação concluida!")


def split_dataset(source_path: str, class_list: list[str], train: float, test: float, validation: float):
    """
    Divide o dataset em treino, teste e validação conforme proporção inserida. Realiza certa estratificação do dataset
    tentando manter uma divisão mais proporcional. Cria as pastas train, test e validation contendo as labels e imagens
    referentes. Começa a divisão por labels de menor ocorrência para maior.

    :param source_path: Path contendo imagens e labels.
    :param class_list: Lista de classes do dataset obedecendo a ordem das classes (ex: ["apple", "pie"])
    :param train: Porcentagem de treino (ex: 0.8)
    :param test: Porcentagem de teste (ex: 0.1)
    :param validation: Porcentagem de validação (ex: 0.1)
    """
    if train == 0 and test == 0:
        raise ValueError

    count_dict = count_yolo_classes(class_list, source_path)
    moved_files, train_files, test_files, validation_files = set(), set(), set(), set()

    classes = [count_dict[key] for key in count_dict if isinstance(count_dict.get(key), dict)]
    classes.sort(key=lambda i: i["count"])
    for info in classes:
        images = info["images"] - moved_files
        moved_files.update(images)

        train_images = set(random.sample(list(images), ceil(len(images) * train)))
        images = images - train_images

        test_images = set(random.sample(list(images), ceil(len(images) * (test / (test + validation)))))
        images = images - test_images

        validation_images = images

        train_files.update(train_images)
        test_files.update(test_images)
        validation_files.update(validation_images)

    all_txt = [os.path.basename(file) for file in glob.glob("*.txt")]
    all_txt = set(all_txt)
    remaining_files = all_txt - moved_files
    train_images = set(random.sample(list(remaining_files), ceil(len(remaining_files) * train)))
    remaining_files = remaining_files - train_images

    test_images = set(random.sample(list(remaining_files), ceil(len(remaining_files) * (test / (test + validation)))))
    remaining_files = remaining_files - test_images

    validation_images = remaining_files
    train_files.update(train_images)
    test_files.update(test_images)
    validation_files.update(validation_images)

    source_path = source_path + "/" if source_path[-1] != "/" else source_path
    labels_train_path = source_path + "labels/train"
    labels_test_path = source_path + "labels/test"
    labels_validation_path = source_path + "labels/val"

    images_train_path = source_path + "images/train"
    images_test_path = source_path + "images/test"
    images_validation_path = source_path + "images/val"

    labels_path = source_path + "labels"
    images_path = source_path + "images"

    paths_list = [
        labels_path,
        images_path,
        labels_train_path,
        labels_test_path,
        labels_validation_path,
        images_train_path,
        images_test_path,
        images_validation_path,
    ]

    for path in paths_list:
        try:
            os.mkdir(path)
        except FileExistsError:
            continue


"""print_infos(
    count_yolo_classes(
        ["spray", "graffiti", "gun", "fire", "smoke", "knife", "puddle", "mud", "person"],
        "C:/yolov8_csis/labels_dest",
    )
)"""

# merge_yolo_files("C:/yolov8_csis/labels_origin", "C:/yolov8_csis/labels_dest")

split_dataset(
    "C:/yolov8_csis/labels_dest",
    ["spray", "graffiti", "gun", "fire", "smoke", "knife", "puddle", "mud", "person"],
    0.8,
    0.1,
    0.1,
)
