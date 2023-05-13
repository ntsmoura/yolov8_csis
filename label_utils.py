import glob, os
from copy import copy
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
            content = source_file.read()
        with open(dest_path + file_path, "a+") as dest_file:
            if content:
                dest_file.write(f"\n{content}")
    print("Opeação concluida!")


"""print_infos(
    count_yolo_classes(
        ["spray", "graffiti", "gun", "fire", "smoke", "knife", "puddle", "mud", "person"],
        "C:/yolov8_csis/predicted_labels",
    )
)"""

merge_yolo_files("C:/yolov8_csis/fake_origin", "C:/yolov8_csis/fake_dest")
