import src_ud2 as ud2
import src_conllu as conllu
from src_setup import Setup
import json


udpiped_test_path = "./conll17/udpiped_test/"
system_model_path = "./conll17/system_model/"


def path_dir(path):
    """str -> str: valid dir path"""
    return path if path.endswith("/") else path + "/"


if '__main__' == __name__:
    from sys import argv
    assert 3 == len(argv)
    task_path = path_dir(argv[1])
    system_parse_path = path_dir(argv[2])

    with open(task_path + "metadata.json") as file:
        metadata = json.load(file)

    for task in metadata:
        outfile = task['outfile']
        lang = task['ltcode']
        if (lang not in ud2.treebanks) and (lang not in ud2.surprise):
            lang = task['lcode']

        print("loading model", lang, "....")
        setup, model = Setup.load("{}{}.npy".format(system_model_path, lang), with_model=True)
        conllu.save((setup.parse(model, sent) for sent in conllu.load(
            udpiped_test_path + outfile)),
                    system_parse_path + outfile)
        print("written", outfile)
