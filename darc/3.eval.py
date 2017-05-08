import src_conllu as conllu
from src_setup import Setup
from keras.models import Model, model_from_json
import numpy as np

system_model_path = "./conll17/system_model/"
udpiped_test_path = "./conll17/udpiped_test/"
system_parse_path = "./conll17/system_parse/"


def parse_save(lang, suffix):
    """parses with setup and models for evaluation"""
    sents = list(conllu.load("{}{}.conllu".format(udpiped_test_path, lang,)))
    setup = Setup.load("{}{}{}.npy".format(system_model_path, lang, suffix))
    for epoch in range(4, 16):
        bean = np.load("{}{}{}-e{:0>2d}.npy"
                       .format(system_model_path, lang, suffix, epoch)) \
                 .item()
        model = model_from_json(bean['model'])
        model.set_weights(bean['weights'])
        del bean
        outfile = "{}{}{}-e{:0>2d}.conllu" \
                  .format(system_parse_path, lang, suffix, epoch)
        conllu.save((setup.parse(model, sent) for sent in sents), outfile)
        print("written", outfile)


if '__main__' == __name__:
    from sys import argv
    for lang in argv[1:]:
        try:
            print("{}-nonp".format(lang))
            parse_save(lang, "-nonp")
        except FileNotFoundError:
            print("{}-nonp skipped".format(lang))
        try:
            print("{}-proj".format(lang))
            parse_save(lang, "-proj")
        except FileNotFoundError:
            print("{}-proj skipped".format(lang))
