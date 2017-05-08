from src_setup import Setup
import src_ud2 as ud2
import numpy as np

silver_train_path = "./conll17/silver_train/"
system_model_path = "./conll17/system_model/"


def make_setup(lang, proj):
    """-> Setup"""
    silver = "{}{}.conllu".format(silver_train_path, lang)
    form_w2v = "{}{}-form.w2v".format(silver_train_path, lang)
    lemm_w2v = "{}{}-lemm.w2v".format(silver_train_path, lang) \
               if lang not in ud2.no_lemma else None
    return Setup.make(silver, form_w2v, lemm_w2v, proj=proj)


def train_save(setup, suffix):
    """save the setup and model weights from epoch 4 to 16"""
    setup.save("{}{}{}.npy".format(system_model_path, lang, suffix), with_data=False)
    model = setup.model()
    for epoch in range(16):
        setup.train(model, verbose=2)
        if 4 <= epoch:
            np.save("{}{}{}-e{:0>2d}.npy"
                    .format(system_model_path, lang, suffix, epoch),
                    {'model': model.to_json(), 'weights': model.get_weights()})


if '__main__' == __name__:
    from sys import argv
    for lang in argv[1:]:
        print("{}-nonp".format(lang))
        train_save(make_setup(lang, False), "-nonp")
        print("{}-proj".format(lang))
        train_save(make_setup(lang, True), "-proj")
