def parse_args():
    """-> argparse.Namespace"""
    import argparse
    parser = argparse.ArgumentParser(description="produce training data for word2vec.")
    parser.add_argument('--verbose', '-v', action='count', help="maximum verbosity: -v")
    parser.add_argument('--data', required=True, nargs='+', help="conllu files to use")
    parser.add_argument('--form', nargs='+', help="files to save the forms")
    parser.add_argument('--lemm', nargs='+', help="files to save the lemmas")
    args = parser.parse_args()
    from sys import exit
    if args.form:
        if len(args.data) != len(args.form):
            exit("the number of data files does not match the number of form files")
    elif args.lemm:
        if len(args.data) != len(args.form):
            exit("the number of data files does not match the number of lemm files")
    else:
        exit("nothing to be done")
    return args


if '__main__' == __name__:
    args = parse_args()
    from itertools import repeat
    if not args.form:
        args.form = repeat(None)
    if not args.lemm:
        args.lemm = repeat(None)
    from src_conllu import Sent
    import src_conllu as conllu
    for data, form, lemm in zip(args.data, args.form, args.lemm):
        if args.verbose:
            print("loading", data, "....")
        sents = list(conllu.load(data, dumb=Sent.root))
        if form:
            with open(form, 'w', encoding='utf-8') as file:
                for line in conllu.select(sents, col='form'):
                    file.write(" ".join(line))
                    file.write(" \n")
            if args.verbose:
                print("written", form, "....")
        if lemm:
            with open(lemm, 'w', encoding='utf-8') as file:
                for line in conllu.select(sents, col='lemma'):
                    file.write(" ".join(line))
                    file.write(" \n")
            if args.verbose:
                print("written", lemm, "....")
