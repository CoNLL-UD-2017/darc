def parse_args():
    """-> argparse.Namespace"""
    import argparse
    parser = argparse.ArgumentParser(description="the parse is darc and full of errors.")
    parser.add_argument('--verbose', '-v', action='count', help="maximum verbosity: -v")
    parser.add_argument('--model', required=True, help="npy model file to load")
    parser.add_argument('--parse', required=True, nargs='+', help="conllu files to parse")
    parser.add_argument('--write', required=True, nargs='+', help="conllu files to write")
    args = parser.parse_args()
    if len(args.parse) != len(args.write):
        from sys import exit
        exit("the number of files to parse does not match the number of files to write")
    return args


if '__main__' == __name__:
    args = parse_args()
    if args.verbose:
        print("loading", args.model, "....")
    from src_setup import Setup
    setup, model = Setup.load(args.model, with_model=True)
    import src_conllu as conllu
    for parse, write in zip(args.parse, args.write):
        if args.verbose:
            print("parsing", parse, "....")
        conllu.save((setup.parse(sent, model) for sent in conllu.load(parse)), write)
        if args.verbose:
            print("written", write)
