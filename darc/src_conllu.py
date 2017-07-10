from collections import namedtuple, Counter


cols = 'id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc'
Sent = namedtuple('Sent', cols + ('multi', ))
Sent.cols = cols
del cols
Sent.obsc = "_"
Sent.root = "\xa0"
Sent.dumb = ""
# obsc & root & dumb are all sentinal symbols.
# dumb stands for dummy or missing values, including the pseudo root;
# root is recognized by the parser when a dummy value is at the root position;
# obsc is used for unknown or rare values.


def cons(lines, dumb=Sent.dumb):
    """[str] -> Sent"""
    multi = []
    nodes = [[0, dumb, dumb, dumb, dumb, dumb, dumb, dumb, dumb, dumb]]
    for line in lines:
        node = line.split("\t")
        assert 10 == len(node)
        try:
            node[0] = int(node[0])
        except ValueError:
            if "-" in node[0]:
                multi.append(line)
        else:
            try:  # head might be empty for interim results
                node[6] = int(node[6])
            except ValueError:
                pass
            nodes.append(node)
    return Sent(*zip(*nodes), tuple(multi))


def fmap_x2u_deprel(sent):
    """-> Sent; acl:relcl -> acl"""
    return sent._replace(deprel=tuple(drel.split(":")[0] for drel in sent.deprel))


Sent.cons = cons
del cons
Sent.fmap_x2u_deprel = fmap_x2u_deprel
del fmap_x2u_deprel


def load(file, dumb=Sent.dumb):
    """-> iter([Sent])"""
    with open(file, encoding='utf-8') as file:
        sent = []
        for line in file:
            line = line.strip()
            if line.startswith("#"):
                pass
            elif line:
                sent.append(line.replace(" ", "\xa0"))
            elif sent:
                yield Sent.cons(sent, dumb)
                sent = []
        if sent:
            yield Sent.cons(sent, dumb)


def save(sents, file):
    """as conllu file"""
    with open(file, 'w', encoding='utf-8') as file:
        for sent in sents:
            multi_idx = [int(multi[:multi.index("-")]) for multi in sent.multi]
            w, m = 1, 0
            while w < len(sent.id):
                if m < len(multi_idx) and w == multi_idx[m]:
                    line = sent.multi[m]
                    m += 1
                else:
                    line = "\t".join([str(getattr(sent, col)[w]) for col in Sent.cols])
                    w += 1
                file.write(line.replace("\xa0", " "))
                file.write("\n")
            file.write("\n")


def select(sents, col='form', min_freq=2, obsc=Sent.obsc):
    """-> iter([[str]])"""
    freq = Counter(x for sent in sents for x in getattr(sent, col))
    for sent in sents:
        yield [x if min_freq <= freq[x] else obsc for x in getattr(sent, col)]


# import src_ud2 as ud2
# sents = list(load(ud2.path('de', 'dev')))
# save(sents, "./lab/tmp.conllu")
# sents2 = list(load("./lab/tmp.conllu"))
# assert sents == sents2
