from src_conllu import Sent
from itertools import repeat
from bisect import bisect_left, insort_right


class Config(object):
    """active nodes: i,j = config.stack[-2:]"""
    __slots__ = 'sent', 'input', 'stack', 'graph', 'deprel'

    def __init__(self, **kwargs):
        super().__init__()
        for attr, val in kwargs.items():
            setattr(self, attr, val)

    @staticmethod
    def cons(sent):
        """"Sent -> Config"""
        n = len(sent.head)
        return Config(
            sent=sent,
            input=list(range(n - 1, 0, -1)),
            stack=[0],
            graph=[[] for _ in range(n)],
            deprel=list(repeat(Sent.dumb, n)))

    def is_terminal(self):
        """-> bool"""
        return not self.input and 1 == len(self.stack)

    def doable(self, act):
        """-> bool; act: 'shift' | 'right' | 'left' | 'swap'"""
        if 'shift' == act:
            return 0 != len(self.input)
        elif 'right' == act:
            return 2 <= len(self.stack)
        elif 'left' == act:
            return 2 <= len(self.stack) and 0 != self.stack[-2]
        elif 'swap' == act:
            return 2 <= len(self.stack) and 0 < self.stack[-2] < self.stack[-1]
        else:
            raise TypeError("unknown act: {}".format(act))

    def shift(self, _=None):
        """(σ, [i|β], A) ⇒ ([σ|i], β, A)"""
        self.stack.append(self.input.pop())

    def right(self, deprel):
        """([σ|i, j], B, A) ⇒ ([σ|i], B, A∪{(i, l, j)})"""
        j = self.stack.pop()
        i = self.stack[-1]
        insort_right(self.graph[i], j)
        # i -deprel-> j
        self.deprel[j] = deprel

    def left(self, deprel):
        """([σ|i, j], B, A) ⇒ ([σ|j], B, A∪{(j, l, i)})"""
        j = self.stack[-1]
        i = self.stack.pop(-2)
        insort_right(self.graph[j], i)
        # i <-deprel- j
        self.deprel[i] = deprel

    def swap(self, _=None):
        """([σ|i,j],β,A) ⇒ ([σ|j],[i|β],A)"""
        self.input.append(self.stack.pop(-2))

    def finish(self):
        """-> Sent"""
        graph = self.graph
        deprel = self.deprel
        if 1 < len(graph[0]):  # ensure single root
            h = graph[0][0]
            for d in graph[0][1:]:
                insort_right(graph[h], d)
                deprel[d] = 'parataxis'
        head = list(repeat(Sent.dumb, len(graph)))
        for h, ds in enumerate(graph):
            for d in ds:
                head[d] = h
        return self.sent._replace(head=tuple(head), deprel=tuple(deprel))


class Oracle(object):
    """three possible modes:

    0. proj=True, arc-standard

    1. lazy=False, non-proj with swap (Nivre 2009)

    2. default, lazy swap (Nivre, Kuhlmann, Hall 2009)

    """
    __slots__ = 'sent', 'mode', 'graph', 'order', 'mpcrt'

    def cons(sent, proj=False, lazy=True):
        """Sent -> Oracle"""
        self = Oracle()
        n = len(sent.head)
        self.sent = sent
        self.mode = 0
        self.graph = [[] for _ in range(n)]
        for i in range(1, n):
            self.graph[sent.head[i]].append(i)
        if proj:
            return self
        self.mode = 1
        self.order = list(range(n))
        self._order(0, 0)
        if not lazy:
            return self
        self.mode = 0
        self.mpcrt = list(repeat(None, n))
        config = Config.cons(sent)
        while not config.is_terminal():
            act, arg = self.predict(config)
            if not config.doable(act):
                break
            getattr(config, act)(arg)
        self._mpcrt(config.graph, 0, 0)
        self.mode = 2
        return self

    def _order(self, n, o):
        # in-order traversal ordering
        i = bisect_left(self.graph[n], n)
        for c in self.graph[n][:i]:
            o = self._order(c, o)
        self.order[n] = o
        o += 1
        for c in self.graph[n][i:]:
            o = self._order(c, o)
        return o

    def _mpcrt(self, g, n, r):
        # maximal projective component root
        self.mpcrt[n] = r
        i = 0
        for c in self.graph[n]:
            if self.mpcrt[c] is None:
                i = bisect_left(g[n], c, i)
                self._mpcrt(g, c, r if i < len(g[n]) and c == g[n][i] else c)

    def predict(self, config):
        """-> act: str, arg: (str | None)

        act: 'shift' | 'right' | 'left' (| 'swap')

        getattr(config, act)(arg)

        """
        if 1 == len(config.stack):
            return 'shift', None
        i, j = config.stack[-2:]
        if 0 != self.mode and self.order[i] > self.order[j]:
            if 1 == self.mode:
                return 'swap', None
            if not config.input or self.mpcrt[j] != self.mpcrt[config.input[-1]]:
                return 'swap', None
        if self.sent.head[i] == j and len(self.graph[i]) == len(config.graph[i]):
            return 'left', self.sent.deprel[i]
        if i == self.sent.head[j] and len(self.graph[j]) == len(config.graph[j]):
            return 'right', self.sent.deprel[j]
        return 'shift', None


# s = Sent.cons(
#     ["1\tA\t_\t_\t_\t_\t2\tDET\t_\t_",
#      "2\thearing\t_\t_\t_\t_\t3\tSBJ\t_\t_",
#      "3\tis\t_\t_\t_\t_\t0\tROOT\t_\t_",
#      "4\tscheduled\t_\t_\t_\t_\t3\tVG\t_\t_",
#      "5\ton\t_\t_\t_\t_\t2\tNMOD\t_\t_",
#      "6\tthe\t_\t_\t_\t_\t7\tDET\t_\t_",
#      "7\tissue\t_\t_\t_\t_\t5\tPC\t_\t_",
#      "8\ttoday\t_\t_\t_\t_\t4\tADV\t_\t_",
#      "9\t.\t_\t_\t_\t_\t3\tP\t_\t_"])
# o = Oracle.cons(s)
# assert o.order == [0, 1, 2, 6, 7, 3, 4, 5, 8, 9]
# assert o.mpcrt == [0, 2, 2, 3, 4, 5, 5, 5, 8, 9]

# s = Sent.cons(
#     ["1\tWho\t_\t_\t_\t_\t7\tNMOD\t_\t_",
#      "2\tdid\t_\t_\t_\t_\t0\tROOT\t_\t_",
#      "3\tyou\t_\t_\t_\t_\t2\tSUBJ\t_\t_",
#      "4\tsend\t_\t_\t_\t_\t2\tVG\t_\t_",
#      "5\tthe\t_\t_\t_\t_\t6\tDET\t_\t_",
#      "6\tletter\t_\t_\t_\t_\t4\tOBJ1\t_\t_",
#      "7\tto\t_\t_\t_\t_\t4\tOBJ2\t_\t_",
#      "8\t?\t_\t_\t_\t_\t2\tP\t_\t_"])
# o = Oracle.cons(s)
# assert o.order == [0, 6, 1, 2, 3, 4, 5, 7, 8]
# assert o.mpcrt == [0, 1, 2, 2, 4, 4, 4, 7, 8]

# def test_oracle(s, verbose=True, proj=False, lazy=True):
#     o = Oracle.cons(s, proj, lazy)
#     c = Config.cons(s)
#     while not c.is_terminal():
#         act, arg = o.predict(c)
#         if verbose:
#             if arg:
#                 print(act, arg)
#             else:
#                 print(act)
#         getattr(c, act)(arg)
#     assert s == c.finish()

# import src_conllu as conllu
# from glob import glob
# for file in glob("/data/ud-treebanks-conll2017/*/*.conllu"):
#     print("testing oracle on", file, "....")
#     for i, s in enumerate(conllu.load(file)):
#         try:
#             test_oracle(s, False)
#         except Exception:
#             print("failed at sent", i)
