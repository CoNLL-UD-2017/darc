"""
Contains the POS tags, dependency relations, and morphological features defined
by the Universal Dependencies project [0], for both of its major versions.

[0] http://universaldependencies.org/
"""

from collections import OrderedDict



"""
Tuples (one for each UD version) listing all possible universal POS tags with
the addition of a non-standard POS tag marking a sentence's root.

[1] http://universaldependencies.org/docsv1/u/pos/index.html
[2] http://universaldependencies.org/u/pos/index.html
"""
POS_TAGS_V1 = (
	'ROOT',  # non-standard! the root of a sentence
	'ADJ',  # adjective
	'ADP',  # adposition
	'ADV',  # adverb
	'AUX',  # auxiliary verb
	'CONJ',  # coordinating conjunction
	'DET',  # determiner
	'INTJ',  # interjection
	'NOUN',  # noun
	'NUM',  # numeral
	'PART',  # particle
	'PRON',  # pronoun
	'PROPN',  # proper noun
	'PUNCT',  # punctuation
	'SCONJ',  # subordinating conjunction
	'SYM',  # symbol
	'VERB',  # verb
	'X',  # other
)

POS_TAGS_V2 = (
	'ROOT',  # non-standard! the root of a sentence
	'ADJ',  # adjective
	'ADP',  # adposition
	'ADV',  # adverb
	'AUX',  # auxiliary
	'CCONJ',  # coordinating conjunction
	'DET',  # determiner
	'INTJ',  # interjection
	'NOUN',  # noun
	'NUM',  # numeral
	'PART',  # particle
	'PRON',  # pronoun
	'PROPN',  # proper noun
	'PUNCT',  # punctuation
	'SCONJ',  # subordinating conjunction
	'SYM',  # symbol
	'VERB',  # verb
	'X',  # other
)


"""
Tuples (one for each UD version) listing all possible universal dependency
relations.

[3] http://universaldependencies.org/docsv1/u/dep/index.html
[4] http://universaldependencies.org/u/dep/index.html
"""
DEP_RELS_V1 = (
	'acl',  # clausal modifier of noun (adjectival clause)
	'advcl',  # adverbial clause modifier
	'advmod',  # adverbial modifier
	'amod',  # adjectival modifier
	'appos',  # appositional modifier
	'aux',  # auxiliary
	'auxpass',  # passive auxiliary
	'case',  # case marking
	'cc',  # coordinating conjunction
	'ccomp',  # clausal complement
	'compound',  # compound
	'conj',  # conjunct
	'cop',  # copula
	'csubj',  # clausal subject
	'csubjpass',  # clausal passive subject
	'dep',  # unspecified dependency
	'det',  # determiner
	'discourse',  # discourse element
	'dislocated',  # dislocated elements
	'dobj',  # direct object
	'expl',  # expletive
	'foreign',  # foreign words
	'goeswith',  # goes with
	'iobj',  # indirect object
	'list',  # list
	'mark',  # marker
	'mwe',  # multi-word expression
	'name',  # name
	'neg',  # negation modifier
	'nmod',  # nominal modifier
	'nsubj',  # nominal subject
	'nsubjpass',  # passive nominal subject
	'nummod',  # numeric modifier
	'parataxis',  # parataxis
	'punct',  # punctuation
	'remnant',  # remnant in ellipsis
	'reparandum',  # overridden disfluency
	'root',  # root
	'vocative',  # vocative
	'xcomp',  # open clausal complement
)

DEP_RELS_V2 = (
	'acl',  # clausal modifier of noun (adjectival clause)
	'advcl',  # adverbial clause modifier
	'advmod',  # adverbial modifier
	'amod',  # adjectival modifier
	'appos',  # appositional modifier
	'aux',  # auxiliary
	'case',  # case marking
	'cc',  # coordinating conjunction
	'ccomp',  # clausal complement
	'clf',  # classifier
	'compound',  # compound
	'conj',  # conjunct
	'cop',  # copula
	'csubj',  # clausal subject
	'dep',  # unspecified dependency
	'det',  # determiner
	'discourse',  # discourse element
	'dislocated',  # dislocated elements
	'expl',  # expletive
	'fixed',  # fixed multiword expression
	'flat',  # flat multiword expression
	'goeswith',  # goes with
	'iobj',  # indirect object
	'list',  # list
	'mark',  # marker
	'nmod',  # nominal modifier
	'nsubj',  # nominal subject
	'nummod',  # numeric modifier
	'obj',  # object
	'obl',  # oblique nominal
	'orphan',  # orphan
	'parataxis',  # parataxis
	'punct',  # punctuation
	'reparandum',  # overridden disfluency
	'root',  # root
	'vocative',  # vocative
	'xcomp',  # open clausal complement
)


"""
Ordered dicts (one for each UD version) where the keys are the universal
features and the values are the respective features' possible values.

[5] http://universaldependencies.org/docsv1/u/feat/index.html
[6] http://universaldependencies.org/u/feat/index.html
"""
MORPH_FEATURES_V1 = {
	'Animacy': ('Anim', 'Inan', 'Nhum'),
	'Aspect': ('Imp ', 'Perf', 'Pro', 'Prog'),
	'Case': ('Abs', 'Acc', 'Erg', 'Nom',
		'Abe', 'Ben', 'Cau', 'Com', 'Dat', 'Dis',
		'Gen', 'Ins', 'Par', 'Tem', 'Tra', 'Voc',
		'Abl', 'Add', 'Ade', 'All', 'Del', 'Ela', 'Ess',
		'Ill', 'Ine', 'Lat', 'Loc', 'Sub', 'Sup', 'Ter'),
	'Definite': ('Com', 'Def', 'Ind', 'Red'),
	'Degree': ('Abs', 'Cmp', 'Pos', 'Sup'),
	'Gender': ('Com', 'Fem', 'Masc', 'Neut'),
	'Mood': ('Cnd', 'Des', 'Imp', 'Ind', 'Jus', 'Nec', 'Opt', 'Pot', 'Qot', 'Sub'),
	'Negative': ('Neg', 'Pos'),
	'NumType': ('Card', 'Dist', 'Frac', 'Gen', 'Mult', 'Ord', 'Range', 'Sets'),
	'Number': ('Coll', 'Dual', 'Plur', 'Ptan', 'Sing'),
	'Person': ('1', '2', '3'),
	'Poss': ('Yes',),
	'PronType': ('Art', 'Dem', 'Ind', 'Int', 'Neg', 'Prs', 'Rcp', 'Rel', 'Tot'),
	'Reflex': ('Yes',),
	'Tense': ('Fut', 'Imp', 'Nar', 'Past', 'Pqp', 'Pres'),
	'VerbForm': ('Fin', 'Ger', 'Inf', 'Part', 'Sup', 'Trans'),
	'Voice': ('Act', 'Cau', 'Pass', 'Rcp')
}

MORPH_FEATURES_V2 = {
	'Abbr': ('Yes',),
	'Animacy': ('Anim', 'Hum ', 'Inan ', 'Nhum'),
	'Aspect': ('Hab', 'Imp ', 'Iter', 'Perf', 'Prog', 'Prosp'),
	'Case': ('Abs', 'Acc', 'Erg', 'Nom',
		'Abe', 'Ben', 'Cau', 'Cmp', 'Com', 'Dat', 'Dis',
		'Equ', 'Gen', 'Ins', 'Par', 'Tem', 'Tra', 'Voc',
		'Abl', 'Add', 'Ade', 'All', 'Del', 'Ela', 'Ess',
		'Ill', 'Ine', 'Lat', 'Loc', 'Sub', 'Sup', 'Ter'),
	'Definite': ('Com', 'Cons', 'Def', 'Ind', 'Spec'),
	'Degree': ('Abs', 'Cmp', 'Equ', 'Pos', 'Sup'),
	'Evident': ('Fh', 'Nfh'),
	'Foreign': ('Yes',),
	'Gender': ('Com', 'Fem', 'Masc', 'Neut'),
	'Mood': ('Adm', 'Cnd', 'Des', 'Imp', 'Ind', 'Jus',
		'Nec', 'Opt', 'Pot', 'Prp', 'Qot', 'Sub'),
	'NumType': ('Card', 'Dist', 'Frac', 'Mult', 'Ord', 'Range', 'Sets'),
	'Number': ('Coll', 'Count', 'Dual', 'Grpa', 'Grpl',
		'Inv', 'Pauc', 'Plur', 'Ptan', 'Sing', 'Tri'),
	'Person': ('0', '1', '2', '3', '4'),
	'Polarity': ('Neg', 'Pos'),
	'Polite': ('Elev', 'Form', 'Humb', 'Infm'),
	'Poss': ('Yes',),
	'PronType': ('Art', 'Dem', 'Emp', 'Exc', 'Ind', 'Int',
		'Neg', 'Prs', 'Rcp', 'Rel', 'Tot'),
	'Reflex': ('Yes',),
	'Tense': ('Fut', 'Imp', 'Past', 'Pqp', 'Pres'),
	'VerbForm': ('Conv', 'Fin', 'Gdv', 'Ger', 'Inf', 'Part', 'Sup', 'Vnoun'),
	'Voice': ('Act', 'Antip', 'Cau', 'Dir', 'Inv', 'Mid', 'Pass', 'Rcp')
}

MORPH_FEATURES_V1 = OrderedDict(sorted(MORPH_FEATURES_V1.items(), key=lambda x: x[0]))
MORPH_FEATURES_V2 = OrderedDict(sorted(MORPH_FEATURES_V2.items(), key=lambda x: x[0]))
