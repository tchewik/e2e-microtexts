import os
from evidencegraph.corpus import GraphCorpus, CORPORA
from evidencegraph.argtree import FULL_RELATION_SET, SIMPLE_RELATION_SET


base_corpora_dir = os.path.join('evidencegraph', 'data', 'corpus')
CORPORA.update({
    'en_112': {'language': 'en',
                'path': os.path.join(base_corpora_dir, 'en_112')},
    'ru_112': {'language': 'ru',
                'path': os.path.join(base_corpora_dir, 'ru_112')},
    'en_full': {'language': 'en',
                'path': os.path.join(base_corpora_dir, 'en_full')},
    'ru_full': {'language': 'ru',
                'path': os.path.join(base_corpora_dir, 'ru_full')},
    'ru2en_full': {'language': 'en',
                'path': os.path.join(base_corpora_dir, 'ru2en_full')},
    'en2ru_full': {'language': 'ru',
                'path': os.path.join(base_corpora_dir, 'en2ru_full')},
})


def collect_eg_data(corpus_name, out_dir):
    """ Takes argument structures and saves them as dependencies in .conll-like format """
    
    corpus = GraphCorpus()
    corpus.load(CORPORA[corpus_name]["path"])

    language = CORPORA[corpus_name]["language"]
    trees = corpus.trees('adu', relation_set=SIMPLE_RELATION_SET)
    texts, trees = corpus.segments_trees(segmentation='adu', relation_set=SIMPLE_RELATION_SET)
    
    new_path_name = os.path.join(out_dir, corpus_name)
    if not os.path.isdir(new_path_name):
        os.mkdir(new_path_name)
        
    header = '\t'.join(['#', 'ADU', 'ro', 'at', 'fu'])

    for doc_id in trees.keys():
        # print(doc_id)
        tree = trees[doc_id]
        text = texts[doc_id]

        cc = tree.get_cc()
        roles_mapping = {0: 'pro', 1: 'opp'}
        roles = [roles_mapping[ro] for ro in tree.get_ro_vector()]
        relations = [tree.edge[i+1] or {0: {'type': 'cc'}} for i in range(len(roles))]
        attachments = [list(rel.keys())[0] for rel in relations]
        functions = [list(rel.values())[0]['type'] for rel in relations]

        with open(os.path.join(out_dir, corpus_name, doc_id + '.conll'), 'w') as f:
            for idx, adu, ro, at, fu in zip(range(len(text)), text, roles, attachments, functions):
                f.write('\t'.join(map(str, [idx+1, adu, ro, at, fu])))
                f.write('\n')
