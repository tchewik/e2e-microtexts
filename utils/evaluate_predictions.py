import json
import os
import string
from dataclasses import dataclass
import random

import fire
from evidencegraph.argtree import ArgTree
from evidencegraph.argtree import FULL_RELATION_SET, SIMPLE_RELATION_SET
from evidencegraph.corpus import GraphCorpus, CORPORA
from evidencegraph.evaluation import eval_prediction
from evidencegraph.folds import folds

# import networkx as nx

base_corpora_dir = os.path.join('baseline', 'evidencegraph', 'data', 'corpus')
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
active_relation_set = SIMPLE_RELATION_SET


@dataclass
class Span:
    id: str
    begin: int
    end: int
    role: int
    text: str

    def __str__(self):
        return f'{self.id}\t{self.role} {self.begin} {self.end}\t{self.text}'


@dataclass
class Relation:
    id: str
    function: str
    arg1: str
    arg2: str

    def __str__(self):
        return f'{self.id}\t{self.function} Arg1:{self.arg1} Arg2:{self.arg2}'


def collect_spans(prediction):
    spans = []
    begin = 0
    for i, txt_span in enumerate(prediction['spans']):
        n = i + 1
        spans.append(Span(id=f'T{n}',
                          begin=begin, end=begin + len(txt_span),
                          role=prediction['predicted_roles'][i],
                          text=txt_span))
        begin += 1 + len(txt_span)

    return spans


def collect_rels(prediction, no_same_arg=True):
    rels = []

    heads = prediction['predicted_heads']
    deprecated_functions = ['root'] + ['same-arg'] * no_same_arg
    functions = [function if (function not in deprecated_functions and head != 0) else 'cc' for function, head in
                 zip(prediction['predicted_functions'], heads)]

    dependencies = list(zip(heads, functions))

    for i, (head, func) in enumerate(dependencies):
        rels.append(Relation(id=f'R{i + 1}',
                             function=func,
                             arg1=f'T{i + 1}',
                             arg2=f'T{head}'))

    return rels


def prediction_to_triplet(prediction):
    rels = collect_rels(prediction)
    return [(int(rel.arg1[1:]), int(rel.arg2[1:]), rel.function) for rel in rels if rel.arg1 != rel.arg2]


def load_gold_data(corpus_key):
    """ Available corpus keys are in the updated CORPORA """

    corpus = GraphCorpus()
    corpus.load(CORPORA[corpus_key]["path"])

    # language = CORPORA[corpus_name]["language"]
    gold_trees = corpus.trees('adu', relation_set=active_relation_set)
    return gold_trees


def evaluate_trees(gold_trees: list, predictions: list, foldnum: int):
    filenames = folds[foldnum]

    pred, gold = [], []
    for prediction, filename in zip(predictions, filenames):
        pred_tree = ArgTree(from_triples=prediction_to_triplet(prediction), text_id=filename,
                            relation_set=active_relation_set)
        gold_tree = gold_trees[filename]

        if len(pred_tree.nodes()) == len(gold_tree.nodes()):
            # With caution! If there is a not-attached node, evaluation is not possible (it happened one time).
            pred.append(pred_tree)
            gold.append(gold_tree)

    evaluations = eval_prediction(gold_trees=gold, pred_trees=pred)
    return evaluations


def evaluate_predictions(model_name: str, corpus_key: str, foldnum: int):
    gold_trees = load_gold_data(corpus_key)
    for part in ('test',):
        # Works only for test as we don't have a list of dev data filenames
        predictions = []
        with open(f'{model_name}/fold_{foldnum}/predictions_{part}.json', 'r') as f:
            for line in f:
                predictions.append(json.loads(line))

        ev = evaluate_trees(gold_trees, predictions, foldnum)
        result = {key: value for key, value in ev}
        las = result['lat']
        result = {key: value.get('macro_avg', las) for key, value in ev}
        with open(f'{model_name}/fold_{foldnum}/evaluations_{part}.json', 'w') as f:
            f.write(json.dumps(result, indent=4))


def evaluate_json_trees(model_name: str, foldnum: int):
    """ This function collects triples from data without 'same-arg' relations. """

    pred_trees, gold_trees = [], []
    with open(f'{model_name}/fold_{foldnum}/predictions_test.json', 'r') as f:
        for line in f:
            filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
            pred = json.loads(line)
            pred_tree_triplets = prediction_to_triplet(pred)
            pred_tree = ArgTree(from_triples=pred_tree_triplets, text_id=filename,
                                relation_set=active_relation_set)
            gold_tree_triplets = [(i+1, int(head), function) for i, (function, head) in enumerate(pred['dependencies'])]
            gold_tree = ArgTree(from_triples=gold_tree_triplets, text_id=filename,
                                relation_set=active_relation_set)

            if len(pred_tree.nodes()) == len(gold_tree.nodes()):
                # With caution! If there is a not-attached node, evaluation is not possible (it happened one time).
                pred_trees.append(pred_tree)
                gold_trees.append(gold_tree)

    evaluations = eval_prediction(gold_trees=gold_trees, pred_trees=pred_trees)
    return evaluations


if __name__ == "__main__":
    """ Example: $ python evaluate_predictions.py --model_name dp_rubert --corpus_key ru_full --foldnum 0 """
    fire.Fire(evaluate_predictions)
