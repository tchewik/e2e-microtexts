import json
import os
from glob import glob

import fire
import numpy as np

from utils.evaluate_predictions import evaluate_predictions


def report_to_latex(report: dict, model_name: str):
    header = "Path  " + "  & ".join(report.keys()) + "\\ \midrule\n"
    values = [model_name] + [f"{value[0]} ± {value[1]}" for value in report.values()]
    body = "  & ".join(values)
    with open(os.path.join(model_name, 'eval_dbap_report.tex'), 'w') as f:
        f.write(header)
        f.write(body)


def evaluate(dir: str, k: int, gold_corpus='en_full', as_latex=False):
    for foldnum in range(k):
        evaluate_predictions(dir, gold_corpus, foldnum)

    evaluations = []
    for eval_file in sorted(glob(os.path.join(dir, 'fold_*', 'evaluations_test.json'))):
        evaluations.append(json.load(open(eval_file, 'r')))
        metrics_file = eval_file.replace('evaluations_test.json', 'metrics.json')
        evaluations[-1]['UAS'] = json.load(open(metrics_file, 'r'))['test_UAS']
        evaluations[-1]['LAS'] = json.load(open(metrics_file, 'r'))['test_LAS']

    evaluations = evaluations[:k]

    metric_res = lambda values: [(np.mean(values) * 100).round(1), (np.std(values) * 100).round(1)]
    results = dict()

    for metric in ('cc', 'ro', 'fu', 'at'):
        evals = [fold_eval[metric]['fscore'] for fold_eval in evaluations]
        results[metric] = metric_res(evals)

    for metric in ('UAS', 'LAS'):
        results[metric] = metric_res([fold_eval[metric] for fold_eval in evaluations])

    if as_latex:
        report_to_latex(results, dir)
    else:
        return results


if __name__ == "__main__":
    """ Example: $ python eval_dbap.py --dir models/dbap/dbap_ru --k 10 --as_latex"""
    fire.Fire(evaluate)
