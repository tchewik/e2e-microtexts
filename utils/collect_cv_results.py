import json
import os
from glob import glob

import fire
import pandas as pd


def collect_cv_results(model_name, filename):
    evaluations = []
    for eval_file in glob(os.path.join(model_name, 'fold_*', 'evaluations_test.json')):
        evaluations.append(json.load(open(eval_file, 'r')))

    ev = pd.DataFrame(evaluations)
    for key in ('cc', 'ro', 'fu', 'at'):
        ev[key] = ev[key].map(lambda row: row['fscore'])
    ev.lat = ev.lat.map(lambda row: row['accuracy'])
    result = pd.DataFrame({'mean': ev.mean(), 'std': ev.std()}).T
    result.to_csv(filename)


if __name__ == "__main__":
    """ Example: $ python collect_cv_results.py --model_name dp_rubert --filename lr_bert_0.0001_lr_parser_0.01 """
    fire.Fire(collect_cv_results)
