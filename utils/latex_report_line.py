import fire
import numpy as np
from scipy import stats

from utils.evaluate_predictions import *


def model_latex_results(model_name, k=10):
    def latex_string(values):
        return '      & '.join(
            [f'{np.mean(value) * 100:.1f} ± {np.std(value) * 100:.1f}' for value in values]) + '      \\\\'

    cc, ro, fu, at, uat, lat = [], [], [], [], [], []
    for foldnum in range(k):
        evals = evaluate_json_trees(model_name, foldnum)
        cc.append(evals[0][1]['macro_avg']['fscore'])
        ro.append(evals[1][1]['macro_avg']['fscore'])
        fu.append(evals[2][1]['macro_avg']['fscore'])
        at.append(evals[3][1]['classwise']['1']['fscore'])
        uat.append(evals[4][1]['accuracy'])
        lat.append(evals[5][1]['accuracy'])

    return latex_string([cc, ro, fu, at, uat, lat])


def paired_ttest_kfold_cv(
        k_scores_1, k_scores_2
):
    """
    Implements the k-fold paired t test procedure
    to compare the performance of two models.

    Returns
    ----------
    t : float
        The t-statistic
    pvalue : float
        Two-tailed p-value.
        If the chosen significance level is larger
        than the p-value, we reject the null hypothesis
        and accept that there are significant differences
        in the two compared models.
    """
    k = len(k_scores_1)
    score_diff = [score1 - score2 for score1, score2 in zip(k_scores_1, k_scores_2)]
    sample_mean = np.mean(score_diff)

    sd = np.sqrt((np.sum([(score_diff[i] - sample_mean) ** 2 for i in range(k)])) / (k - 1))
    t_stat = np.sqrt(k) * sample_mean / sd

    pvalue = stats.t.sf(np.abs(t_stat), k - 1) * 2.0
    return float(t_stat), float(pvalue)


def stat_report(model1: str, model2: str, k=10):
    evaluations = {model_name: {key: [] for key in ['cc', 'ro', 'fu', 'at', 'uas', 'las']} for model_name in
                   (model1, model2)}
    for model_name in (model1, model2):
        for foldnum in range(k):
            evals = evaluate_json_trees(model_name, foldnum)
            evaluations[model_name]['cc'].append(evals[0][1]['macro_avg']['fscore'])
            evaluations[model_name]['ro'].append(evals[1][1]['macro_avg']['fscore'])
            evaluations[model_name]['fu'].append(evals[2][1]['macro_avg']['fscore'])
            evaluations[model_name]['at'].append(evals[3][1]['classwise']['1']['fscore'])
            evaluations[model_name]['uas'].append(evals[4][1]['accuracy'])
            evaluations[model_name]['las'].append(evals[5][1]['accuracy'])

    for part in ('cc', 'ro', 'fu', 'at', 'uas', 'las'):
        t_stat, p_value = paired_ttest_kfold_cv(evaluations[model1][part], evaluations[model2][part])
        print('{}:\tv={:.4f},\tp={:.4f} {}'.format(part, t_stat, p_value,
                                                   '*' * (p_value < 0.05) + '*' * (p_value < 0.005)))


if __name__ == "__main__":
    """ Example: $ python latex_report_line.py --model_name dp_rubert """
    fire.Fire(model_latex_results)
