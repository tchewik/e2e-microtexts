import os
from typing import List

import fire
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm


def rst_weights_plot(model_name, k, forward_arcs=True, backward_arcs=True, plot_width=8, plot_height=-1, xlim=-1):
    def real_coeffs(theta: torch.tensor, beta: torch.tensor, relations) -> pd.Series:
        """ What are the real coefficients c_ij for specific RST relations in the final model. """
        weights_dict = dict(zip(relations, torch.nn.functional.relu(theta + beta) / beta))
        return pd.Series(weights_dict).map(lambda tensor: tensor.item()).sort_values()

    def weights_stats(weights: List[pd.Series], backward=False) -> list:
        if not weights: return []
        w_stats = pd.DataFrame(weights).agg([np.mean, np.std])
        w_stats.columns = [column[0].upper() + column[1:] + '⁻¹' * backward for column in w_stats.columns]
        if 'Root⁻¹' in w_stats.columns: del w_stats['Root⁻¹']
        return [w_stats]

    pos_res = []
    neg_res = []
    for fold in tqdm(range(k), desc=f'Processing folds'):
        weights = torch.load(f'{model_name}/fold_{fold}/best.th', map_location='cpu')
        relations = [rel.strip() for rel in
                     open(f'{model_name}/fold_{fold}/vocabulary/rst_rels_labels.txt', 'r').readlines()]

        if forward_arcs:
            theta, beta = weights['ff_rst_rels_arc.weight'][0], weights['ff_rst_rels_arc.bias'][0]
            pos_res.append(real_coeffs(theta, beta, relations))

        if backward_arcs:
            theta, beta = weights['ff_rst_rels_arc_back.weight'][0], weights['ff_rst_rels_arc_back.bias'][0]
            neg_res.append(real_coeffs(theta, beta, relations))

    all_rels = pd.concat(weights_stats(pos_res) + weights_stats(neg_res, backward=True), axis=1)
    std = all_rels.T.max()['std']
    std = 0. if np.isnan(std) else std  # If k==1, there will be nan
    xlim = xlim if xlim != -1 else all_rels.T.max()[
                                       'mean'] + std  # right limit for X; enforce or compute from the statistics (-1)
    plot_height = plot_height if plot_height != -1 else (all_rels.shape[
                                                             1] + 1) * 0.2  # plot height; enforce or compute from the number of relations (-1)
    ax = all_rels.T.sort_values('mean').plot(kind='barh', y='mean', legend=False, xerr='std', logx=False,
                                             color='#81AD75',
                                             xlim=(0, xlim),
                                             figsize=(plot_width, plot_height))
    ax.axvline(x=1, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Weight')
    ax.set_ylabel('RST relation')
    # ax.set_title(f'Weights for {model_name}')
    plt.savefig(f"{model_name}/rst_coeff_{os.path.basename(model_name)}.pdf", format="pdf", bbox_inches="tight")
    return all_rels.T.sort_values('mean')


if __name__ == '__main__':
    fire.Fire(rst_weights_plot)
