""" Makes k copies of the allennlp config for k folds (train_i, dev_i files)
Usage: python utils/make_k_copies.py --filename configs/${METHOD}_0.jsonnet --k 5 """

import fire

def copy(filename, k):
    for i in range(k):
        with open(filename, 'r') as f:
            text = f.read()

        text = text.replace('local foldnum = 0;', f'local foldnum = {i};')
        with open(filename.replace('_0', f'_{i}'), 'w') as f:
            f.write(text)


if __name__ == '__main__':
    fire.Fire(copy)
