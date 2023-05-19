import os

import fire
from tqdm import tqdm

from src.data.rst2arguments_mapper import RST2ArgumentsMapper


def convert(input_dir='data/conll_cv', output_dir='data/conll_cv_rst', k=10):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    mapper = RST2ArgumentsMapper()
    for part in ['en', 'ru', 'en_aug', 'ru_aug']:
        dir_output = os.path.join(output_dir, part)
        if not os.path.isdir(dir_output):
            os.mkdir(dir_output)

        for fold in tqdm(range(k)):
            train_annot = open(os.path.join(input_dir, part, f'train_fold_{fold}.conll')).read(
            ).replace('\n\n\n', '\n\n').split('\n\n')
            extended_train_annot = '\n\n'.join(
                [mapper.joined_conll_shrinked(doc_annot, part) for doc_annot in train_annot if doc_annot])
            with open(os.path.join(dir_output, f'train_fold_{fold}.conll'), 'w') as f:
                f.write(extended_train_annot)

            dev_annot = open(os.path.join('data/conll_cv', part, f'dev_fold_{fold}.conll')).read(
            ).replace('\n\n\n', '\n\n').split('\n\n')
            extended_dev_annot = '\n\n'.join(
                [mapper.joined_conll_shrinked(doc_annot, part) for doc_annot in dev_annot if doc_annot])
            with open(os.path.join(dir_output, f'dev_fold_{fold}.conll'), 'w') as f:
                f.write(extended_dev_annot)

            test_annot = open(os.path.join('data/conll_cv', part, f'test_fold_{fold}.conll')).read(
            ).replace('\n\n\n', '\n\n').split('\n\n')
            extended_test_annot = '\n\n'.join(
                [mapper.joined_conll_shrinked(doc_annot, part) for doc_annot in test_annot if doc_annot])
            with open(os.path.join(dir_output, f'test_fold_{fold}.conll'), 'w') as f:
                f.write(extended_test_annot)


if __name__ == '__main__':
    fire.Fire(convert)
    print('Done!')
