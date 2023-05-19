import os
import pickle

import fire
from tqdm import tqdm

from src.data.rst2arguments_mapper import RST2ArgumentsMapper


def make_annot_from_filenames(part, fold, lang, output_dir):
    annots = open(os.path.join('data/conll_cv', lang, f'{part}_fold_{fold}.conll')).read(
        ).replace('\n\n\n', '\n\n').split('\n\n')

    mapper = RST2ArgumentsMapper()
    with open(os.path.join(output_dir, lang, f'{part}_fold_{fold}.conll'), 'w') as f:
        for arg_annot in annots:
            if arg_annot:
                filename = arg_annot.split('\n')[0][7:]
                if '(machine translation)' in filename:
                    _filename = filename.split(' (machine translation)')[0]
                    _part = 'ru2en' if lang == 'en_aug' else 'en2ru'
                    trees = pickle.load(open(f'data/nlp_annot/{_part}/{_filename}.pkl', 'rb'))['rst']
                else:
                    _part = lang.replace('_aug', '')
                    trees = pickle.load(open(f'data/nlp_annot/{_part}/{filename}.pkl', 'rb'))['rst']

                arg_conll = '\n'.join(arg_annot.split('\n')[1:])
                rst_conll = mapper.convert_rst2conll(trees)
                try:
                    joined_conll = mapper.joined_conll(rst_conll, arg_conll)
                except Exception as e:
                    print(e)
                    print(filename, lang)

                f.write(f'# id = {filename}\n')
                f.write(joined_conll)
                f.write('\n\n')


def convert(output_dir='data/conll_cv_rst_end2end', k=10):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for lang in ['en', 'ru', 'en_aug', 'ru_aug']:
        dir_output = os.path.join(output_dir, lang)
        if not os.path.isdir(dir_output):
            os.mkdir(dir_output)

        for fold in tqdm(range(k)):
            make_annot_from_filenames('train', fold, lang, output_dir)
            make_annot_from_filenames('dev', fold, lang, output_dir)
            make_annot_from_filenames('test', fold, lang, output_dir)


if __name__ == '__main__':
    fire.Fire(convert)
    print('Done!')
