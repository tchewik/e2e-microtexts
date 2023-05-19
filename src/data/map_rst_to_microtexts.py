import os
import pickle
import shutil

import fire
from isanlp.annotation_rst import ForestExporter
from tqdm import tqdm

from collect_micro_essays import Doc
from src.data.rst2arguments_mapper import RST2ArgumentsMapper


def collect_text(document: Doc, key='txt_en'):
    return ' '.join([vars(edu).get(key) for edu in document.edus])


def main(lang, input_dir='data/nlp_annot', output_dir='data/rst_shrinked'):

    lang_parts = ['en', 'ru2en'] if lang == 'english' else ['ru', 'en2ru']

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for lang_part in lang_parts:
        current_output = os.path.join(output_dir, lang_part)
        if os.path.exists(current_output):
            shutil.rmtree(current_output)
        os.mkdir(current_output)

    e = ForestExporter('utf8')
    mapper = RST2ArgumentsMapper()
    data = pickle.load(open('data/data.pkl', 'rb'))

    for lang_part in lang_parts:
        for doc in tqdm(data):
            with open(os.path.join(input_dir, lang_part, doc.id + '.pkl'), 'rb') as f:
                trees = pickle.load(f)['rst']

            if trees:
                new_tree = mapper.argtree2rsttree(trees[0], doc, key='txt_' + lang_part, mask_text=True)
                pkl_filename = os.path.join(output_dir, lang_part, doc.id + '.pkl')
                pickle.dump(new_tree, open(pkl_filename, 'wb'))

                rst_filename = os.path.join(output_dir, lang_part, doc.id + '.rs3')
                e([new_tree], rst_filename)


if __name__ == '__main__':
    fire.Fire(main)
