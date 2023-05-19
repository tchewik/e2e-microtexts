import os
import pickle
import shutil

import fire
from isanlp import PipelineCommon
from isanlp.annotation_rst import ForestExporter
from isanlp.processor_razdel import ProcessorRazdel
from isanlp.processor_remote import ProcessorRemote
from tqdm import tqdm

from collect_micro_essays import Doc


def collect_text(document: Doc, key='txt_en'):
    return ' '.join([vars(edu).get(key) for edu in document.edus])


class NLProcessor:
    def __init__(self, servername, syntax_port, rst_port, lang):

        self._servername = servername
        self._syntax_port = syntax_port
        self._rst_port = rst_port
        self.lang = lang

        self._ppl = self._construct_pipeline()

    def __call__(self, document):
        return self._ppl(document)

    def _construct_pipeline(self):
        if self.lang in ['en', 'english']:
            self._ppl = PipelineCommon([
                (ProcessorRemote(self._servername, self._syntax_port, '0'),
                 ['text'],
                 {'tokens': 'tokens',
                  'sentences': 'sentences'}),
                (ProcessorRemote(self._servername, self._rst_port, 'default'),
                 ['text', 'tokens', 'sentences'],
                 {'rst': 'rst'})
            ])

        elif self.lang in ['ru', 'russian']:
            self._ppl = PipelineCommon([
                (ProcessorRazdel(), ['text'],
                 {'tokens': 'tokens',
                  'sentences': 'sentences'}),
                (ProcessorRemote(self._servername, self._syntax_port, '0'),
                 ['tokens', 'sentences'],
                 {'lemma': 'lemma',
                  'morph': 'morph',
                  'syntax_dep_tree': 'syntax_dep_tree',
                  'postag': 'postag'}),
                (ProcessorRemote(self._servername, self._rst_port, 'default'),
                 ['text', 'tokens', 'sentences', 'postag', 'morph', 'lemma', 'syntax_dep_tree'],
                 {'rst': 'rst'})
            ])

        else:
            raise ValueError('Unknown language: {}'.format(self.lang))

        return self._ppl


def main(servername, syntax_port, rst_port, lang, output_path='data/nlp_annot'):
    data = pickle.load(open('data/data.pkl', 'rb'))
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    rst_parser = NLProcessor(servername, syntax_port, rst_port, lang)
    rst_exporter = ForestExporter('utf8')

    lang_parts = ('en', 'ru2en') if lang in ['en', 'english'] else ('ru', 'en2ru')
    for lang_part in lang_parts:
        current_output = os.path.join(output_path, lang_part)
        if os.path.exists(current_output):
            shutil.rmtree(current_output)

        os.mkdir(current_output)

        for document in tqdm(data):
            rst_filename = os.path.join(current_output, document.id + '.rs3')
            txt_en = collect_text(document, 'txt_' + lang_part)
            res = rst_parser(txt_en)
            pickle.dump(res, open(os.path.join(current_output, document.id + '.pkl'), 'wb'))
            rst_exporter(res['rst'], rst_filename)


if __name__ == '__main__':
    fire.Fire(main)
