import os
import pickle

import fire
from isanlp import PipelineCommon
from isanlp.annotation import Token, Sentence
from isanlp.processor_remote import ProcessorRemote
from tqdm import tqdm

from src.data.collect_micro_essays import Doc, EDU, ADU, Edge


def collect_text(document: Doc, key='txt_en'):
    return ' '.join([vars(edu).get(key) for edu in document.edus])


# Instead of true sentence segmentation, we'll use the MicroEssays' ADUs annotated in the data,
# as there can be multiple sentences in a single ADU
class ArgumentEDUSegmenter:
    def __init__(self, lang='txt_en'):
        self.lang = lang

    def __call__(self, document):
        tokens, sentences = [], []
        begin = 0
        for idx, edu in enumerate(document.edus):
            text = edu.txt_en if self.lang == 'txt_en' else edu.txt_ru
            tokens += [Token(text, begin=begin, end=begin + len(text))]
            begin += len(text) + 1
            sentences += [Sentence(idx, idx + 1)]
        return {'text': collect_text(document, self.lang), 'tokens': tokens, 'sentences': sentences}


class BackTranslator:
    def __init__(self, servername, port):
        self._pipelines = {
            lang: PipelineCommon([
                (ArgumentEDUSegmenter('txt_' + lang), ['document'],
                 {'text': 'text',
                  'tokens': 'tokens',
                  'sentences': 'sentences'}),
                (ProcessorRemote(servername, port, 'default'),
                 ['text', 'tokens', 'sentences'],
                 {'text_translated': 'text_translated'})
            ])
            for lang in ['en', 'ru']
        }

    def translate(self, document, source='en'):
        result = self._pipelines[source](document)
        return result['text_translated']


def collect_translations(document, translations):
    assert document.id == translations['id']

    for i in range(len(document.edus)):
        txt_en2ru = translations['translation']['en2ru'][i]
        txt_ru2en = translations['translation']['ru2en'][i]

        # Remove sentence ending added by the neural translator #####

        if txt_en2ru.endswith('.') and len([edu for edu in [document.edus[i].txt_ru,
                                                            document.edus[i].txt_en,
                                                            document.edus[i].txt_ru2en] if
                                            edu and edu.strip().endswith('.')]) < 2:

            while txt_en2ru.endswith('.'):
                txt_en2ru = txt_en2ru[:-1]

        if txt_ru2en.endswith('.') and len([edu for edu in [document.edus[i].txt_ru,
                                                            document.edus[i].txt_en,
                                                            document.edus[i].txt_en2ru] if
                                            edu and edu.strip().endswith('.')]) < 2:

            while txt_ru2en.endswith('.'):
                txt_ru2en = txt_ru2en[:-1]

        # If there is no sentence separation in the previous EDU, there is no need in uppercase #####

        if i > 0:
            if document.edus[i - 1].txt_en2ru[-1] not in ".?!":
                txt_en2ru = txt_en2ru[0].lower() + txt_en2ru[1:]

            if document.edus[i - 1].txt_ru2en[-1] not in ".?!":
                txt_ru2en = txt_ru2en[0].lower() + txt_ru2en[1:]

        document.edus[i].txt_en2ru = txt_en2ru
        document.edus[i].txt_ru2en = txt_ru2en

    return document


def main(servername, port=3332, data_path='data/data.pkl'):
    # Translation #####
    translator = BackTranslator(servername, port)
    data = pickle.load(open(data_path, 'rb'))

    translated_data = []
    for document in tqdm(data):
        entry = {'id': document.id}
        sentences_ru = translator.translate(document, source='en')
        sentences_en = translator.translate(document, source='ru')
        entry['translation'] = {'en2ru': sentences_ru,
                                'ru2en': sentences_en}
        translated_data.append(entry)

    pickle.dump(translated_data, open(os.path.join('data', 'translations_beam4.pkl'), 'wb'))

    # Collect translations #####

    data = pickle.load(open(data_path, 'rb'))
    assert len(data) == 283

    translations = pickle.load(open('data/translations_beam4.pkl', 'rb'))
    assert len(translations) == 283

    updated_data = []
    for doc, transl in tqdm(zip(data, translations), total=len(data)):
        updated_data.append(collect_translations(doc, transl))

    pickle.dump(updated_data, open(data_path, 'wb'))


if __name__ == '__main__':
    fire.Fire(main)
