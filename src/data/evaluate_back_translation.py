import pickle
import string

import fire
import nltk


def main(target_lang='english', data_path='data/data.pkl'):
    key_hand = 'txt_en' if target_lang == 'english' else 'txt_ru'
    key_auto = 'txt_ru2en' if target_lang == 'english' else 'txt_en2ru'
    tokenizer_en = nltk.tokenize.TweetTokenizer(preserve_case=False)

    data = pickle.load(open(data_path, 'rb'))

    references = []
    hypotheses = []
    for doc in data:
        references += [
            [[tok for tok in tokenizer_en.tokenize(vars(edu).get(key_hand)) if not tok in string.punctuation]]
            for edu in doc.edus]
        hypotheses += [[tok for tok in tokenizer_en.tokenize(vars(edu).get(key_auto)) if not tok in string.punctuation]
                       for edu in doc.edus]

    print('BLEU score =', nltk.translate.bleu_score.corpus_bleu(references, hypotheses))


if __name__ == '__main__':
    fire.Fire(main)
