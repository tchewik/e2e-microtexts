import os
import pickle
import zipfile
from dataclasses import dataclass

import charset_normalizer
import fire
import tqdm
from bs4 import BeautifulSoup


@dataclass
class EDU:
    """Class for keeping track of an EDU in the annotated dataset."""
    id: str
    txt_en: str
    txt_ru: str
    txt_en2ru: str = None
    txt_ru2en: str = None
    txt_en2ru_google: str = None
    txt_en2ru_yandex: str = None
    txt_en2ru_promt: str = None


@dataclass
class ADU:
    """Class for keeping track of an ADU in the annotation."""
    id: str
    type: str


@dataclass
class Edge:
    """Describes edges between ADUs."""
    id: str
    src: str
    trg: str
    type: str


@dataclass
class Doc:
    id: str
    edus: list
    adus: list
    edges: list


def collect_doc(filename, text=None):
    def sort_by_id(l):
        return sorted(l, key=lambda element: int(element.id[1:]))

    if not text:
        text = open(filename, 'r').read()

    soup = BeautifulSoup(text, features="xml")

    arggraph = soup.find_all('arggraph')[0]
    doc_id = arggraph['id']

    all_edus = soup.find_all('edu')
    collected_edus = []
    for edu_id in set([edu['id'] for edu in all_edus]):
        for entry in soup.find_all(attrs={'id': edu_id}):
            if entry['lang'] == 'en':
                txt_en = entry.text.strip()
            elif entry['lang'] == 'ru':
                txt_ru = entry.text.strip()

        collected_edus.append(EDU(id=edu_id, txt_en=txt_en, txt_ru=txt_ru))

    all_adus = soup.find_all('adu')
    collected_adus = [ADU(id=adu['id'], type=adu['type']) for adu in all_adus]

    all_edges = soup.find_all('edge')
    collected_edges = [Edge(id=edge['id'], type=edge['type'], src=edge['src'], trg=edge['trg']) for edge in all_edges]

    return Doc(id=doc_id,
               edus=sort_by_id(collected_edus),
               adus=sort_by_id(collected_adus),
               edges=sort_by_id(collected_edges))


def collect(input_path='data/corpus', output_path='data/data.pkl'):
    print(f'Collecting the documents from {input_path} to {output_path}...\t')
    result = []
    for file in ['translated_part1_human.zip', 'translated_part2_human.zip']:
        zipfile_path = os.path.join(input_path, file)

        with zipfile.ZipFile(zipfile_path) as zf:
            for filename in tqdm.tqdm(zf.namelist()):
                if filename[-4:] == '.xml':
                    contents = zf.read(filename)

                    encoding = charset_normalizer.detect(contents)['encoding']
                    contents = contents.decode(encoding)

                    result.append(collect_doc(filename, text=contents))

    pickle.dump(result, open(output_path, 'wb'))


if __name__ == '__main__':
    fire.Fire(collect)
