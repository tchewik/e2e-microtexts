import logging
from typing import Dict, Tuple, List

import razdel
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("ud_microtexts", exist_ok=True)
class UDMicrotextsDatasetReader(DatasetReader):
    """
    Reads a microtext dependency file in the conllu-like format.
    # Parameters
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        The token indexers to be applied to the words TextField.
    tokenizer : `Tokenizer`, optional (default = `None`)
        A tokenizer to use to split the text. This is useful when the tokens that you pass
        into the model need to have some particular attribute. Typically it is not necessary.
    custom_tokenizer_type : `str`, optional (default = `None`)
        A custom option to split the text if tokenizer==None. Available option: ''razdel''
    """

    def __init__(
            self,
            token_indexers: Dict[str, TokenIndexer] = None,
            tokenizer: Tokenizer = None,
            custom_tokenizer_type: str = None,
            read_rst_rels: bool = False,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.tokenizer = tokenizer
        self.tokenization_type = custom_tokenizer_type
        self.read_rst_rels = read_rst_rels

        self.custom_tokenization_f = None
        if self.tokenization_type == "razdel":
            self.custom_tokenization_f = lambda text: [token.text for token in razdel.tokenize(text)]

    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as conllu_file:
            logger.info("Reading UD instances from conll-like dataset at: %s", file_path)
            conllu = conllu_file.read().split('\n\n')

        for doc in conllu:
            if self.read_rst_rels:
                all_keys = ['id', 'spans', 'rst_heads', 'rst_rels_labels']
            else:
                all_keys = ['id', 'spans']

            all_keys += ['roles_tags', 'heads', 'functions_labels']  # targets
            annotation = {key: [] for key in all_keys[1:]}
            filename = ''
            for line in doc.strip().split('\n'):
                if line.strip():
                    if line.strip()[0] == '#':
                        filename = line.strip()[6:].strip()
                    else:
                        line_annot = dict(zip(all_keys, [value.strip() for value in line.strip().split('\t')]))
                        for key in annotation.keys():
                            annotation[key].append(line_annot[key])

            if annotation['spans']:
                annotation['dependencies'] = list(zip(annotation['functions_labels'], annotation['heads']))
                del annotation['functions_labels']
                del annotation['heads']

                if self.read_rst_rels:
                    annotation['rst_dependencies'] = list(zip(annotation['rst_rels_labels'], annotation['rst_heads']))
                    del annotation['rst_rels_labels']
                    del annotation['rst_heads']

                annotation['filename'] = filename
                yield self.text_to_instance(**annotation)

    def text_to_instance(
            self,  # type: ignore
            spans: List[str],
            roles_tags: List[str] = None,
            dependencies: List[Tuple[str, str]] = None,
            rst_dependencies: List[Tuple[str, str]] = None,
            filename: str = None,
            *args, **kwargs
    ) -> Instance:

        """
        # Parameters
        words : `List[str]`, required.
            The words in the sentence to be encoded.
        roles : `List[str]`, optional (default = `None`).
            Roles values for each span ('pro' or 'opp').
        dependencies : `List[Tuple[str, int]]`, optional (default = `None`)
            A list of  (function, head index) tuples. Indices are 1 indexed,
            meaning an index of 0 corresponds to that span being the Central Claim (root) of
            the dependency tree.
        # Returns
        An instance containing spans, roles, dependency functions and head
        indices as fields.
        """
        fields: Dict[str, Field] = {}

        if self.tokenizer is not None:
            tokens = [self.tokenizer.tokenize(span) for span in spans]
        else:
            try:
                tokens = [[Token(tok) for tok in self.custom_tokenization_f(span)] for span in spans]
            except:
                raise AttributeError("No valid tokenizer found. Specify the tokenizer for data reading.")

        text_fields = ListField([TextField(span_tokens, self._token_indexers) for span_tokens in tokens])

        fields["spans"] = text_fields

        if rst_dependencies is not None:
            fields['rst_rels_labels'] = SequenceLabelField(
                [x[0] for x in rst_dependencies], text_fields, label_namespace="rst_rels_labels"
            )
            fields['rst_heads'] = SequenceLabelField(
                [int(x[1]) for x in rst_dependencies], text_fields, label_namespace="rst_head_tags"
            )

        if roles_tags is not None:
            fields["roles_tags"] = SequenceLabelField(roles_tags, text_fields, label_namespace="roles_tags")

        if dependencies is not None:
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
            fields["functions_labels"] = SequenceLabelField(
                [x[0] for x in dependencies], text_fields, label_namespace="functions_labels"
            )
            fields["head_indices"] = SequenceLabelField(
                [int(x[1]) for x in dependencies], text_fields, label_namespace="head_indices_tags"
            )

        fields["metadata"] = MetadataField(
            {"filename": filename, "spans": spans, "roles": roles_tags, "rst_dependencies": rst_dependencies,
             "dependencies": dependencies})
        return Instance(fields)
