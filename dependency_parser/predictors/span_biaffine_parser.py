from typing import Dict, Any, List, Tuple

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

# POS tags have a unified colour.
NODE_TYPE_TO_STYLE = {}

NODE_TYPE_TO_STYLE["root"] = ["color5", "strong"]
NODE_TYPE_TO_STYLE["dep"] = ["color5", "strong"]

# Support
NODE_TYPE_TO_STYLE["support"] = ["color1"]
NODE_TYPE_TO_STYLE["example"] = ["color1"]
NODE_TYPE_TO_STYLE["link"] = ["color1"]

# Attack
NODE_TYPE_TO_STYLE["rebut"] = ["color2"]
NODE_TYPE_TO_STYLE["undercut"] = ["color2"]

LINK_TO_POSITION = {}
# Put support on the left
LINK_TO_POSITION["support"] = "left"
LINK_TO_POSITION["example"] = "left"
LINK_TO_POSITION["link"] = "left"

# Put attack on the right
LINK_TO_POSITION["rebut"] = "right"
LINK_TO_POSITION["undercut"] = "right"


@Predictor.register("span_biaffine_parser", exist_ok=True)
class SpanBiaffineDependencyParserPredictor(Predictor):
    """
    Predictor for the [`BiaffineDependencyParser`](../models/biaffine_dependency_parser.md) model.
    """

    def __init__(
            self, model: Model, dataset_reader: DatasetReader, language: str = "en_core_web_sm"
    ) -> None:
        super().__init__(model, dataset_reader)
        # TODO(Mark) Make the language configurable and based on a model attribute.
        self._tokenizer = SpacyTokenizer(language=language, pos_tags=True)

    def predict(self, document: List[str]) -> JsonDict:
        """
        Predict a dependency parse for the given spans (discourse units).
        # Parameters
        document The document to parse.
        # Returns
        A dictionary representation of the dependency tree.
        """
        return self.predict_json({"document": document})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"spans": ["...",], "rst_dependencies": ["...",]}`.
        """

        spans = json_dict["spans"]
        rst_dependencies = json_dict.get("rst_dependencies")
        return self._dataset_reader.text_to_instance(spans=spans, rst_dependencies=rst_dependencies)

    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        for output in outputs:
            spans = output["spans"]
            roles = output["predicted_roles"]
            functions = output["predicted_functions"]
            output["hierplane_tree"] = self._build_hierplane_tree(spans, heads, functions, roles)
        return sanitize(outputs)

    @staticmethod
    def _build_hierplane_tree(
            spans: List[str], heads: List[int], functions: List[str], roles: List[str]
    ) -> Dict[str, Any]:
        """
        # Returns
        A JSON dictionary render-able by Hierplane for the given tree.
        """

        span_index_to_cumulative_indices: Dict[int, Tuple[int, int]] = {}
        cumulative_index = 0
        for i, span in enumerate(spans):
            span_length = len(span) + 1
            span_index_to_cumulative_indices[i] = (cumulative_index, cumulative_index + span_length)
            cumulative_index += span_length

        def node_constuctor(index: int):
            children = []
            for next_index, child in enumerate(heads):
                if child == index + 1:
                    children.append(node_constuctor(next_index))

            # These are the icons which show up in the bottom right
            # corner of the node.
            attributes = []
            start, end = span_index_to_cumulative_indices[index]

            hierplane_node = {
                "span": spans[index],
                # The type of the node - all nodes with the same
                # type have a unified colour.
                "nodeType": roles[index],
                # Attributes of the node.
                "attributes": attributes,
                # The link between  the node and it's parent.
                "link": functions[index],
                "spans": [{"start": start, "end": end}],
            }
            if children:
                hierplane_node["children"] = children
            return hierplane_node

        # We are guaranteed that there is a single word pointing to
        # the root index, so we can find it just by searching for 0 in the list.
        root_index = heads.index(0)
        hierplane_tree = {
            "text": " ".join(spans),
            "root": node_constuctor(root_index),
            "nodeTypeToStyle": NODE_TYPE_TO_STYLE,
            "linkToPosition": LINK_TO_POSITION,
        }
        return hierplane_tree
