import copy
import logging
import os
from collections import defaultdict
from typing import Dict, Tuple, Any, List

import numpy
import torch
import torch.nn.functional as F
from allennlp.common.checks import check_dimensions_match
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder, InputVariationalDropout
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.nn import InitializerApplicator
from allennlp.nn.chu_liu_edmonds import decode_mst
from allennlp.nn.util import (
    get_device_of,
    masked_log_softmax,
    get_lengths_from_binary_sequence_mask,
)
from allennlp.nn.util import get_text_field_mask, get_range_vector
from allennlp.training.metrics import AttachmentScores, F1Measure, CategoricalAccuracy
from torch import nn
from torch.nn.modules import Dropout

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

logger = logging.getLogger(__name__)


@Model.register("biaffine_argument_parser")
class BiaffineArgumentParser(Model):
    """
    This dependency parser follows the model of
    [Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)]
    (https://arxiv.org/abs/1611.01734) .
    Spans representations are generated using some encoder,
    followed by separate biaffine classifiers for pairs of spans,
    predicting whether a directed arc exists between the two spans
    and the dependency label the arc should have. Decoding can either
    be done greedily, or the optimal Minimum Spanning Tree can be
    decoded using Edmond's algorithm by viewing the dependency tree as
    a MST on a fully connected graph, where nodes are words and edges
    are scored dependency arcs.
    Roles are predicted jointly with the arc tags (functions).
    # Parameters
    vocab : `Vocabulary`, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : `TextFieldEmbedder`, required
        Used to embed the `tokens` `TextField` we get as input to the model.
    encoder : `Seq2VecEncoder`
        The encoder that we will use to generate representations of spans.
    tag_representation_dim : `int`, required.
        The dimension of the MLPs used for dependency tag prediction.
    arc_representation_dim : `int`, required.
        The dimension of the MLPs used for head arc prediction.
    head_sentinel_encoder : `Seq2VecEncoder`, optional, (default = `None`)
        Used to encode a virtual ROOT node as a combination of all text spans representations.
        By default, initializes randomly.
    tag_feedforward : `FeedForward`, optional, (default = `None`).
        The feedforward network used to produce tag representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    arc_feedforward : `FeedForward`, optional, (default = `None`).
        The feedforward network used to produce arc representations.
        By default, a 1 layer feedforward network with an elu activation is used.  # ToDO
    use_mst_decoding_for_validation : `bool`, optional (default = `True`).
        Whether to use Edmond's algorithm to find the optimal minimum spanning tree during inference.
        If false, decoding is greedy.
    dropout : `float`, optional, (default = `0.0`)
        The variational dropout applied to the output of the encoder and MLP layers.
    input_dropout : `float`, optional, (default = `0.0`)
        The dropout applied to the embedded text input.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    """

    def __init__(
            self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            encoder: Seq2VecEncoder,
            tag_representation_dim: int,
            arc_representation_dim: int,
            head_sentinel_encoder: Seq2VecEncoder = None,
            tag_feedforward: FeedForward = None,
            arc_feedforward: FeedForward = None,
            use_mst_decoding_for_validation: bool = True,
            input_layernorm: bool = False,
            dropout: float = 0.0,
            input_dropout: float = 0.0,
            initializer: InitializerApplicator = InitializerApplicator(),
            **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder

        encoder_dim = encoder.get_output_dim()
        self._input_layernorm = None if not input_layernorm else nn.LayerNorm(encoder_dim)

        # For HEAD prediction #####
        self.head_arc_feedforward = arc_feedforward or FeedForward(
            encoder_dim, 1, arc_representation_dim, torch.nn.ReLU()  # Activation.by_name("elu")()
        )
        self.child_arc_feedforward = copy.deepcopy(self.head_arc_feedforward)

        self.arc_attention = BilinearMatrixAttention(
            arc_representation_dim, arc_representation_dim, use_input_biases=True, activation=torch.nn.ReLU()  # ? ReLU?
        )

        # For FUNCTION prediction #####
        num_functions = self.vocab.get_vocab_size("functions_labels")

        self.head_tag_feedforward = tag_feedforward or FeedForward(
            encoder_dim, 1, tag_representation_dim, torch.nn.ReLU()  # Activation.by_name("elu")()
        )
        self.child_tag_feedforward = copy.deepcopy(self.head_tag_feedforward)

        self.tag_bilinear = torch.nn.modules.Bilinear(
            tag_representation_dim, tag_representation_dim, num_functions
        )

        self._dropout = Dropout(input_dropout)
        self._input_dropout = InputVariationalDropout(dropout)

        self._head_sentinel_encoder = head_sentinel_encoder
        if not head_sentinel_encoder:
            self._head_sentinel = torch.nn.Parameter(torch.randn([1, 1, encoder.get_output_dim()]))

        representation_dim = text_field_embedder.get_output_dim()

        check_dimensions_match(
            representation_dim,
            encoder.get_input_dim(),
            "text field embedding dim",
            "encoder input dim",
        )

        check_dimensions_match(
            tag_representation_dim,
            self.head_tag_feedforward.get_output_dim(),
            "tag representation dim",
            "tag feedforward output dim",
        )

        check_dimensions_match(
            arc_representation_dim,
            self.head_arc_feedforward.get_output_dim(),
            "arc representation dim",
            "arc feedforward output dim",
        )

        self.use_mst_decoding_for_validation = use_mst_decoding_for_validation

        self._cc_idx = self.vocab.get_token_index('cc', namespace='functions_labels')
        self._pro_idx = self.vocab.get_token_index('pro', namespace='roles_tags')
        self._att_idx = self.vocab.get_token_index('att', namespace='functions_labels')

        self._roles_accuracy = CategoricalAccuracy()
        self._roles_f1_pro = F1Measure(self.vocab.get_token_index('pro', namespace='roles_tags'))
        self._roles_f1_opp = F1Measure(self.vocab.get_token_index('opp', namespace='roles_tags'))
        self._attachment_scores = AttachmentScores()
        self._role_las = AttachmentScores()
        initializer(self)

    def forward(
            self,  # type: ignore
            spans: List[TextFieldTensors],
            metadata: List[Dict[str, Any]],
            roles_tags: torch.LongTensor = None,
            functions_labels: torch.LongTensor = None,
            head_indices: torch.LongTensor = None,
            *args, **kwargs
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters
        words : `TextFieldTensors`, required
            The output of `TextField.as_array()`, which should typically be passed directly to a
            `TextFieldEmbedder`. This output is a dictionary mapping keys to `TokenIndexer`
            tensors.  At its most basic, using a `SingleIdTokenIndexer` this is : `{"tokens":
            Tensor(batch_size, sequence_length)}`. This dictionary will have the same keys as were used
            for the `TokenIndexers` when you created the `TextField` representing your
            sequence.  The dictionary is designed to be passed directly to a `TextFieldEmbedder`,
            which knows how to combine different word representations into a single vector per
            token in your input.
        pos_tags : `torch.LongTensor`, required
            The output of a `SequenceLabelField` containing POS tags.
            POS tags are required regardless of whether they are used in the model,
            because they are used to filter the evaluation metric to only consider
            heads of words which are not punctuation.
        metadata : `List[Dict[str, Any]]`, optional (default=`None`)
            A dictionary of metadata for each batch element which has keys:
                words : `List[str]`, required.
                    The tokens in the original sentence.
                pos : `List[str]`, required.
                    The dependencies POS tags for each word.
        functions_labels : `torch.LongTensor`, optional (default = `None`)
            A torch tensor representing the sequence of integer gold class labels for the arcs
            in the dependency parse. Has shape `(batch_size, sequence_length)`.
        head_indices : `torch.LongTensor`, optional (default = `None`)
            A torch tensor representing the sequence of integer indices denoting the parent of every
            word in the dependency parse. Has shape `(batch_size, sequence_length)`.
        # Returns
        An output dictionary consisting of:
        loss : `torch.FloatTensor`, optional
            A scalar loss to be optimised.
        arc_loss : `torch.FloatTensor`
            The loss contribution from the unlabeled arcs.
        loss : `torch.FloatTensor`, optional
            The loss contribution from predicting the dependency
            tags for the gold arcs.
        heads : `torch.FloatTensor`
            The predicted head indices for each word. A tensor
            of shape (batch_size, sequence_length).
        head_types : `torch.FloatTensor`
            The predicted head types for each arc. A tensor
            of shape (batch_size, sequence_length).
        mask : `torch.BoolTensor`
            A mask denoting the padded elements in the batch.
        """

        embedded_text_input = self.text_field_embedder(spans, num_wrapping_dims=1)
        mask = get_text_field_mask(spans)
        text_encoded, text_mask = self._encode_multiple_spans(self.encoder, embedded_text_input, mask)

        if self._input_layernorm:
            text_encoded = self._input_layernorm(text_encoded)
        text_encoded = self._input_dropout(text_encoded)

        predicted_heads, predicted_functions_labels, augmented_mask, arc_nll, tag_nll = self._parse(
            text_encoded, text_mask, functions_labels, head_indices)

        loss = arc_nll + tag_nll

        if head_indices is not None and functions_labels is not None and roles_tags is not None:
            evaluation_mask = augmented_mask.detach()[:, 1:]

            # We calculate attachment scores for the whole sentence
            # but excluding the symbolic ROOT token at the start,
            # which is why we start from the second element in the sequence.
            self._attachment_scores(
                predicted_heads[:, 1:],
                predicted_functions_labels[:, 1:],
                head_indices,
                functions_labels,
                evaluation_mask,
            )

        if self.training or not self.use_mst_decoding_for_validation:
            predicted_roles = []
        else:
            predicted_roles = [self._decode_roles(he, fu) for he, fu in
                               zip(predicted_heads[:, 1:].cpu().detach().numpy(),
                                   predicted_functions_labels[:,
                                   1:].cpu().detach().numpy())]

        output_dict = {
            "predicted_roles": predicted_roles,
            "predicted_heads": predicted_heads,
            "predicted_functions": predicted_functions_labels,
            "arc_loss": arc_nll,
            "function_loss": tag_nll,
            "loss": loss,
            "mask": augmented_mask,
            "spans": [meta["spans"] for meta in metadata],
            "roles": [meta["roles"] for meta in metadata],
            "dependencies": [meta["dependencies"] for meta in metadata],
        }

        return output_dict
        

    @staticmethod
    def bfs(adjacency_dict, vertex):
        """ Implements breadth-first search for a graph """

        visited_set = set()
        queue = []
        visited_set.add(vertex)
        queue.append(vertex)
        result = []
        while queue:
            v = queue[0]
            result.append(v)
            queue = queue[1:]
            for neighbor in adjacency_dict[v]:
                if neighbor not in visited_set:
                    visited_set.add(neighbor)
                    queue.append(neighbor)
        return result

    def _decode_roles(self, heads, functions):
        adj_dict = defaultdict(list)
        for node, head in zip(numpy.arange(len(heads)) + 1, heads):
            adj_dict[head].append(node)
        root = adj_dict[0]
        if type(root) == list:
            # multiple roots, happens with greedy decoding
            path = []
            for r in root:
                path += self.bfs(adj_dict, r)
        else:
            path = self.bfs(adj_dict, adj_dict[0])

        roles = [True] * len(functions)
        for i, node in enumerate(path):
            if i > 0 and functions[node - 1] == self._att_idx:
                roles[node - 1] = not roles[path[i - 1] - 1]

        m_roles = {True: 'pro', False: 'opp'}
        return [m_roles[role] for role in roles]

    def make_output_human_readable(
            self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        roles = output_dict.pop("predicted_roles")  # .cpu().detach().numpy()
        functions = output_dict.pop("predicted_functions").cpu().detach().numpy()

        heads = output_dict.pop("predicted_heads").cpu().detach().numpy()
        mask = output_dict.pop("mask")
        lengths = get_lengths_from_binary_sequence_mask(mask)

        role_labels = []
        # role_probas = []
        head_tag_labels = []
        head_indices = []
        for instance_roles, instance_heads, instance_functions, length in zip(roles, heads, functions, lengths):
            # role_labels.append(
            #     [self.vocab.get_token_from_index(role, "roles_tags") for role in instance_roles[1:length]])
            role_labels.append(instance_roles)

            instance_heads = list(instance_heads[1:length])
            instance_functions = instance_functions[1:length]
            labels = [self.vocab.get_token_from_index(label, "functions_labels") for label in instance_functions]
            head_tag_labels.append(labels)
            head_indices.append(instance_heads)

        output_dict["predicted_roles"] = role_labels
        output_dict["predicted_functions"] = head_tag_labels
        output_dict["predicted_heads"] = head_indices
        return output_dict

    @staticmethod
    def _encode_multiple_spans(encoder, embeddings, mask):
        """ Use to encode tensors of shape (num_batch, num_spans, num_words, embedding_size)
            with some Seq2Vec encoder in order to reach shape (num_batch, num_spans, embedding_size) """

        batch_size = mask.shape[0]
        encodings = [encoder(embeddings[batch], mask[batch]) for batch in range(batch_size)]
        encoded_text = torch.stack(encodings)
        encoded_mask = mask.sum(2).bool()

        return encoded_text, encoded_mask

    def _parse(
            self,
            encoded_text: torch.Tensor,
            mask: torch.BoolTensor,
            head_tags: torch.LongTensor = None,
            head_indices: torch.LongTensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size, _, encoding_dim = encoded_text.size()

        # Encode dummy
        if self._head_sentinel_encoder:
            head_sentinel = self._head_sentinel_encoder(encoded_text, mask).unsqueeze(1)
        else:
            head_sentinel = self._head_sentinel.expand(batch_size, 1, encoding_dim)

        encoded_text = torch.cat([head_sentinel, encoded_text], 1)

        mask = torch.cat([mask.new_ones(batch_size, 1), mask], 1)
        if head_indices is not None:
            head_indices = torch.cat([head_indices.new_zeros(batch_size, 1), head_indices], 1)
        if head_tags is not None:
            head_tags = torch.cat([head_tags.new_zeros(batch_size, 1), head_tags], 1)

        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag_representation = self._dropout(self.head_tag_feedforward(encoded_text))
        child_tag_representation = self._dropout(self.child_tag_feedforward(encoded_text))

        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc_representation = self._dropout(self.head_arc_feedforward(encoded_text))
        child_arc_representation = self._dropout(self.child_arc_feedforward(encoded_text))

        # shape (batch_size, sequence_length, sequence_length)
        attended_arcs = self.arc_attention(head_arc_representation, child_arc_representation)

        minus_inf = -1e8
        minus_mask = ~mask * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        if self.training or not self.use_mst_decoding_for_validation:
            predicted_heads, predicted_head_tags = self._greedy_decode(
                head_tag_representation, child_tag_representation,
                attended_arcs, mask
            )
        else:
            predicted_heads, predicted_head_tags = self._mst_decode(
                head_tag_representation, child_tag_representation,
                attended_arcs, mask
            )

        if head_indices is not None and head_tags is not None:
            arc_nll, tag_nll = self._construct_loss(
                head_tag_representation=head_tag_representation,
                child_tag_representation=child_tag_representation,
                attended_arcs=attended_arcs,
                head_indices=head_indices,
                head_tags=head_tags,
                mask=mask,
            )

        else:
            arc_nll, tag_nll = self._construct_loss(
                head_tag_representation=head_tag_representation,
                child_tag_representation=child_tag_representation,
                attended_arcs=attended_arcs,
                head_indices=predicted_heads.long(),
                head_tags=predicted_head_tags.long(),
                mask=mask,
            )

        return predicted_heads, predicted_head_tags, mask, arc_nll, tag_nll

    def _construct_loss(
            self,
            head_tag_representation: torch.Tensor,
            child_tag_representation: torch.Tensor,
            attended_arcs: torch.Tensor,
            head_indices: torch.Tensor,
            head_tags: torch.Tensor,
            mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the arc and tag loss for a sequence given gold head indices and tags.
        # Parameters
        head_tag_representation : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.
        head_indices : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length).
            The indices of the heads for every word.
        head_tags : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length).
            The dependency labels of the heads for every word.
        mask : `torch.BoolTensor`, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.
        # Returns
        arc_nll : `torch.Tensor`, required.
            The negative log likelihood from the arc loss.
        tag_nll : `torch.Tensor`, required.
            The negative log likelihood from the arc tag loss.
        """

        batch_size, sequence_length, _ = attended_arcs.size()
        # shape (batch_size, 1)
        range_vector = get_range_vector(batch_size, get_device_of(attended_arcs)).unsqueeze(1)
        # shape (batch_size, sequence_length, sequence_length)
        normalised_arc_logits = (
                masked_log_softmax(attended_arcs, mask) * mask.unsqueeze(2) * mask.unsqueeze(1)
        )

        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(
            head_tag_representation, child_tag_representation, head_indices
        )
        normalised_head_tag_logits = masked_log_softmax(
            head_tag_logits, mask.unsqueeze(-1)
        ) * mask.unsqueeze(-1)

        timestep_index = get_range_vector(sequence_length, get_device_of(attended_arcs))
        child_index = (
            timestep_index.view(1, sequence_length).expand(batch_size, sequence_length).long()
        )
        # shape (batch_size, sequence_length)
        arc_loss = normalised_arc_logits[range_vector, child_index, head_indices]
        tag_loss = normalised_head_tag_logits[range_vector, child_index, head_tags]
        # We don't care about predictions for the symbolic ROOT token's head,
        # so we remove it from the loss.
        arc_loss = arc_loss[:, 1:]
        tag_loss = tag_loss[:, 1:]

        # The number of valid positions is equal to the number of unmasked elements minus
        # 1 per sequence in the batch, to account for the symbolic HEAD token.
        valid_positions = mask.sum() - batch_size

        arc_nll = -arc_loss.sum() / valid_positions.float()
        tag_nll = -tag_loss.sum() / valid_positions.float()
        return arc_nll, tag_nll

    def _greedy_decode(
            self,
            head_tag_representation: torch.Tensor,
            child_tag_representation: torch.Tensor,
            attended_arcs: torch.Tensor,
            mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions by decoding the unlabeled arcs
        independently for each word and then again, predicting the head tags of
        these greedily chosen arcs independently. Note that this method of decoding
        is not guaranteed to produce trees (i.e. there maybe be multiple roots,
        or cycles when children are attached to their parents).
        # Parameters
        head_tag_representation : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.
        # Returns
        heads : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the greedily decoded heads of each word.
        """
        # Mask the diagonal, because the head of a word can't be itself.
        attended_arcs = attended_arcs + torch.diag(
            attended_arcs.new(mask.size(1)).fill_(-numpy.inf)
        )
        # Mask padded tokens, because we only want to consider actual words as heads.
        if mask is not None:
            minus_mask = ~mask.unsqueeze(2)
            attended_arcs.masked_fill_(minus_mask, -numpy.inf)

        # Compute the heads greedily.
        # shape (batch_size, sequence_length)
        _, heads = attended_arcs.max(dim=2)

        if not self.training:
            # In case there is no root predicted (yes, happens)
            # or there are multiple roots predicted (yes, happens)
            for i in range(len(heads)):
                if torch.bincount(heads[i])[0] > 1 or 0 not in heads[i][1:]:
                    heads_i = [head for head in heads[i][1:].cpu().numpy() if head]
                    if heads_i:
                        rootlike = torch.mode(torch.tensor(heads_i)).values.item()
                        for j in range(len(heads[i])):
                            if j > 0:
                                if heads[i][j].item() == 0:
                                    heads[i][j] = rootlike
                                if j == rootlike:
                                    heads[i][j] = 0

        # Given the greedily predicted heads, decode their dependency tags.
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(
            head_tag_representation, child_tag_representation, heads
        )
        _, head_tags = head_tag_logits.max(dim=2)

        return heads, head_tags

    def _mst_decode(
            self,
            head_tag_representation: torch.Tensor,
            child_tag_representation: torch.Tensor,
            attended_arcs: torch.Tensor,
            mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions using the Edmonds' Algorithm
        for finding minimum spanning trees on directed graphs. Nodes in the
        graph are the words in the sentence, and between each pair of nodes,
        there is an edge in each direction, where the weight of the edge corresponds
        to the most likely dependency label probability for that arc. The MST is
        then generated from this directed graph.
        # Parameters
        head_tag_representation : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.
        # Returns
        heads : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the optimally decoded heads of each word.
        """
        batch_size, sequence_length, tag_representation_dim = head_tag_representation.size()

        lengths = mask.data.sum(dim=1).long().cpu().numpy()

        expanded_shape = [batch_size, sequence_length, sequence_length, tag_representation_dim]
        head_tag_representation = head_tag_representation.unsqueeze(2)
        head_tag_representation = head_tag_representation.expand(*expanded_shape).contiguous()
        child_tag_representation = child_tag_representation.unsqueeze(1)
        child_tag_representation = child_tag_representation.expand(*expanded_shape).contiguous()
        # Shape (batch_size, sequence_length (current), sequence_length (head), num_head_tags)
        pairwise_head_logits = self.tag_bilinear(head_tag_representation, child_tag_representation)
        # restrain the root function: always cc
        pairwise_head_logits[:, :, 0] = 0.
        pairwise_head_logits[:, :, 0, self._cc_idx] = 1.  # 100% root being the central claim

        # Note that this log_softmax is over the tag dimension, and we don't consider pairs
        # of tags which are invalid (e.g are a pair which includes a padded element) anyway below.
        # Shape (batch, num_functions,sequence_length, sequence_length)
        normalized_pairwise_head_logits = F.log_softmax(pairwise_head_logits, dim=3).permute(
            0, 3, 1, 2
        )

        # Mask padded tokens, because we only want to consider actual words as heads.
        minus_inf = -1e8
        minus_mask = ~mask * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        # Shape (batch_size, sequence_length, sequence_length)
        normalized_arc_logits = F.log_softmax(attended_arcs, dim=2).transpose(1, 2)

        # Shape (batch_size, num_head_tags, sequence_length, sequence_length)
        # This energy tensor expresses the following relation:
        # energy[i,j] = "Score that i is the head of j". In this
        # case, we have heads pointing to their children.
        batch_energy_functions = torch.exp(normalized_arc_logits.unsqueeze(1) + normalized_pairwise_head_logits)
        return self._run_mst_decoding(batch_energy_functions, lengths)

    @staticmethod
    def _run_mst_decoding(
            batch_energy_functions: torch.Tensor,
            lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        head_tags = []
        for energy_func, length in zip(batch_energy_functions.detach().cpu(),
                                       lengths):
            scores, tag_ids = energy_func.max(dim=0)
            # _, role_ids = energy_role.max(dim=0)
            # Although we need to include the root node so that the MST includes it,
            # we do not want any span to be the parent of the root node.
            # Here, we enforce this by setting the scores for all span -> ROOT
            # edges to be 0.
            scores[0, :] = 0
            # Decode the heads. Because we modify the scores to prevent
            # adding in word -> ROOT edges, we need to find the labels ourselves.
            instance_heads, _ = decode_mst(scores.numpy(), length, has_labels=False)

            # Find the labels (functions) which correspond to the edges in the max spanning tree.
            instance_head_tags = []
            for child, parent in enumerate(instance_heads):
                instance_head_tags.append(tag_ids[parent, child].item())

            # We don't care what the head or tag is for the root token, but by default it's
            # not necessarily the same in the batched vs unbatched case, which is annoying.
            # Here we'll just set them to zero.
            instance_heads[0] = 0
            instance_head_tags[0] = 0

            heads.append(instance_heads)
            head_tags.append(instance_head_tags)

        return (
            torch.from_numpy(numpy.stack(heads)).to(batch_energy_functions.device),
            torch.from_numpy(numpy.stack(head_tags)).to(batch_energy_functions.device),
        )

    def _get_head_tags(
            self,
            head_tag_representation: torch.Tensor,
            child_tag_representation: torch.Tensor,
            head_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decodes the head tags given the head and child tag representations
        and a tensor of head indices to compute tags for. Note that these are
        either gold or predicted heads, depending on whether this function is
        being called to compute the loss, or if it's being called during inference.

        # Parameters

        head_tag_representation : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        head_indices : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length). The indices of the heads
            for every span.

        # Returns

        head_tag_logits : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length, num_head_tags),
            representing logits for predicting a distribution over tags
            for each arc.
        """
        batch_size = head_tag_representation.size(0)
        range_vector = get_range_vector(
            batch_size, get_device_of(head_tag_representation)
        ).unsqueeze(1)

        # This next statement is quite a complex piece of indexing, which you really
        # need to read the docs to understand. See here:
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing
        # In effect, we are selecting the indices corresponding to the heads of each word from the
        # sequence length dimension for each element in the batch.

        # shape (batch_size, sequence_length, tag_representation_dim)
        selected_head_tag_representations = head_tag_representation[range_vector, head_indices]
        selected_head_tag_representations = selected_head_tag_representations.contiguous()

        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self.tag_bilinear(
            selected_head_tag_representations, child_tag_representation
        )

        return head_tag_logits

    def _get_head_roles(
            self,
            head_role_representation: torch.Tensor,
            child_role_representation: torch.Tensor,
            head_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decodes the head roles given the head and child tag representations
        and a tensor of head indices to compute tags for. Note that these are
        either gold or predicted heads, depending on whether this function is
        being called to compute the loss, or if it's being called during inference.

        # Parameters

        head_role_representation : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_role_representation : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        head_indices : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length). The indices of the heads
            for every DU.

        # Returns

        head_role_logits : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length, num_head_tags),
            representing logits for predicting a distribution over roles
            for each arc.
        """
        batch_size = head_role_representation.size(0)
        range_vector = get_range_vector(
            batch_size, get_device_of(head_role_representation)
        ).unsqueeze(1)

        # This next statement is quite a complex piece of indexing, which you really
        # need to read the docs to understand. See here:
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing
        # In effect, we are selecting the indices corresponding to the heads of each word from the
        # sequence length dimension for each element in the batch.

        # shape (batch_size, sequence_length, tag_representation_dim)
        selected_head_role_representations = head_role_representation[range_vector, head_indices]
        selected_head_role_representations = selected_head_role_representations.contiguous()

        # shape (batch_size, sequence_length, num_head_tags)
        head_role_logits = self.role_bilinear(
            selected_head_role_representations, child_role_representation
        )

        return head_role_logits

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        result = self._attachment_scores.get_metric(reset)
        # result['roles_las'] = self._role_las.get_metric(reset)['LAS']

        # result['roles_acc'] = self._roles_accuracy.get_metric(reset)
        # result['roles_f1:pro'] = self._roles_f1_pro.get_metric(reset)['f1']
        # result['roles_f1:opp'] = self._roles_f1_opp.get_metric(reset)['f1']
        # result['mean_las'] = numpy.mean([result['LAS'], result['roles_las']*result['roles_f1:opp']])
        # result['avg'] = numpy.mean([result['LAS'], result['roles_f1:pro'], result['roles_f1:opp']])
        return result

    default_predictor = "span_biaffine_parser"
