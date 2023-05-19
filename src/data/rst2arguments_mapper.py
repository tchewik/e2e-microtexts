import copy
import os
import pickle

import networkx as nx


class RST2ArgumentsMapper:
    def argtree2rsttree(self, rsttree, argtree, key='txt_ru', mask_text=True):
        """ Returns RST tree with presented ADUs as terminals.
            Use mask_text=False to preserve the original EDU texts. """

        rsttree = copy.deepcopy(rsttree)
        adus = [vars(adu).get(key) for adu in argtree.edus]

        if rsttree.text.strip() in adus:
            # EDU == ADU[i]

            if mask_text:
                rsttree.text = 'ADU' + str(adus.index(rsttree.text.strip()))

            if rsttree.relation == 'elementary':
                return rsttree

            rsttree.relation, rsttree.nuclearity = 'elementary', '_'
            rsttree.left, rsttree.right = None, None

        elif rsttree.relation == 'elementary':
            inside_adu = [rsttree.text.strip() in adu for adu in adus]
            if any(inside_adu):
                # EDU is inside ADU[i]
                if mask_text:
                    rsttree.text = 'ADU' + str(inside_adu.index(True))

            extends_adu = [adu.strip() in rsttree.text.strip() for adu in adus]
            if any(extends_adu):
                # EDU extends ADU[i]
                if mask_text:
                    numbers = ' '.join(
                        ['ADU' + str(i) for i, x in enumerate(adus) if x.strip() in rsttree.text.strip()])
                    rsttree.text = numbers

        else:
            rsttree.left = self.argtree2rsttree(rsttree.left, argtree, key, mask_text)
            rsttree.right = self.argtree2rsttree(rsttree.right, argtree, key, mask_text)

        return rsttree

    def convert_rst2dep(self, tree):
        """ Converts RST tree to dependency tree."""

        def get_nucleus_edu(node):
            if node.relation == 'elementary':
                return node
            elif node.nuclearity[0] == 'N':
                return get_nucleus_edu(node.left)
            else:
                return get_nucleus_edu(node.right)

        if tree.relation == 'elementary':
            return []

        if tree.nuclearity == 'SN':
            child, parent = get_nucleus_edu(tree.left).id, get_nucleus_edu(tree.right).id
        else:
            child, parent = get_nucleus_edu(tree.right).id, get_nucleus_edu(tree.left).id

        return [(child, parent, tree.relation)] + self.convert_rst2dep(tree.left) + self.convert_rst2dep(tree.right)

    def convert_rst2conll(self, trees):
        """ Converts RST trees of a document to conll format.
        >> tree = pickle.load(open('data/nlp_annot/en/micro_c121.pkl', 'rb'))
        >> mapper.convert_rst2dep(tree)
        << 1	Nuclear energy has been proven time and time again	2	Attribution
        << 2	to be safe.	0	Root
        << 3	It has been proven, over six decades,	2	Elaborate
        << 4	to have a minimal impact on the environment,	3	Same-Unit
        << 5	as they emit almost no greenhouse gases.	4	Background
        << 6	The chances of anything untoward happening to a nuclear facility are also slim.	2	Elaborate
        << 7	There have been a few nuclear events since Chernobyl,	8	Contrast
        << 8	but there have been relatively few fatalities	2	Elaborate
        << 9	associated with it.	8	Elaborate
        """

        def get_edus(node):
            if node.relation == 'elementary':
                return [(node.id, node.text)]
            else:
                return get_edus(node.left) + get_edus(node.right)

        def append_edu_root(edus, conll_tree):
            """ Given a list of EDUs and a conll tree, appends the Root EDU. """

            result = []
            j = 0
            for i in range(len(edus)):
                if j >= len(conll_tree):
                    # The last EDU is the Root EDU
                    result.append([i + 1, edus[i], 0, 'Root'])
                    return result

                if conll_tree[j][0] == i + 1:
                    edu = edus[i]
                    result.append(conll_tree[j])
                    j += 1
                else:
                    result.append([i + 1, edus[i], 0, 'Root'])

            return result

        dependencies = []
        edus = []
        for tree in trees:
            dependencies += self.convert_rst2dep(tree)
            edus += get_edus(tree)

        edus = dict(edus)
        renumbering_dict = {edu: i + 1 for i, edu in enumerate(edus.keys())}
        conll = []
        for i, dep in enumerate(dependencies):
            conll.append([renumbering_dict[dep[0]], edus[dep[0]], renumbering_dict[dep[1]], dep[2]])

        conll = sorted(conll, key=lambda x: x[0])
        return '\n'.join(['\t'.join(map(str, row)) for row in append_edu_root(list(edus.values()), conll)])

    def convert_shrinkedrst2dep(self, tree):
        """ Converts ADU-annotated RST tree to dependency tree.
        >> tree = pickle.load(open('data/rst_shrinked/en/micro_c121.pkl', 'rb'))
        >> mapper.convert_shrinkedrst2dep(tree)
        << [('ADU5', 'ADU0', 'Elaborate'),
            ('ADU3', 'ADU0', 'Elaborate'),
            ('ADU1', 'ADU0', 'Elaborate'),
            ('ADU1', 'ADU1', 'Same-Unit'),
            ('ADU2', 'ADU1', 'Background'),
            ('ADU4', 'ADU5', 'Contrast')]
        """

        def get_nucleus_edu(node):
            if node.relation == 'elementary':
                return node
            elif node.nuclearity[0] == 'N':
                return get_nucleus_edu(node.left)
            else:
                return get_nucleus_edu(node.right)

        if tree.relation == 'elementary':
            return []

        if tree.nuclearity == 'SN':
            child, parent = get_nucleus_edu(tree.left).text.strip(), get_nucleus_edu(tree.right).text.strip()
        else:
            child, parent = get_nucleus_edu(tree.right).text.strip(), get_nucleus_edu(tree.left).text.strip()

        if parent[:3] != 'ADU' or child[:3] != 'ADU':
            return self.convert_shrinkedrst2dep(tree.left) + self.convert_shrinkedrst2dep(tree.right)

        return [(child, parent, tree.relation)] + self.convert_shrinkedrst2dep(
            tree.left) + self.convert_shrinkedrst2dep(tree.right)

    def joined_conll_shrinked(self, doc_annot: str, part: str):
        """ Gets a document annotation in conll-like format:
            id text role head function
            as in the example:

            # id = micro_c058
            1	A deposit on soft drink bottles is a really good idea,	pro	0	cc
            2	it would help the environment	pro	1	sup
            ...
            5	The items you can make from recycling are good for the environment	pro	1	sup

            and part ('en', 'ru', 'en2ru' or 'ru2en').

            Returns the same annotation with RST dependencies:
            id text rst_head rst_relation role head function
        """

        doc_annot = doc_annot.strip().split('\n')
        filename, contents = doc_annot[0][7:].strip(), doc_annot[1:]

        if '(machine translation)' in filename:
            _filename = filename.split(' (machine translation)')[0]
            _part = 'ru2en' if part == 'en_aug' else 'en2ru'
            rst_annot_path = os.path.join('data/rst_shrinked', _part, _filename + '.pkl')
            rst_annot = pickle.load(open(rst_annot_path, 'rb'))

        else:
            _part = part.replace('_aug', '')
            rst_annot_path = os.path.join('data/rst_shrinked', _part, filename + '.pkl')
            rst_annot = pickle.load(open(rst_annot_path, 'rb'))

        deps = self.convert_shrinkedrst2dep(rst_annot)

        numerized_deps = dict()
        for dep in deps:
            child, parent = int(dep[0][3:]) + 1, int(dep[1][3:]) + 1
            numerized_deps[child] = (parent, dep[2])

        dannot = [doc_annot[0]]
        for i, line in enumerate(contents):
            line = line.split('\t')
            if i + 1 in numerized_deps:
                parent, relation = numerized_deps[i + 1]
                line = line[:-3] + [str(parent), relation] + line[-3:]
            else:
                line = line[:-3] + ['0', 'root'] + line[-3:]

            dannot.append('\t'.join(line))

        return '\n'.join(dannot)

    def joined_conll(self, rst_conll: str, arg_conll: str):
        """ Matches two conll annotations: one with RST structure and other with Argument structure.
            Returns the RST annotation with EDU-based argumentative functions.

        >> rst_conll = \"""1	A deposit on soft drink bottles is a really good idea,	0	Root
            2	it would help the environment	1	Explanation
            3	and clean up the roads.	2	Joint
            4	The areas of green space are littered with cans.	1	Background
            5	The items	1	Evaluation
            6	you can make from recycling	5	Elaborate
            7	are good for the environment	5	Same-Unit
            \"""
        >> arg_conll = \"""1	A deposit on soft drink bottles is a really good idea,	pro	0	cc
            2	it would help the environment	pro	1	sup
            3	and clean up the roads.	pro	1	sup
            4	The areas of green space are littered with cans.	pro	3	sup
            5	The items you can make from recycling are good for the environment	pro	1	sup
            \"""
        >> mapper.match_two_conlls(rst_conll, arg_conll)
        <<  1	A deposit on soft drink bottles is a really good idea,	0	Root	pro	0	cc
            2	it would help the environment	1	Explanation	pro	1	sup
            3	and clean up the roads.	2	Joint	pro	1	sup
            4	The areas of green space are littered with cans.	1	Background	pro	3	sup
            5	The items	1	Evaluation	pro	1	sup
            6	you can make from recycling	5	Elaborate	5	same-arg
            7	are good for the environment	5	Same-Unit	5	same-arg
        """

        def make_nx_subtree(triplets):
            """ Make nx subtree from rst triplets (parent, child, relation) """
            nx_subtree = nx.DiGraph()
            for triplet in triplets:
                nx_subtree.add_edge(triplet[0], triplet[1], relation=triplet[2])
            return nx_subtree

        def find_root(nx_subtree):
            """ Find root of nx subtree """
            for node in nx_subtree.nodes():
                if nx_subtree.out_degree(node) == 0:
                    nx_subtree.remove_node(node)
                    break

            for node in nx_subtree.nodes():
                if nx_subtree.out_degree(node) == 0:
                    return node

        rst_conll = rst_conll.strip().split('\n')
        arg_conll = arg_conll.strip().split('\n')

        rst_id2text = dict()
        for line in rst_conll:
            line = line.split('\t')
            rst_id2text[line[0]] = ' ' + line[1] + ' '

        arg_id2text = dict()
        for line in arg_conll:
            line = line.split('\t')
            arg_id2text[line[0]] = ' ' + line[1] + ' '

        rst_id2arg_id = dict()  # 1 to 1
        m_rst_id2arg_id = dict()  # 1 to many
        for rst_id, rst_text in rst_id2text.items():
            for arg_id, arg_text in arg_id2text.items():
                if rst_text == arg_text or arg_text in rst_text:
                    rst_id2arg_id[rst_id] = arg_id
                    break
                elif rst_text in arg_text:
                    m_rst_id2arg_id[rst_id] = arg_id

        # 1 to 1
        arg_id2rst_id = {v: k for k, v in rst_id2arg_id.items()}

        # 1 to many
        for rst_id, arg_id in m_rst_id2arg_id.items():
            if arg_id in arg_id2rst_id:
                if type(arg_id2rst_id[arg_id]) == list:
                    arg_id2rst_id[arg_id].append(rst_id)
                else:
                    arg_id2rst_id[arg_id] = [arg_id2rst_id[arg_id], rst_id]
            else:
                arg_id2rst_id[arg_id] = [rst_id]

        matched_conll = []
        # EDU >= ADU
        for line in rst_conll:
            line = line.split('\t')
            if line[0] in rst_id2arg_id:
                arg_id = rst_id2arg_id[line[0]]
                for arg_line in arg_conll:
                    arg_line = arg_line.split('\t')
                    if arg_line[0] == arg_id:
                        if arg_line[3] == '0':
                            line = line + arg_line[2:]
                        elif arg_id2rst_id.get(arg_line[3]) and type(arg_id2rst_id.get(arg_line[3])) == str:
                            # 1 rst = 1 arg, 1 arg = 1 rst
                            line = line + [arg_line[2], arg_id2rst_id[arg_line[3]], arg_line[4]]
                        else:
                            # 1 rst = 1 arg, 1 arg = many rst
                            line = line + [arg_line[2], '-1', arg_line[4]]
                        break

                matched_conll.append(line)

        # EDU != ADU
        unmatched_conll = []
        clusters = set(m_rst_id2arg_id.values())
        for cluster in clusters:
            arg_line = [arg_line for arg_line in arg_conll if arg_line.split('\t')[0] == cluster][0]
            arg_line = arg_line.split('\t')
            rst_lines = [rst_line for rst_line in rst_conll if
                         rst_line.split('\t')[0] in m_rst_id2arg_id and m_rst_id2arg_id[
                             rst_line.split('\t')[0]] == cluster]
            rst_lines = [rst_line.split('\t') for rst_line in rst_lines]
            matched_lines = [line + arg_line[2:] for line in rst_lines]
            unmatched_conll.append([[line[0], line[2], line[3]] for line in matched_lines])


        # Here we'll deal only with the unmatched RST EDUs.
        # Find the local roots of unmatched EDUs and match them with the argument DUs.
        # arg_id2rst_id = {arg_id: [rst_id1, rst_id2, ...]}
        original_arg_id2rst_id = arg_id2rst_id.copy()
        arg_id2rst_id = dict()
        for rst_id, arg_id in m_rst_id2arg_id.items():
            if arg_id not in arg_id2rst_id:
                arg_id2rst_id[arg_id] = []
            arg_id2rst_id[arg_id].append(rst_id)

        unmatched_subgraphs = [make_nx_subtree(cluster) for cluster in unmatched_conll]
        roots = [find_root(subgraph) for subgraph in unmatched_subgraphs]
        arg_id2rst_id = {arg_id: [rst_id for rst_id in values if rst_id in roots][0] for arg_id, values in
                         arg_id2rst_id.items()}

        # add unmatched nodes to matched nodes linking them to the roots
        for root in roots:
            for node in unmatched_subgraphs[roots.index(root)].nodes():
                line = [line for line in rst_conll if line.split('\t')[0] == node][0].split('\t')
                if node not in [l[0] for l in matched_conll]:
                    if node == root:
                        # current EDU is the root of its ADU
                        for arg_id, arg_text in arg_id2text.items():
                            if line[1] in arg_text:
                                for arg_line in arg_conll:
                                    arg_line = arg_line.split('\t')
                                    id, text, role, head, function = arg_line
                                    if id == arg_id:
                                        arg_head_in_rst = arg_id2rst_id.get(head, original_arg_id2rst_id.get(head, head))
                                        if len(line) == 4:
                                            line += [role, arg_head_in_rst, function]
                                        else:
                                            line[4] = 'pro'
                                            line[5] = line[2]
                                            line[6] = 'sup'
                                        break
                    else:
                        # current EDU is not the root of its ADU; save the intra-ADU relation and define role by root ADU
                        arg_line = [arg_line for arg_line in arg_conll if
                                    arg_line.split('\t')[0] == m_rst_id2arg_id.get(root)][0].split('\t')

                        line += [arg_line[2], line[2], 'same-arg']

                    matched_conll.append(line)

        # Difficult case when the EDU is already matched but the head is not.
        # We'll find the head of the EDU and match it with the head of the ADU.
        # the unmatched EDUs are the ones with head==-1
        for i, line in enumerate(matched_conll):
            head = line[5]
            if head == '-1':
                # find the original argument head
                arg_id = rst_id2arg_id[line[0]]
                arg_line = [arg_line for arg_line in arg_conll if arg_line.split('\t')[0] == arg_id][0].split('\t')
                arg_head = arg_line[3]
                converted_arg_head = arg_id2rst_id.get(arg_head, arg_head)
                matched_conll[i][5] = converted_arg_head

        # if matched_conll containing nodes 1, 2, 3, ..., n is missing some node, add it
        broken_lines = [line.split('\t') for line in [
            '5	Однако, поскольку скейт, шахматы и т.д. не принимаются в качестве олимпийских соревнований, компьютерные игры также не должны быть признаны олимпийскими событиями.	4	contrast	pro	6	sup',]]

        for node in range(1, len(rst_conll) + 1):
            if str(node) not in [line[0] for line in matched_conll]:
                line = [line for line in rst_conll if line.split('\t')[0] == str(node)][0].split('\t')
                matched_conll.append(line + ['pro', line[2], 'same-arg'])

        for i, line in enumerate(matched_conll):
            if line in broken_lines:
                matched_conll[i] = line[:-3] + ['pro', '0', 'cc']

            if int(line[5]) > len(matched_conll):
                matched_conll[i][5] = str(len(matched_conll))

        return '\n'.join(['\t'.join(row) for row in sorted(matched_conll, key=lambda x: int(x[0]))])
