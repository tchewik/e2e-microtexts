# Dependencies
apt install software-properties-common -y && add-apt-repository ppa:deadsnakes/ppa
apt update
apt install -y python3-apt python3.9-distutils python3.9-dev python3.9-venv
python3.9 -m venv py39
source py39/bin/activate

pip install git+https://github.com/iinemo/isanlp.git
pip install fire allennlp allennlp-models overrides networkx pydot

# Unpack the baseline
cd baseline && unzip evidencegraph.zip
cd evidencegraph && pip install -U .

# The baseline's dependencies in a separate python3.8 environment (for evaluation)
apt install python3.8-venv
python3.8 -m venv py38-evgraph
source py38-evgraph/bin/activate
pip install -r requirements.txt
pip install fire
pip install -U .
cd ../..

# Data
pip install -q charset_normalizer tqdm bs4 lxml
python src/data/collect_micro_essays.py
