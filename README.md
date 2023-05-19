# End-to-End Argument Mining over Varying Rhetorical Structures

This repository contains the official code and data for the ACL 2023 Findings paper "End-to-End Argument Mining over Varying Rhetorical Structures".

## Environment setup 

```commandline
bash src/dependencies_and_data.bash
source py39/bin/activate
```

The bash script sets up the environment and collects the corpus (283 documents total) in a single ``data/data.pkl`` file.
Run the script first, _then_ install the appropriate [pytorch version](https://pytorch.org/get-started/previous-versions/) for your cuda if necessary.


## 1. Data preparation

### 1.1 Back translation

> :warning: **Processing documents in our setting takes around 15 s/document on average.**
> 
> The results from the paper are given in [``data/data.pkl``](data/data.pkl).


<details>
<summary>Click to view the steps to generate the data/data.pkl file (Optional)</summary>

Run the [`tchewik/isanlp_hf_translator`](https://hub.docker.com/r/tchewik/isanlp_hf_translator) Docker image on the same or remote machine with the following command:

```commandline
docker run --rm --name translator -p 3332:3333 -d tchewik/isanlp_hf_translator
```

The script requires the IP or domain name of the machine (``servername``) and the source language (english or russian) as arguments:

```commandline
python src/data/run_back_translation.py --servername servername
```

It updates the `data/data.pkl` file with back translations: Ru -> En and En -> Ru.

</details>

For BLUE evaluation, run:
```commandline
python src/data/evaluate_back_translation.py --target_lang english|russian
```


### 1.2 RST parsing

> :warning: **In our setting, processing one document takes roughly 3 seconds (English) or 5 seconds (Russian) on average**. 
> 
> The NLP annotations obtained in our experiments are given in ``data/nlp_annot.zip``. Unpack it into ``data/`` to skip this step: `cd data && unzip nlp_annot.zip`.

<details>

<summary>Click to view the steps to reproduce the data/nlp_annot/ directory (Optional)</summary>

To analyze texts on either local or remote machines, use these commands:

#### For English:
 ```commandline
# spaCy: syntax port 3333
 docker run --rm -d -p 3333:3333 --name spacy_en tchewik/isanlp_spacy:en
# RST parser: parser port 3222
 docker run --rm -d -p 3222:3333 --name rst_en tchewik/isanlp_zhang21
 ```

#### For Russian:
```commandline
# spaCy: syntax port 3334
docker run --rm -d -p 3334:3333 --name spacy_ru tchewik/isanlp_spacy:ru
# RST parser: parser port 3222
docker run --rm -d -p 3222:3333 --name rst_ru tchewik/isanlp_rst:2.1-rstreebank
```

Wait for the parsers to load completely, then run the script:

```commandline 
python src/data/run_rst_parsing.py \
       --servername servername \
       --syntax_port <number> \
       --rst_port <number> \
       --lang english|russian
```

It will save the linguistic annotations in the ``data/nlp_annot/`` directory.

</details>

### 1.3 RST to Microtexts mapping

Run the next command to collect the RST-like structures with ADU as terminal nodes into the ``data/rst_shrinked/`` directory:
```commandline
python src/data/map_rst_to_microtexts.py --lang english|russian
```

### 1.4 Convert RST do dependencies

Run the next command to convert the RST-like structures into dependencies for DBAP models under `data/conll_cv_rst/` directory:

```commandline
python src/data/convert_rst_to_dep.py 
```

The next command converts the RST-like structures into EDU-based dependencies for e2e models under `data/conll_cv_rst_end2end/` directory:

```commandline
python src/data/convert_rst_to_e2e_dep.py
```


## 2. Experiments

### 2.1 Training
All the required training configuration files are in the [`initial_configs/`](initial_configs) directory.

```commandline
# Parsing over handcrafted segmentation
bash src/{bap|dbap}_train_and_predict.bash --lang en|ru --train_data en|ru|en_aug|ru_aug

# End-to-end
bash src/e2e_{bap|dbap}_train_and_predict.bash --lang en|ru --train_data en|ru|en_aug|ru_aug
```

### 2.2 Evaluation
The script requires the dependencies the evidencegraph virtual environment: `source baseline/evidencegraph/py38-evgraph/bin/activate`.

```commandline
# For MODELNAME in {bap|dbap|e2e_bap|e2e_dbap}:
(py38-evgraph) python utils/crossval/eval_parsing.py --dir models/MODELNAME/MODELNAME_{en|ru|en_aug|ru_aug} --k 10

# e2e evaluation without 'same-arg' relation:
(py38-evgraph) python utils/latex_report_line.py --model_name models/MODELNAME/MODELNAME_{en|ru|en_aug|ru_aug}
```
