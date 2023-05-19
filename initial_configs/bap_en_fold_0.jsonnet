local foldnum = 0;
local TRAIN_DATA = std.extVar('TRAIN_DATA');
local TRAIN_DATA_PATH = "data/conll_cv/" + TRAIN_DATA + "/train_fold_" + foldnum + ".conll";
local DEV_DATA_PATH = "data/conll_cv/en/dev_fold_" + foldnum + ".conll";
local TEST_DATA_PATH = "data/conll_cv/en/test_fold_" + foldnum + ".conll";
local BERT_MODEL = "microsoft/mdeberta-v3-base";
local MAX_TOKENS = 150;

local BETA_1 = 0.9;
local BETA_2 = 0.9;

# Comment for tuning
local LR_BERT = 2e-5;
local LR_PARSER = 2e-6;

# Uncomment for tuning
# local LR_PARSER = std.parseJson(std.extVar('LR_PARSER'));
# local LR_BERT = std.parseJson(std.extVar('LR_BERT'));


{
    "dataset_reader":{
        "type": "dependency_parser.dataset_readers.ud_microtexts.UDMicrotextsDatasetReader",
        "tokenizer": {
            "type": "pretrained_transformer",
            "add_special_tokens": true,
            "model_name": BERT_MODEL,
            "max_length": MAX_TOKENS,
        },
        "token_indexers": {
          "bert": {
              "type": "pretrained_transformer",
              "model_name": BERT_MODEL,
              "max_length": MAX_TOKENS,
          },
        }
    },
    "train_data_path": TRAIN_DATA_PATH,
    "validation_data_path": DEV_DATA_PATH,
    "test_data_path": TEST_DATA_PATH,
    "evaluate_on_test": true,
    "model": {
        "type": "dependency_parser.models.biaffine_argument_parser.BiaffineArgumentParser",
        "text_field_embedder": {
            "token_embedders": {
                "bert": {
                    "type": "pretrained_transformer",
                    "max_length": MAX_TOKENS,
                    "model_name": BERT_MODEL,
                    "train_parameters": true,
                    "last_layer_only": true,
                },
        },
      },
      "encoder": {
          "type": "cls_pooler",
          "embedding_dim": 768,
      },
      "use_mst_decoding_for_validation": true,
      "arc_representation_dim": 100,
      "tag_representation_dim": 50,
      "dropout": 0.2,
      "input_dropout": 0.0,
      "initializer": {
        "regexes": [
          [".*projection.*weight", {"type": "xavier_uniform"}],
          [".*projection.*bias", {"type": "zero"}],
          [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
          [".*tag_bilinear.*bias", {"type": "zero"}],
        ]
      }
    },
    "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "batch_size" : 4,
        "sorting_keys": ["spans"],
      }
    },
    "trainer": {
      "num_epochs": 300,
      "patience": 30,
      "validation_metric": "+LAS",
      "optimizer": {
        "type": "huggingface_adamw",
        "weight_decay": 0.1,
        "betas": [BETA_1, BETA_2],
        "parameter_groups": [
                   [[".*bert.*", "encoder.pooler.*", "_head_sentinel_encoder.*"], {"lr": LR_BERT}],
                   [["head_arc_feedforward.*", "child_arc_feedforward.*", "arc_attention",
                     "head_tag_feedforward.*", "child_tag_feedforward.*", "tag_bilinear.*",],
                    {"lr": LR_PARSER}],
       ]
      },
      "learning_rate_scheduler": {
            "type": "linear_with_warmup",
            "warmup_steps":100
      },
      "keep_most_recent_by_count": 1,
      "cuda_device": 0,
    }
}
