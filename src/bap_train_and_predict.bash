# Using: $ bash bap_train_and_predict.bash --lang {en|ru} --train_data {en|ru|en_aug|ru_aug}

# Parsing parameters
lang=${lang:-"en"}
train_data=${train_data:-"en_aug"}
lrb=${lrb:-0.00002}
lrp=${lrp:-0.000002}

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        echo $1 $2
   fi

  shift
done


export LR_BERT=$lrb
export LR_PARSER=$lrp
export LANG=$lang
export TRAIN_DATA=$train_data

echo LANG $LANG
echo TRAIN_DATA $TRAIN_DATA

# Prepare directory
export DIRECTORY=bap_${TRAIN_DATA}
rm -r $DIRECTORY
mkdir $DIRECTORY

# Prepare configs for K-fold validation
export CONFIGFILE=configs/bap_${LANG}_fold_0.jsonnet
rm configs/bap_${LANG}_fold*
cp initial_configs/bap_${LANG}_fold_0.jsonnet $CONFIGFILE
echo CONFIGFILE $CONFIGFILE

K=10
python utils/make_k_copies.py --filename $CONFIGFILE --k $K

for (( FOLD=0; FOLD<$K; FOLD++ ))
do
    allennlp train -s ${DIRECTORY}/fold_${FOLD} configs/bap_${LANG}_fold_${FOLD}.jsonnet
    allennlp predict --use-dataset-reader --silent \
                     --output-file ${DIRECTORY}/fold_${FOLD}/predictions_test.json ${DIRECTORY}/fold_${FOLD}/model.tar.gz \
                     data/conll_cv/${LANG}/test_fold_${FOLD}.conll \
                     --include-package dependency_parser
done