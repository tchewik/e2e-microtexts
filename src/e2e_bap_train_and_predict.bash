# Using: $ bash e2e_bap_train_and_predict.bash --lang {en|ru} --train_data {en|ru|en_aug|ru_aug}

# Parsing parameters
lang=${lang:-"en"}
train_data=${train_data:-"en_aug"}
while [ $# -gt 0 ]; do
   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        echo $1 $2
   fi
   shift
done

crossval ()
{
    LANG=$lang
    TRAIN_DATA=$train_data

    # Prepare directory
    DIRECTORY=e2e_bap_${TRAIN_DATA}

    # Prepare configs for K-fold validation
    CONFIGFILE=configs/e2e_bap_${LANG}_fold_0.jsonnet
    rm configs/e2e_bap_${LANG}_fold*
    cp initial_configs/e2e_bap_${LANG}_fold_0.jsonnet $CONFIGFILE
    echo CONFIGFILE $CONFIGFILE

    K=10
    python utils/make_k_copies.py --filename $CONFIGFILE --k $K

    # Default training
    rm -r $DIRECTORY
    mkdir $DIRECTORY
    for (( FOLD=0; FOLD<$K; FOLD++ ))
    do
        TRAIN_DATA=${TRAIN_DATA} allennlp train -s ${DIRECTORY}/fold_${FOLD} configs/e2e_bap_${LANG}_fold_${FOLD}.jsonnet
        allennlp predict --use-dataset-reader --silent \
                          --output-file ${DIRECTORY}/fold_${FOLD}/predictions_test.json ${DIRECTORY}/fold_${FOLD}/model.tar.gz \
                          data/conll_cv_rst_end2end/${LANG}/test_fold_${FOLD}.conll \
                          --include-package dependency_parser
    done

    # Metrics report
    python utils/latex_report_line.py --model_name ${DIRECTORY} >> ${DIRECTORY}/latex_report.tex
    # Weights illustration (not for BAP)
    # python utils/crossval/rst_weights_plot.py --model_name ${DIRECTORY} --k ${K}
}

crossval_tune ()
{
    LANG=$lang
    TRAIN_DATA=$train_data

    # Prepare directory
    DIRECTORY=e2e_bap_${TRAIN_DATA}

    # Prepare configs for K-fold validation
    CONFIGFILE=configs/e2e_bap_${LANG}_fold_0.jsonnet
    rm configs/e2e_bap_${LANG}_fold*
    cp initial_configs/e2e_bap_${LANG}_fold_0.jsonnet $CONFIGFILE
    echo CONFIGFILE $CONFIGFILE

    K=10
    python utils/make_k_copies.py --filename $CONFIGFILE --k $K

    mkdir parameters_adjustments_results
    # Fill in the parameters for adjustment
    for DROPOUT in 0.0 0.1 0.3 0.4
    do
       rm -r $DIRECTORY
       mkdir $DIRECTORY
       for (( FOLD=0; FOLD<$K; FOLD++ ))
       do
           TRAIN_DATA=${TRAIN_DATA} DROPOUT=${DROPOUT} allennlp train -s ${DIRECTORY}/fold_${FOLD} configs/e2e_bap_${LANG}_fold_${FOLD}.jsonnet
           allennlp predict --use-dataset-reader --silent \
                            --output-file ${DIRECTORY}/fold_${FOLD}/predictions_test.json ${DIRECTORY}/fold_${FOLD}/model.tar.gz \
                            data/conll_cv_rst_end2end/${LANG}/test_fold_${FOLD}.conll \
                            --include-package dependency_parser
       done
       python utils/latex_report_line.py --model_name ${DIRECTORY} >> parameters_adjustments_results/${DIRECTORY}_DROPOUT_${DROPOUT}.tex
    done
}

crossval