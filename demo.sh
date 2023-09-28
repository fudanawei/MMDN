# datafolder=../Data/20230831233500-rawdata-fr100-randPattern32_bin12_100000_a16_wbg_round-exp100
# datafolder=../Data/20230407210410-rawdata-fr100-randPattern32_bin12_100000_wbg_round-exp70  # best
# datafolder=/media/aa/SSD1T/MMF/Data/20221027015935-rawdata-fr100-randPattern32_bin12_100000_wbg_round-exp80
# datafolder=/media/aa/SSD1T/MMF/Data/20230906005102-rawdata-fr100-randPattern32_bin12_100000_a16_wbg_round-exp50
# datafolder=/media/aa/SSD1T/MMF/Data/20230907223124-rawdata-fr100-randPattern32_bin12_100000_a16_wbg_round-exp40
# datafolder=/media/aa/SSD1T/MMF/Data/20230910190421-rawdata-fr100-randPattern34_bin11_100000_a16_wbg_round-exp50
datafolder=/media/aa/SSD1T/MMF/Data/20230913183156-rawdata-fr100-randPattern34_bin11_100000_a16_wbg_round-exp50

# echo "Test: rebuild update_step 20"
# python co_train_confidence.py --datafolder ${datafolder} \
#                               --saverootfolder ${datafolder}/result_rebuild_freq10 \
#                               --modelfolder ${datafolder}/model_v2 \
#                               --configFile config.yaml \
#                               --sizeOfPretrain 20000 \
#                               --sizeOfBatch 2000 \
#                               --sizeOfUpdateInvertal 1000 \
#                               --speckle_dim 240 \
#                               --gpu 0 \
#                               --pretrain_state 1 \
#                               --update_step 0 \
#                               --flag_rebuild 1 \
#                               --flag_dynamic_learning 1 \
#                               --flag_multi_model 1 \
#                               --use_model pretrain

echo "Test: rebuild update_step 40"
python co_train_confidence_2.py --datafolder ${datafolder} \
                              --saverootfolder ${datafolder}/result_rebuild_freq16_fromscratch \
                              --modelfolder ${datafolder}/model_v3 \
                              --configFile config.yaml \
                              --sizeOfPretrain 20000 \
                              --sizeOfBatch 2000 \
                              --sizeOfUpdateInvertal 1000 \
                              --speckle_dim 240 \
                              --gpu 0 \
                              --pretrain_state 0 \
                              --update_step 64 \
                              --flag_rebuild 1 \
                              --rebuild_interval 8 \
                              --flag_dynamic_learning 1 \
                              --flag_multi_model 1 \
                              --use_model update_64  