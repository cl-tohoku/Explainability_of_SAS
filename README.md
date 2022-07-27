# Explainability_of_SAS
Code for [Plausibility and Faithfulness of Feature Attribution-based Explanations in Automated Short Answer Scoring]() paper (AIED 2022)

## Installation 

```git clone https://github.com/cl-tohoku/Explainability_of_SAS.git```


### Requirements 

```bash
pip install -r requirements.txt
```

## Preparing the Datasets 
download from https://doi.org/10.32130/rdata.3.1


## Questions I used
|    prompt   | item | 
| ----------- | ---- |
| Y14_1-2_1_3 |  A   |
|             |  B   |
|             |  C   |
|             |  D   |
| Y14_1-2_2_4 |  A   |
|             |  C   |
| Y14_2-1_1_5 |  A   |
|             |  C   |
| Y14_2-1_2_3 |  A   |
|             |  B   |
|             |  D   |
| Y14_2-2_1_4 |  B   |
|             |  C   |
| Y14_2-2_2_3 |  A   |
|             |  B   |
|             |  C   |
| Y15_1-1_1_4 |  A   |
|             |  C   |
| Y15_1-3_1_2 |  A   |
|             |  B   |
| Y15_2-2_1_5 |  A   |
|             |  B   |
| Y15_2-2_2_4 |  A   |
|             |  B   |
|             |  C   |
| Y15_1-3_1_5 |  A   |
| Y15_2-2_1_3 |  A   |
|             |  B   |


## Training & Running Experiments
```bash
DATADIR=[data path]
prompt=Y14_2-2_1_4
item=A
train_size=200
seed=0
OUTPUT_DIR=[output directory]
OUTPUT_PREFIX=[output file name prefix]
seed=0
GPU=0
attention_size=200
word2vec_path=[embedding path]
emb_dim=[embedding dim]

mkdir -p ${OUTPUT_DIR}
CUDA_VISIBLE_DEVICES=${GPU} python train_item_scoring_model_eraser.py \
    -tr ${DATADIR}/${prompt}_train.${train_size}.${seed}.json \
    -dv ${DATADIR}/${prompt}_dev.${seed}.json \
    -ts ${DATADIR}/${prompt}_test.${seed}.json  \
    -o ${OUTPUT_DIR}/${OUTPUT_PREFIX} \
    -pa -ph --epochs 50 -b 32 -dmy --item ${item} --emb ${word2vec_path} -seed ${seed} -satt --attention_train_size ${attention_size} \
    -emb_dim ${emb_dim}   

CUDA_VISIBLE_DEVICES=${GPU} python make_feature_map.py -info ${OUTPUT_DIR}/${OUTPUT_PREFIX}_train_info.pickle 

CUDA_VISIBLE_DEVICES=${GPU} python measure_justification_identification.py -info ${OUTPUT_DIR}/${OUTPUT_PREFIX}_train_info.pickle -izs

CUDA_VISIBLE_DEVICES=${GPU} python measure_faithfulness_eraser.py -info ${OUTPUT_DIR}/${OUTPUT_PREFIX}_train_info.pickle --flip_mode 
```