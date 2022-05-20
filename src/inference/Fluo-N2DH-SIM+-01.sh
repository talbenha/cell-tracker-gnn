#!/bin/bash

SEQUENCE="01"
FOV="0"
DATASET="${PWD}/../Fluo-N2DH-SIM+"
CODE_SEG="${PWD}/seg_code/old_version"
CODE_TRA="${PWD}"
SEG_MODEL="${PWD}/parameters/Seg_Models/Fluo-N2DH-SIM+/"
MODEL_METRIC_LEARNING="${PWD}/parameters/Features_Models/Fluo-N2DH-SIM+/all_params.pth"
MODEL_PYTORCH_LIGHTNING="${PWD}/parameters/Tracking_Models/Fluo-N2DH-SIM+/checkpoints/epoch=132.ckpt"
MODALITY="2D"
        
# seg prediction
python ${CODE_SEG}/Inference2D.py --gpu_id 0 --model_path ${SEG_MODEL} --sequence_path "${DATASET}/${SEQUENCE}" --output_path "${DATASET}/${SEQUENCE}_SEG_RES" --edge_dist 3 --edge_thresh=0.3 --min_cell_size 100 --max_cell_size 1000000 --fov 0 --centers_sigmoid_threshold 0.8 --min_center_size 10 --pre_sequence_frames 4 --data_format NCHW --save_intermediate --save_intermediate_path ${DATASET}/${SEQUENCE}_SEG_intermediate

# cleanup
rm -r "${DATASET}/${SEQUENCE}_SEG_intermediate"

# Finish segmentation - start tracking

# our model needs CSVs, so let's create from image and segmentation.
python ${CODE_TRA}/preprocess_seq2graph_clean.py -cs 20 -ii "${DATASET}/${SEQUENCE}" -iseg "${DATASET}/${SEQUENCE}_SEG_RES" -im "${MODEL_METRIC_LEARNING}" -oc "${DATASET}/${SEQUENCE}_CSV"

# run the prediction
python ${CODE_TRA}/inference_clean.py -mp "${MODEL_PYTORCH_LIGHTNING}" -ns "${SEQUENCE}" -oc "${DATASET}"

# postprocess
python ${CODE_TRA}/postprocess_clean.py -modality "${MODALITY}" -iseg "${DATASET}/${SEQUENCE}_SEG_RES" -oi "${DATASET}/${SEQUENCE}_RES_inference"

rm -r "${DATASET}/${SEQUENCE}_CSV" "${DATASET}/${SEQUENCE}_RES_inference" "${DATASET}/${SEQUENCE}_SEG_RES"