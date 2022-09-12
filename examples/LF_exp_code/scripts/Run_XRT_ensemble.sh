#================= inputs =====================
dataset=$1 # This is th dataset name (i.e LF-Amazon-131K).
work_dir="."

echo "Start running XR-Transformer"
./scripts/xtransformer-raw.sh "v0-raw" BoW $dataset
./scripts/xtransformer-raw.sh "v0-raw-s1" BoW $dataset
./scripts/xtransformer-raw.sh "v0-raw-s2" BoW $dataset

echo "Do ensemble!"
python3 Ensemble.py --dataset $dataset \
                    --DS_model_names "v0-raw,v0-raw-s1,v0-raw-s2" \
                    --feature_name BoW \
                    --ens_name softmax

ENS_path="./models_LF/xtransformer/${dataset}/v0-raw,v0-raw-s1,v0-raw-s2/BoW"

echo $ENS_path

./scripts/Ensemble_evaluation.sh softmax $dataset $ENS_path

