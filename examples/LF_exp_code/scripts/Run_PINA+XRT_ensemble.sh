#================= inputs =====================
dataset=$1 # This is th dataset name (i.e LF-Amazon-131K).
work_dir="."

echo "Preparing PINA data"

python3 PrepareXYstack-raw.py --dataset $dataset

echo "Pretrain PINA neighbor predictor"

./scripts/xtransformer-XYstack-raw.sh v0-raw BoW $dataset

echo "Run PINA augmentation"

python3 PINA_augmentation.py --model_name v0-raw --feature_name BoW --dataset $dataset --L_option Lft_xrt --Pk 5

echo "Start running XR-Transformer with PINA augmented features"
./scripts/xtransformer-XYstack-DS-raw.sh v0-raw v0-raw BoW $dataset 5 Lft_xrt
./scripts/xtransformer-XYstack-DS-raw.sh v0-raw v0-raw-s1 BoW $dataset 5 Lft_xrt
./scripts/xtransformer-XYstack-DS-raw.sh v0-raw v0-raw-s2 BoW $dataset 5 Lft_xrt

echo "Do ensemble!"
python3 Ensemble-PINA.py --dataset $dataset \
                    --DS_model_names v0-raw,v0-raw-s1,v0-raw-s2 \
                    --feature_name BoW \
                    --ens_name softmax

ENS_path="./models_LF/xtransformer/${dataset}/v0-raw,v0-raw-s1,v0-raw-s2/BoW"

echo $ENS_path

./scripts/Ensemble_evaluation.sh softmax $dataset $ENS_path

