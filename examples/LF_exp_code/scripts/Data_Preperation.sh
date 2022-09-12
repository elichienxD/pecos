#================= inputs =====================
dataset=$1 # This is th dataset name (i.e LF-Amazon-131K).
work_dir="."

python3 DataPrep_forXCrepo.py --dataset $dataset

python3 pecos_dataform_full.py $dataset

perl convert_format.pl ./dataset/$dataset/train.txt ./dataset/$dataset/trn_X_Xf.txt ./dataset/$dataset/trn_X_Y.txt

perl convert_format.pl ./dataset/$dataset/test.txt ./dataset/$dataset/tst_X_Xf.txt ./dataset/$dataset/tst_X_Y.txt

echo "Data Preparation for ${dataset} is done!"
