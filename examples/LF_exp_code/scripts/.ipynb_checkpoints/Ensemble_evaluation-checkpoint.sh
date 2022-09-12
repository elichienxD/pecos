#================= inputs =====================
ens_name=$1
dname=$2
topk=20
model_dir=$3

echo "===After reciprocal pair removal==="
python3 -u ./evaluate.py \
            "./dataset/${dname}/trn_X_Y.txt" \
            "./dataset/${dname}/tst_X_Y.txt" \
            "${model_dir}/P.${topk}.${ens_name}" ./dataset/${dname} \
            |& tee ${model_dir}/Reciprocal_Removed_eval.log 