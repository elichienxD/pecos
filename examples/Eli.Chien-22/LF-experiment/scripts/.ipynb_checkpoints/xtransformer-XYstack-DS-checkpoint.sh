#================= inputs =====================
model_name=$1
DS_model_name=$2
feature_name=$3
work_dir="../"
dname=$4
topk=20
P_topk=$5
L_option=$6

feature_dir="${work_dir}/dataset/LF-dataset/${dname}/tfidf/default/XYstack/normalized"
params_path=${work_dir}/scripts_LF/params/xtransformer/${DS_model_name}.json

# model_dir="${work_dir}/models_LF/xtransformer/${dname}/${model_name}/${feature_name}/XYstack"
model_dir="${work_dir}/models_LF/xtransformer/${dname}/${model_name}/${feature_name}/XYstack/downstream/${DS_model_name}/${P_topk}/${L_option}"
pretrain_dir="${work_dir}/models_LF/xtransformer/${dname}/${model_name}/${feature_name}/XYstack"
mkdir -m777 -p ${model_dir}

X_trn=${work_dir}/dataset/LF-dataset/${dname}/normalized/X.trn.txt
Xf_trn=${pretrain_dir}/X_trn_P${P_topk}${L_option}.npz
Y_trn=${work_dir}/dataset/LF-dataset/${dname}/normalized/Y.trn.npz

X_tst=${work_dir}/dataset/LF-dataset/${dname}/normalized/X.tst.txt
Xf_tst=${pretrain_dir}/X_tst_P${P_topk}${L_option}.npz
Y_tst=${work_dir}/dataset/LF-dataset/${dname}/normalized/Y.tst.npz

# ================ training ====================

python3 -m pecos.xmc.xtransformer.train -t ${X_trn} -x ${Xf_trn} -y ${Y_trn} -m ${model_dir} --only-topk $topk\
	--params-path ${params_path} \
	|& tee ${model_dir}/train.log
    
# ================ eval ========================
python3 -m pecos.xmc.xtransformer.predict -t ${X_tst} -x ${Xf_tst} -m ${model_dir} --only-topk $topk\
    -o ${model_dir}/P.${topk}.npz \
    |& tee ${model_dir}/eval_tst.log

echo "===Before reciprocal pair removal==="
python3 -m pecos.xmc.xlinear.evaluate -y ${Y_tst} -p ${model_dir}/P.${topk}.npz -k 10 \
    |& tee ${model_dir}/eval_tst.log
    
    
echo "===After reciprocal pair removal==="
python3 -u ../evaluate.py \
            "../dataset/${dname}/trn_X_Y.txt" \
            "../dataset/${dname}/tst_X_Y.txt" \
            "${model_dir}/P.${topk}" ../dataset/${dname} \
            |& tee ${model_dir}/Reciprocal_Removed_eval.log 