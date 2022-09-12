#================= inputs =====================
model_name=$1 # This corresponds to the name of the json file in ./params/xtransformer/
feature_name=$2 # Choose between "BoW" or else. "BoW" refers to using default BoW features and else of self-built tfidf feature.
work_dir="."
dname=$3 # The name of dataset
topk=20 # Usually do not need to modify this

params_path=${work_dir}/scripts/params/xtransformer/${dname}/${model_name}.json

model_dir="${work_dir}/models_LF/xtransformer/${dname}/${model_name}/${feature_name}"
mkdir -m777 -p ${model_dir}

if [[ $feature_name == "BoW" ]]
then
    echo $feature_name
    feature_dir="${work_dir}/dataset/${dname}"
    Xf_trn=${feature_dir}/X_bow.trn.npz
    Xf_tst=${feature_dir}/X_bow.tst.npz
    
else
    echo $feature_name
    feature_dir="${work_dir}/dataset/${dname}/tfidf/${feature_name}"
    Xf_trn=${feature_dir}/X.tfidf.trn.npz
    Xf_tst=${feature_dir}/X.tfidf.tst.npz
fi


X_trn=${work_dir}/dataset/${dname}/raw/X.trn.txt
Y_trn=${work_dir}/dataset/${dname}/normalized/Y.trn.npz

X_tst=${work_dir}/dataset/${dname}/raw/X.tst.txt
Y_tst=${work_dir}/dataset/${dname}/normalized/Y.tst.npz


# ================ training ====================

python3 -m pecos.xmc.xtransformer.train -t ${X_trn} -x ${Xf_trn} -y ${Y_trn} -m ${model_dir} --only-topk $topk \
	--params-path ${params_path} \
	|& tee ${model_dir}/train.log
    
# ================ eval ========================
python3 -m pecos.xmc.xtransformer.predict -t ${X_tst} -x ${Xf_tst} -m ${model_dir} --only-topk $topk \
    -o ${model_dir}/P.${topk}.npz \
    |& tee ${model_dir}/eval_tst.log

echo "===Before reciprocal pair removal==="
python3 -m pecos.xmc.xlinear.evaluate -y ${Y_tst} -p ${model_dir}/P.${topk}.npz -k 10 \
    |& tee ${model_dir}/eval_tst.log
    
echo "===After reciprocal pair removal==="
python3 -u ./evaluate.py \
            "./dataset/${dname}/trn_X_Y.txt" \
            "./dataset/${dname}/tst_X_Y.txt" \
            "${model_dir}/P.${topk}" ./dataset/${dname} \
            |& tee ${model_dir}/Reciprocal_Removed_eval.log 

    