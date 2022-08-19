#================= inputs =====================
model_name=$1 # This should corresponds to the json file in ./params/tfidf, which specifies the configs for tfidf.
work_dir="../"
dname=$2 # This is th dataset name (i.e LF-Amazon-131K).

data_dir="${work_dir}/dataset/${dname}"
model_dir="${data_dir}/tfidf/${model_name}"
mkdir -p ${model_dir}/normalized
mkdir -p ${model_dir}/filtered


params_dir=${work_dir}/scripts/params/tfidf/${model_name}.json

X_trn=${data_dir}/normalized/X.trn.txt
X_tst=${data_dir}/normalized/X.tst.txt

# build tfidf model
python3 -m pecos.utils.featurization.text.preprocess build \
  --text-pos 0 --from-file true \
  --input-text-path ${X_trn} \
  --vectorizer-config-path ${params_dir} \
  --output-model-folder ${model_dir}/normalized

# vectorize
python3 -m pecos.utils.featurization.text.preprocess run \
  --text-pos 0 --from-file true \
  --input-preprocessor-folder ${model_dir}/normalized \
  --input-text-path ${X_trn} \
  --output-inst-path ${model_dir}/X.tfidf.trn.npz
  
# vectorize
python3 -m pecos.utils.featurization.text.preprocess run \
  --text-pos 0 --from-file true \
  --input-preprocessor-folder ${model_dir}/normalized \
  --input-text-path ${X_tst} \
  --output-inst-path ${model_dir}/X.tfidf.tst.npz