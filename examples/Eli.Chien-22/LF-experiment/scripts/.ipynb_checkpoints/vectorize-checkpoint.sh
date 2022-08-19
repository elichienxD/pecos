#================= inputs =====================
model_name="v4"
work_dir="/efs/core-pecos/users/elichie/MainProject"
dname="LF-WikiSeeAlso-320K"

data_dir="${work_dir}/dataset/LF-dataset/${dname}"
model_dir="${data_dir}/tfidf/${model_name}"
mkdir -p ${model_dir}/normalized
mkdir -p ${model_dir}/filtered


params_dir=${work_dir}/scripts_LF/params/tfidf/${model_name}.json

X_trn=${data_dir}/normalized/X.trn.txt
X_tst=${data_dir}/normalized/X.tst.txt
# X_trn_filter=${data_dir}/filtered/X.trn.txt
# X_tst_filter=${data_dir}/filtered/X.tst.txt

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

# # build tfidf model
# python3 -m pecos.utils.featurization.text.preprocess build \
#   --text-pos 0 --from-file true \
#   --input-text-path ${X_trn_filter} \
#   --vectorizer-config-path ${params_dir} \
#   --output-model-folder ${model_dir}/filtered

# # vectorize
# python3 -m pecos.utils.featurization.text.preprocess run \
#   --text-pos 0 --from-file true \
#   --input-preprocessor-folder ${model_dir}/filtered \
#   --input-text-path ${X_trn_filter} \
#   --output-inst-path ${model_dir}/X.tfidf.trn.filter.npz
  
# # vectorize
# python3 -m pecos.utils.featurization.text.preprocess run \
#   --text-pos 0 --from-file true \
#   --input-preprocessor-folder ${model_dir}/filtered \
#   --input-text-path ${X_tst_filter} \
#   --output-inst-path ${model_dir}/X.tfidf.tst.filter.npz