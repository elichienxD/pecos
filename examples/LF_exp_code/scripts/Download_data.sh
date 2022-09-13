#================= inputs =====================
dataset=$1 # This is th dataset name (i.e LF-Amazon-131K).
work_dir="."
cd dataset
echo "$(pwd)"

echo "Downloading ${dataset}"
if [[ $dataset == "LF-Amazon-131K" ]]
then
    rawtext_id="1WuquxCAg8D4lKr-eZXPv4nNw2S2lm7_E"
    BoW_id="1YNGEifTHu4qWBmCaLEBfjx07qRqw9DVW"
elif [[ $dataset == "LF-AmazonTitles-131K" ]]
then
    rawtext_id="1WuquxCAg8D4lKr-eZXPv4nNw2S2lm7_E"
    BoW_id="1VlfcdJKJA99223fLEawRmrXhXpwjwJKn"
elif [[ $dataset == "LF-WikiSeeAlsoTitles-320K" ]]
then
    rawtext_id="1QZD4dFVxDpskCI2kGH9IbzgQR1JSZT-N"
    BoW_id="1edWtizAFBbUzxo9Z2wipGSEA9bfy5mdX"
elif [[ $dataset == "LF-WikiSeeAlso-320K" ]]
then
    rawtext_id="1QZD4dFVxDpskCI2kGH9IbzgQR1JSZT-N"
    BoW_id="1N8C_RL71ErX6X92ew9h8qRuTWJ9LywE8"
elif [[ $dataset == "LF-WikiTitles-500K" ]]
then
    rawtext_id="1lKUyg4VUYBdILyb6t8gKpOyCz9pdpJQw"
    BoW_id="1qa1HTTD509J5r4yNAH-Aq6wArljgg4mx"
elif [[ $dataset == "LF-AmazonTitles-1.3M" ]]
then
    rawtext_id="12zH4mL2RX8iSvH0VCNnd3QxO4DzuHWnK"
    BoW_id="1Davc6BIfoTIAS3mP1mUY5EGcGr2zN2pO"
fi

echo $rawtext_id

gdown $rawtext_id
gdown $BoW_id

unzip $dataset.raw.zip
unzip -o $dataset.bow.zip

data_dir="${dataset}"
mkdir -p ${data_dir}/normalized
mkdir -p ${data_dir}/raw

cd ${dataset}

mv *.json.gz raw

cd raw

gunzip *.gz 

echo "${dataset} downlowded and unzipped!!!"

