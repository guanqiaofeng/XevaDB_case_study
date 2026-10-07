# nest_env/ is not tracked in git (gitignored). Recreate it with:
#   python3.9 -m venv nest_env && source nest_env/bin/activate && pip install -r requirements_predict.txt
source nest_env/bin/activate

for i in 1 2 3 4 5
do
python src/predict.py \
 -gene2id test_data/McGill_cisplatin/gene2ind.txt \
 -cell2id test_data/McGill_cisplatin/cell2ind.txt \
 -mutations test_data/McGill_cisplatin/cell2mutation.txt \
 -cn_deletions test_data/McGill_cisplatin/cell2cndeletion.txt \
 -cn_amplifications test_data/McGill_cisplatin/cell2amplification.txt \
 -predict test_data/McGill_cisplatin/test_data.txt \
 -hidden hidden/McGill_cisplatin/fold$i \
 -result results/McGill_cisplatin/fold$i \
 -load pretrained_models/cisplatin/model_cisplatin_$i.pt \
 -std sample/std.txt \
 -cuda 0 \
 -batchsize 2000
done