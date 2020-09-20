# --- Args ---
# === Solver ===
model_type="PyramidNet_ShakeDrop"
loss_type="VAT" #[None, VAT, VATo]
dataset1="TONAS" #Train Dataset, [TONAS, DALI]
dataset2="MIR_1K" #Semi-Supervised Dataset, [MIR_1K, MIR_1K_Polyphonic, Pop_Rhythm, DALI]
dataset3="None" #Instrumental Dataset, [Pop_Rhythm_Instrumental, MIR_1K_Instrumental]
dataset4="None" #Validation Dataset, [DALI]
dataset5="ISMIR2014" #Test Dataset, [DALI, ISMIR2014]
mix_ratio=0.5
meta_path="./meta/"
data_path="../data/" #"/media/austinhsu/AA0A7F590A7F220D/Ubuntu_Backup/MIR_data/"
lr=0.001
lr_warmup=0 #40000
max_steps=100000 #240000
max_epoch=20
se=2
num_feat=9
k=9
batch_size=64
num_workers=0 #1
# === Trainer ===
exp_name="Training_01"
log_path="./log/"
save_path="./checkpoints/"
project="note_segmentation"
entity="austinhsu"
checkpoint_name="epoch=20.pt"
amp_level="O1"
accumulate_grad_batches=1

# --- Flags ---
#--use_ground_truth
#--shuffle
#--pin_memory
#--use_cp
#--skip_val
#--use_gpu
#--test_no_offset
#--train
#--test

# --- Make Path ---
mkdir -p $log_path
mkdir -p $save_path

python -u main.py \
	--model_type $model_type \
	--loss_type $loss_type \
	--dataset1 $dataset1 \
	--dataset2 $dataset2 \
	--dataset3 $dataset3 \
	--dataset4 $dataset4 \
	--dataset5 $dataset5 \
	--mix_ratio $mix_ratio \
	--meta_path $meta_path \
	--data_path $data_path \
	--lr $lr \
	--lr_warmup $lr_warmup \
	--max_steps $max_steps \
	--max_epoch $max_epoch \
	--se $se \
	--num_feat $num_feat \
	--k $k \
	--batch_size $batch_size \
	--num_workers $num_workers \
	--exp_name $exp_name \
	--log_path $log_path \
	--save_path $save_path \
	--project $project \
	--entity $entity \
	--checkpoint_name $checkpoint_name \
	--amp_level $amp_level \
	--accumulate_grad_batches $accumulate_grad_batches \
	--shuffle \
	--pin_memory \
	--use_cp \
	--skip_val \
	--use_gpu \
	--train