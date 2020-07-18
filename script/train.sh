# --- Args ---
# === Solver ===
model_type="PyramidNet_ShakeDrop"
loss_type="VAT"
dataset1="TONAS" #Train Dataset
dataset2="Pop_Rhythm" #Semi-Supervised Dataset
dataset3="Pop_Rhythm_Instrumental" #Instrumental Dataset
dataset4="None" #Validation Dataset
dataset5="DALI" #Test Dataset
mix_ratio=0.5
meta_path="./meta/"
data_path="../data/"
lr=0.0001
lr_warmup=40000
max_steps=240000
se=2
num_feat=9
k=9
batch_size=64
num_workers=1
# === Trainer ===
exp_name="Train_0"
log_path="./log/"
save_path="./checkpoints/"
project="note_segmentation"
entity="austinhsu"
amp_level="O0"
accumulate_grad_batches=1

# --- Flags ---
#--shuffle
#--pin_memory
#--use_cp
#--train
#--test

# --- Make Path ---
mkdir -p $log_path
mkdir -p $save_path

python main.py \
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
	--amp_level $amp_level \
	--accumulate_grad_batches $accumulate_grad_batches \
	--shuffle \
	--pin_memory \
	--use_cp \
	--train