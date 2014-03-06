dir=`pwd`
local=`pwd`/local
pc_local=`pwd`/../phoneclassification/local
pc_exp=$dir/../phoneclassification/exp/parts_pegasos
data=$dir/../phoneclassification/data/local/data
pc_data=$pc_exp
conf=$dir/conf
mkdir -p $conf
exp=exp/discriminative_parts
mkdir -p $exp


python $local/class_specific_part_training.py --root_dir $dir \
    --data_dir $pc_data/ \
    --use_sparse_suffix bsparse.npy \
    --dev_sparse_suffix dev_bsparse.npy \
    --total_iter 200 \
    --total_init 8 \
    --ncomponents_per_class 5 \
    --part_size 9 9 \
    --noverlap 4 \
    --out_suffix classspec_5C_9T9F4O.npy \
    --out_prefix $exp \
    --tol 1e-6 \
    --min_counts 30 \
    
python $local/compute_class_specific_features.py --root_dir $dir \
    --data_dir $pc_data/\
    --use_sparse_suffix bsparse.npy \
    --dev_sparse_suffix dev_bsparse.npy \
    --model_prefix $exp \
    --model_avgs classspec_5C_9T9F4O.npy \
    --ncomponents_per_class 5 \
    --part_size 9 9 \
    --noverlap 4 \
    --out_suffix classspec_5C_9T9F4O.npy \
    --out_prefix $exp 

python $local/fast_48phone_EM.py --root_dir $dir \
    --in_prefix $exp/ \
    --in_suffix classspec_5C_9T9F4O.npy \
    --label_in_prefix $pc_exp/ \
    --label_in_suffix bsparse.npy \
    --label_in_suffix_test dev_bsparse.npy \
    --out_prefix $exp \
    --out_suffix classspec_5C_5C_9T9F0.npy \
    --total_iter 1000 \
    --total_init 8 \
    --min_counts 30 \
    --tol 1e-6 \
    --ncomponents 5


python $local/fast_48phone_EM.py --root_dir $dir \
    --in_prefix $exp/ \
    --in_suffix classspec_5C_9T9F4O.npy \
    --label_in_prefix $pc_exp/ \
    --label_in_suffix bsparse.npy \
    --label_in_suffix_test dev_bsparse.npy \
    --out_prefix $exp \
    --out_suffix classspec_8C_5C_9T9F0.npy \
    --total_iter 1000 \
    --total_init 8 \
    --min_counts 30 \
    --tol 1e-6 \
    --ncomponents 8

python $local/fast_multicomponent48_pegasos_training.py --root_dir $dir \
    --in_prefix $exp/ \
    --in_suffix classspec_5C_9T9F4O.npy \
    --label_in_prefix $pc_exp/ \
    --label_in_suffix bsparse.npy \
    --label_in_suffix_test dev_bsparse.npy \
	--model_avgs $exp/avgs_classspec_5C_5C_9T9F0.npy \
	--model_meta $exp/meta_classspec_5C_5C_9T9F0.npy \
	--save_prefix $exp/W_fast_pegasos_1tsc_classspec_5C_5C_9T9F0_\
	-l .05 .02  \
        --niter 12 --time_scaling 1.0 \
        --use_hinge 1 --reuse_previous_iterates



