dir=`pwd`
local=`pwd`/local
pc_local=`pwd`/../phoneclassification/local
pc_exp=$dir/../phoneclassification/exp/parts_pegasos
pc_data=$pc_exp
conf=$dir/conf
mkdir -p $conf
dp_exp=$dir/exp/discriminative_parts
exp=$dir/exp/ambiguity_modeling
mkdir -p $exp


python $local/ambiguity_modeling_sanity_check.py --data_prefix $pc_data/ \
    --data_suffix bsparse.npy \
    --phn ih \
    --dim $pc_data/dim_bsparse.npy \
    --cov $dp_exp/general_covariance_cov_9T9F_parts.npy \
    --mean $dp_exp/general_covariance_feature_counts_all_9T9F_parts.npy\
    --leehon $conf/phones.48-39
