
# $1 gpuid

cd ..
epochs=500
clsadapt=True
lr=1e-4
lr_scheduler=Step
factor_num=14
test_epoch=best
lambda_causal=1
lambda_re=1
batchsize=32
stride=3
randm=True
randn=True
autoaug=CA_multiple


root=saved-digit
svroot=$root/${autoaug}_${factor_num}fa_all_ep${epochs}_lr${lr}_lr_scheduler${lr_scheduler}0.8_bs${batchsize}_lamCa_${lambda_causal}_lamRe_${lambda_re}_cls1_adt2_EW2_100_rm${randm}_rn${randn}_str${stride}

python3 main_my_joint_v13_auto.py --gpu $1 --data mnist --epochs $epochs --autoaug $autoaug --lambda_causal ${lambda_causal} --lambda_re ${lambda_re} \
--lr $lr --lr_scheduler $lr_scheduler --svroot $svroot --clsadapt $clsadapt --factor_num $factor_num --batchsize ${batchsize} --randm ${randm} --randn ${randn} --stride ${stride}

python3 main_test_digit_v13.py --gpu $1 --svroot $svroot --svpath $svroot/${factor_num}factor_${test_epoch}.csv --factor_num $factor_num --epoch $test_epoch \
									--stride ${stride}







