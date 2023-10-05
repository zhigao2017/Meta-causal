
# $1 gpuid
# $2 runid

# base方法
cd ..
epochs=70
clsadapt=True
lr=0.01
factor_num=16
lr_scheduler=cosine
lambda_causal=1
lambda_re=1
batchsize=6
stride=5
randm=True
randn=True
autoaug=CA_multiple
network=resnet18

root=saved-PACS/
data=art_painting
svroot=$root/${data}/${autoaug}_${factor_num}fa_v2_ep${epochs}_lr${lr}_${lr_scheduler}_base0.01_bs${batchsize}_lamCa_${lambda_causal}_lamRe${lambda_re}_adt4_cls1_EW2_70_rm${randm}_rn${randn}_str${stride}
python3 main_my_joint_v13_auto.py --gpu $1 --data ${data} --epochs $epochs --autoaug $autoaug --lambda_causal ${lambda_causal} --lambda_re ${lambda_re} \
--lr $lr --svroot $svroot --clsadapt $clsadapt --factor_num $factor_num --lr_scheduler ${lr_scheduler} --batchsize ${batchsize} --network ${network} --randm ${randm} --randn ${randn} --stride ${stride}

test_epoch=last
python3 main_test_pacs_v13.py --gpu $1 --source_domain $data --svroot $svroot --svpath $svroot/${data}_${factor_num}factor_${test_epoch}_test_check.csv --factor_num $factor_num --epoch $test_epoch \
									--network ${network} --stride ${stride}






