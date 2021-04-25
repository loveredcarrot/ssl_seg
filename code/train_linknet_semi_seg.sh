nohup python3  train_fully_supervised_2D.py --exp  /Fu_86v1_linkbase_4_7  --model LinkNetBase  --gpu 0 --labeled_csv train_label_86_v1.txt >Fu_86v1_linkbase_4_7.log 2>&1 &  && \
nohup python3  train_fully_supervised_2D.py --exp  /Fu_86v1_4_7  --model LinkNetBaseWithDrop --gpu 1 --labeled_csv train_label_86_v1.txt >Fu_86v1_4_7.log 2>&1 &    && \
nohup python3  train_mean_teacher_2D.py --exp /Mt_86v1_4_7 --gpu 2 --labeled_csv train_label_86_v1.txt --unlabeled_csv train_unlabeled_8462_v1.txt  --consistency 1 --noise_variance 0 --noise_range 0 >Mt_86v1_4_7.log 2>&1 &  && \
nohup python3 train_cutmix_2D_1.py --exp /CutMix_86v1_4_7 --gpu 3 --labeled_csv train_label_86_v1.txt --unlabeled_csv train_unlabeled_8462_v1.txt --consistency 5 >CutMix_86v1_4_7.log 2>&1 & && \
nohup python3  train_uncertainty_aware_mean_teacher_2D.py --exp /uamt_temp_4_23 --gpu 0 >uamt_temp_4_23.log 2>&1 &