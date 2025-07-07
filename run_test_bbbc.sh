LOG_FILE="outputs/fid_bbbc.txt"
NUM_TEST=1024
NAME=bbbc_SB

echo "Number of test samples: $NUM_TEST" >> $LOG_FILE


# Loop through the specified epochs
for epoch in iter_100000 iter_200000 iter_300000; do

    # Generate the test images
    python test.py \
        --cell_dataset_name bbbc021 \
        --dataset_mode unaligned_cell \
        --name $NAME \
        --checkpoints_dir ./checkpoints \
        --mode sb \
        --eval \
        --phase test \
        --direction AtoB \
        --gpu_ids 0 \
        --cond_dim 1024 \
        --num_test $NUM_TEST \
        --epoch $epoch \


    # Evaluate FID for each of NFE=1,2,3,4,5
    for nfe in 1 2 3 4 5; do
        echo "Epoch: $epoch, NFE: $nfe" >> $LOG_FILE
        python -m pytorch_fid results/${NAME}/test_${epoch}/images/fake_${nfe} /pasteur/u/shiye/datasets/bbbc021/testB_1024 >> $LOG_FILE 2>&1
    done

done
