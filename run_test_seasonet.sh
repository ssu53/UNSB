LOG_FILE="outputs/fid_seasonet.txt"
NUM_TEST=1024
NAME=seasonet_SB

echo "Number of test samples: $NUM_TEST" >> $LOG_FILE



# Loop through the specified epochs
for epoch in iter_100000 iter_200000; do

    # Generate the test images
    python test.py \
        --dataroot ./datasets/seasonet \
        --name $NAME \
        --checkpoints_dir ./checkpoints \
        --mode sb \
        --eval \
        --phase test \
        --direction AtoB \
        --num_test $NUM_TEST \
        --eval \
        --epoch $epoch \
        --gpu_ids 0 \
        --load_size 120 \
        --crop_size 120 \
    
    # Evaluate FID for each of NFE=1,2,3,4,5
    for nfe in 1 2 3 4 5; do
        echo "Epoch: $epoch, NFE: $nfe" >> $LOG_FILE
        python -m pytorch_fid results/${NAME}/test_${epoch}/images/fake_${nfe} datasets/seasonet/testB_1024 >> $LOG_FILE 2>&1
    done

done

