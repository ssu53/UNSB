LOG_FILE="outputs/fid_h2z.txt"
NUM_TEST=1000


echo "Number of test samples: $NUM_TEST" >> $LOG_FILE


# Loop through the specified epochs
for epoch in 50 100 150 200 250 300; do

    # Generate the test images
    python test.py \
        --dataroot ./datasets/horse2zebra \
        --name h2z_SB \
        --checkpoints_dir ./checkpoints \
        --mode sb \
        --eval \
        --phase test \
        --num_test $NUM_TEST \
        --eval \
        --epoch $epoch \
        --gpu_ids 0 \

    # Evaluate FID for each of NFE=1,2,3,4,5
    for nfe in 1 2 3 4 5; do
        echo "Epoch: $epoch, NFE: $nfe" >> $LOG_FILE
        python -m pytorch_fid results/h2z_SB/test_${epoch}/images/fake_${nfe} datasets/horse2zebra/testB >> $LOG_FILE 2>&1
    done

done


