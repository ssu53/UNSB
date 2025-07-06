LOG_FILE="outputs/fid_m2s.txt"
NUM_TEST=2000


echo "Number of test samples: $NUM_TEST" >> $LOG_FILE


# Loop through the specified epochs
for epoch in 400 350 300; do

    # Generate the test images
    python test.py \
        --dataroot ./datasets/maps \
        --name maps_SB \
        --checkpoints_dir ./checkpoints \
        --mode sb \
        --eval \
        --phase test \
        --direction BtoA \
        --num_test $NUM_TEST \
        --eval \
        --epoch $epoch \
        --gpu_ids 0 \

    # Evaluate FID for each of NFE=1,2,3,4,5
    for nfe in 1 2 3 4 5; do
        echo "Epoch: $epoch, NFE: $nfe" >> $LOG_FILE
        python -m pytorch_fid results/maps_SB/test_${epoch}/images/fake_${nfe} datasets/maps/testA >> $LOG_FILE 2>&1
    done

done

