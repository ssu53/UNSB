LOG_FILE="outputs/fid_s2w.txt"
NUM_TEST=1000 # there are 309 images in summer2winter_yosemite/testA and 238 images in summer2winter_yosemite/testB


echo "Number of test samples: $NUM_TEST" >> $LOG_FILE


# Loop through the specified epochs
for epoch in 400 350; do

    # Generate the test images
    python test.py \
        --dataroot ./datasets/summer2winter_yosemite \
        --name sum2win_vgg \
        --checkpoints_dir vgg_sb/checkpoints \
        --mode sb \
        --eval \
        --phase test \
        --direction AtoB \
        --num_test $NUM_TEST \
        --eval \
        --epoch $epoch \
        --gpu_ids 0 \

    # Evaluate FID for each of NFE=1,2,3,4,5
    for nfe in 1 2 3 4 5; do
        echo "Epoch: $epoch, NFE: $nfe" >> $LOG_FILE
        python -m pytorch_fid results/sum2win_vgg/test_${epoch}/images/fake_${nfe} datasets/summer2winter_yosemite/testB >> $LOG_FILE 2>&1
    done

done

