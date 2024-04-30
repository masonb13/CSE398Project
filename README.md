### How to Run Code

## Build a dataset
1. Put the sharp images in a directory
2. Set the input_directory & output_directory var in DeblurGAN/datasets/my_data.py
3. Run this file to create blurred images
4. Run python DeblurGAN/datasets/combine_A_and_B.py --fold_A /path/to/blurry/images --fold_B path/to/sharp/images --fold_AB path/to/combined/images

You will now have a directory of images that contain the sharp image appended to the blurry images ready for training

### Train on dataset
1. Run python DeblurGAN/train.py --dataroot /path/to/combined/images --learn_residual --resize_or_crop crop --fineSize 256
2. In this file you can set the model name to change what the model saves as every n epochs
3. The number of epochs and other training parameters can be fine tuned in this file lines 70-80

### test on dataset
1. In the DeblurGAN/models/base_model.py file, change the model_name_prefix on line 56 to the name of the model from training you want to test
2. Run python test.py --dataroot /path/to/blurry/images --model test --dataset_mode single --learn_residual
3. This will output a SSIM and PSNR metrics for each image in the test set and an average across the entire set. More about metrics can be found in the paper

### change loss 
1. In the DeblurGAN/models/conditional_gan_model.py line 99 you can change the backward_G function for whichever loss you want
2. The original loss is commented out and the SSIM loss is currently implemented in the file

### change number of resnet blocks
1. In line 45 of the DeblurGAN/models/networks.py file you can change the n_blocks parameter to choose how many resnet blocks the model trains on
2. Make sure this aligns with the number of blocks in the saved model you test or else you will get an error

Original README from the paper can be found in the DeblurGAN directory