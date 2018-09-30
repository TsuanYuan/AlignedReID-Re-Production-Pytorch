There are two data formats of training data to train the model, 1) the original format by the author of AlignedReID-Re-Production-Pytorch. 2) a simpler format with just subfolders of different IDs.

For format 1) by AlignedReID-Re-Production-Pytorch
## prepare your training data
Assume your folder structure is as following:

```console
folder_dir --> 00001

         --> 00002
         
         ...
         
         --> 00100
```

where each of the "00xxx" is a subfolder with crops of the same person of ID=00xxx.
With the script 

```console
python transform_folder.py folder_dir formatted_dir --num_test 0 --num_folds 1 
```
You will get a folder "formatted_dir" for the training script. The create a txt file with the path to this formatted_dir
content of "train.txt":

```sh
/path/to/formatted_dir
```

## training with the AlignedReID-Re-Production-Pytorch scripts

```console
python script/experiment/train_ml.py -d '[[6,7]]' --dataset customized --ids_per_batch 32 --ims_per_id 8 normalize_feature true -gm 0.5 -lm 0.5 -glw 1.0 -llw 0 -idlw 0 -gdmlw 0.0 -ldmlw 0.0 -pmlw 0.0 --base_lr 1e-4 --lr_decay_type exp --exp_decay_at_epoch 100 --total_epochs 300 --exp_dir /path/to/model_folder/ --crop_prob 0.5 --crop_ratio 0.9 --num_models 1 --customized_folder_path_file train.txt --base_model resnet50
```
