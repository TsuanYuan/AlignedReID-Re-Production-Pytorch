There are two data formats of training data to train the model, 1) a simpler format with just subfolders of different IDs. 2) the original format by the author of AlignedReID-Re-Production-Pytorch. 


## For format 1) which is a simple "root_dir" with subfolders:
```console
root_dir --> 00001

         --> 00002
         
         ...
         
         --> 00100
```
where each of the "00xxx" is a subfolder with crops of the same person of ID=00xxx.
To train the model:
```console
 python train_multi_folder.py /ssd/qyuan/data/appearance/wcc/train_folders.txt  /path/to/output/model.pth --sample_size 8 --batch_size 64 --lr 0.0001 --margin 0.3 --num_epoch 300 --optimizer adam --gpu_ids 4 5 6 7 --loss triplet
```
the first argument "/ssd/qyuan/data/appearance/wcc/train_folders.txt" lists training sets in rows. You only need one row if you only have one training set. Though it supports training simultaneously on multiple training sets.
```console
/path/to/root_dir
```

## For format 2) by AlignedReID-Re-Production-Pytorch
### prepare your training data
Assume your folder structure is as the above "root_dir":


```console
python transform_folder.py root_dir formatted_dir --num_test 0 --num_folds 1 
```
You will get a folder "formatted_dir" as the output. Then create a txt file "train.txt" with just one row of the path to this formatted_dir
content of "train.txt":
```sh
/path/to/formatted_dir
```

### training with the AlignedReID-Re-Production-Pytorch scripts

```console
python script/experiment/train_ml.py -d '[[6,7]]' --dataset customized --ids_per_batch 32 --ims_per_id 8 normalize_feature true -gm 0.5 -lm 0.5 -glw 1.0 -llw 0 -idlw 0 -gdmlw 0.0 -ldmlw 0.0 -pmlw 0.0 --base_lr 1e-4 --lr_decay_type exp --exp_decay_at_epoch 100 --total_epochs 300 --exp_dir /path/to/model_folder/ --crop_prob 0.5 --crop_ratio 0.9 --num_models 1 --customized_folder_path_file train.txt --base_model resnet50
```

#### sample triplet at frame intervals
There is a utils function "get_sample_within_interval" in the trainer of format 2 which imposes a constraint that all same pairs are within a time interval (or frame interval)

https://github.com/TsuanYuan/AlignedReID-Re-Production-Pytorch/blob/experts/aligned_reid/dataset/TrainSet.py#L66
