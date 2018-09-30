There are two data formats of training data to train the model, 1) the original format by the author of AlignedReID-Re-Production-Pytorch. 2) a simpler format with just subfolders of different IDs.

For format 1) by AlignedReID-Re-Production-Pytorch
# prepare your training data
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
