# RandAugment Experiment README
## Cloning
The Github repo we are using uses Git LFS. See here: https://git-lfs.github.com/ and here: https://www.atlassian.com/git/tutorials/git-lfs

To clone this repo, you need to do git lfs clone, not just git clone. If you try to do a normal git clone without lfs installed, the model and classification data directories will look all wrong. If you don't want to deal with this, we recommend just downloading the .zip file.
## Required Libraries
See requirements.txt. `pip install -r requirements.txt` should be all you need to do.
## Directory Structure Overview
All scripts are written to be run from the root directory. converter.py and training_mobilenet_v2.py are the main files. PerformanceTest.py and training_efficientnet_b1.py are deprecated and kept for source control purposes, do not use.

The Classification_data_blah directories store augmented and unaugmented versions of the datasets converted into RGB .png files. These serve as inputs to our model.

Classification_data_raw is the raw, unprocessed dataset provided to us, still in .mhd and .raw format. It is a hardcoded input to converter.py.

Classification_data_unaugmented is the unaugmented dataset used to prepare the report.

Classification_data_augmented_new is the augmented dataset used to prepare the report.

Classification_data_augmented is deprecated and contains images which have invalid augmentations for our use case.

Classification_data_unaugmented_balanced was briefly used to test if the data imbalance problem could be solved by deleting some training data. It was not used to prepare the report, and is not recommended for use. Initial trials showed this left too little data available to be useful for training.

The model checkpoints used in the report are stored in models/mobilenet_augmented_100ep/ and models/mobilenet_unaugmented_100ep/. The rest are kept for source control reasons, do not use. They may not even work with the current versions of the scripts.

automl dir contains lightly modified code from  https://github.com/google/automl/. They provide the RandAugment implementation we have used.

## Script Overview
### converter.py
converter.py augments and converts the dataset into RGB .png files. It works in the Classification_data_raw directory; the new files will be written to that directory, which can then be passed to the training_mobilenet_v2.py script. To prevent data augmentation from occurring, set should_augment = False in the flie.

Warning. This script is fragile. It can only be run from the root directory of this structure, since it requires automl.autoaugment to be accessible. Also, it checks whether an image is Healthy or Diseased simply by checking if those strings are in the full file path, and it will augment images if Training is in the file path. If this folder is located in a place where Healthy, Diseased, or Training are in the full path, bad things will happen.

I assume you may want to test the model on some new data. To do this, replace Classification_data_raw with another directory structure in the same format. That is, Classification_data_raw should contain two directories, Training and Testing1; each of those should then contain two directories, Diseased and Healthy; and then those directories should contain .mhd files. Then, just run `python converter.py` and you should be able to pass the new Classification_data_raw folder to the script below.

### training_mobilenet_v2.py [mode] [model_path] [ds_path]

The mode parameter should be either 'train' or 'test' (without quotes). 

The model_path parameter should be where model checkpoints should be saved or loaded from. Note: A trailing slash is important. See example commands below. 

The ds_path parameter is the directory containing the data to train or test on. See converter.py documentation or Classification_data_augmented_new for an example. Remember, this script operates on .png files, so converter.py must be run first.

###
Example Commands:

Test our best augmented-trained model on the dataset given to us. Output is the prediction vectors for each image in format [Diseased Probability, Healthy Probability]. The list displayed at the bottom is a list of metrics: Loss, accuracy, AUC, F1 score diseased, then F1 socre healthy.
```
python training_mobilenet_v2.py test models/mobilenet_augmented_100ep/ Classification_data_augmented_new/
```

Do the same, but with the unaugmented model (the dataset dir can be the same, because testing images obviously are not augmented):
```
python training_mobilenet_v2.py test models/mobilenet_unaugmented_100ep/ Classification_data_augmented_new/
```

Training commands we used in our report:
```
python converter.py
python training_mobilenet_v2.py train models/mobilenet_augmented_100ep/ Classification_data_augmented_new/
python training_mobilenet_v2.py train models/mobilenet_unaugmented_100ep/ Classification_data_unaugmented/
```