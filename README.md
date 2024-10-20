# RSNA 2024 Lumbar Spine Degenerative Classification
9th place solution training and validation codes for [RSNA 2024 Lumbar Spine Degenerative Classification](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification)

## Environment
Create the conda environment
```
conda env create -f environment.yml
```
Install the rsna2024 package in the environment
```
pip install -e .
```

Training was performed using a single NVIDIA GTX A6000 GPU (48GB VRAM).

## Data
Download the data from Kaggle
```
kaggle competitions download -c rsna-2024-lumbar-spine-degenerative-classification
```
and extract into `data/raw`

## Training
Run the following for training all models:
```
python main.py
```
The configuration files for model training can be found in the `experiments` folder.

Trained weights and configuration files will be saved in the `model/<model_name>` folder. If Wandb is enabled, the model names are randomly generated, otherwise they consist of the model type and a timestamp. Checkpoints with the `_best` postfix contain the weights for the best epoch (based on validation loss) and were used for all models.

## Inference
See the [inference notebook](https://www.kaggle.com/code/adamnarai/rsna2024-two-stage-split-global-3-base?scriptVersionId=199792909) with the trained model weights.

For OOF validation, first run
```
python rsna2024/preproc/generate_coords.py
```
then model performance can be investigated using `notebooks/combined_ensemble_model_pred.ipynb`

# Model details

Stage 1: Gaussian heatmap-based keypoint detection using DeepLabV3Plus, with separate models for each of the three series types

Stage 2: Level-wise ROI classification using an ensemble of 2.5D models with GRU head and ResNet18/Swin-Tiny/ConvNeXt-Nano bases

## Keypoint detection

I created a gaussian heatmap for each coordinate and trained DeepLabV3Plus models with resnet34 encoder separately for the three series types (Sag T2, Sag T1, Axi). Coordinates were then defined as the argmax of each predicted map. The inputs were 5, 3 and 3 slices (as channels), respectively for the three series types resized to 512x512 pixels and intensity normalised. Slices were selected from the middle for Sag T2 series and from predetermined mm positions relative to the middle for Sag T1 series. For the Axi series the relevant Sag T2 coordinate was projected into the axial series space and slices were selected around the one closest to this coordinate. I used rotation, sheer, channel shuffle (p=0.5), and one of sharpening or motion blur (p=0.5) augmentations. Using AdamW optimizer with 0.001 base learning rate, batch size 16 and CosineAnnealingLR scheduler I trained the three models for 30, 20 and 10 epochs, respectively with MSE loss and 5-fold cross validation.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8375965%2F933188b82687f65f63b0028409e40850%2Frsna2024_seg_model2x.png?generation=1728553648423715&alt=media)

## Classification

I extracted 50 mm x 50 mm ROIs centred on the coordinates with 5 slices (as channel) resized to either 128x128 pixels (ResNet18 base) or 224x224 pixels (Swin-Tiny, and ConvNeXt-Nano bases) and normalised their intensity. Slices were selected the same way as for keypoint detection, only with level-dependent mm positions for Sag T1 in this case. I used rotation, sheer, and one of sharpening or motion blur (p=0.5) augmentations. I also used channel shuffle (p=0.5) only for the Axi series, since despite using a recurrent network head, it significantly improved performance. The classifiers were 2.5D models based on resnet18, swin\_tiny and convnext\_nano feature extractors with GRU head. Using AdamW optimizer with 0.001 or 0.00003 base learning rate, batch size 16 and StepLR or CosineAnnealingLR scheduler I trained the models for 3-5 epochs with cross entropy loss and 5-fold cross validation (optimal parameters depended on model type). I used two approaches: “split” models with separate one-output models for each core condition (spinal canal stenosis, neural foraminal narrowing and subarticular stenosis) and “global” models with outputs for all five conditions (including the two sides).

### Split models

Spinal: Using Sag T2 and Axi (centred on mean of left and right coordinates) ROIs.  
Foraminal: Using Sag T1 ROIs, same model for both sides.  
Subarticular: Using Axi ROIs, same model for both sides, but right side images are mirrored.  
Predictions were obtained for each level and side using these three models and concatenated to make the 25 outputs per study.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8375965%2Ff3c84f957b2678f463e92ea32f40d6e9%2Frsna_2024_split_model2x.png?generation=1728553678267138&alt=media)

### Global model

Sag T2, left/right Sag T2 and left/right (right mirrored) Axi ROIs were used as inputs to the model, resulting in predictions for all 5 conditions at a given level. The same feature extractor model was used for both left and right side.  
Predictions were obtained for each level and concatenated to make the 25 outputs per study.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8375965%2F094459632507cb2c3512e4ef9baf3300%2Frsna2024_global_model2x.png?generation=1728553689737251&alt=media)

## Submission

My final submission was an ensemble of 6 models, combining both split and global models with three different feature extractors ResNet18, Swin-Tiny, and ConvNeXt-Nano. Each of these models was further a 5-fold ensemble, and the final predictions were the simple mean of these ensembles.
