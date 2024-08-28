# HDR-AE
Code for paper `Automatic Exposure Strategy Network for Robust Visual Odometry in Environments with High Dynamic Range`
## CODE
```
conda create --name myenv python=3.9
conda activate myenv
pip install -r requirement.txt
```
1. To select images using the proposed image information metric on [Shin's dataset](https://github.com/UkcheolShin/Noise-AwareCameraExposureControl) and run pose estimation based on the selected images
```
python select_images.py --dataset-dir DATASET_FOLDER

python test_selected_images.py --selected-result ./selected_image_info.txt --output-dir OUTPUT_FOLDER
``` 
2. To train the model please download the [dataset](https://drive.google.com/drive/folders/1aTjSEuPMKvv19RZQatH6E0kyJoQeUn8I?usp=sharing) into folder[hdr-ae-data](./hdr-ae-data/), and run

```
python run.py --stage train --data-path ./hdr-ae-data --NM --joint-learning
```
## DATASET
Dataset can be found in [Google drive](https://drive.google.com/drive/folders/1aTjSEuPMKvv19RZQatH6E0kyJoQeUn8I?usp=sharing)

