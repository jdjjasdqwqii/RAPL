# RAPL: Towards Few-Shot Out-of-Distribution Detection via Region-Aware Prompt Learning

## Pipeline

<p align="center">
  <img src="https://github.com/user-attachments/assets/40b255c8-eb02-48b0-bbba-9138248b18fe" width="800">
</p>

Figure 2: Overview of the proposed method. Our framework combines global and local prompts to guide region-level understanding. Global prompts guide region generation (e.g., via Grounded SAM), while local prompts adapt to specific regions for fine-grained recognition and OOD detection.

## Installation

```bash
pip install -r requirements.txt
```


##  Datasets

Please follow instructions in the [CoOp](https://github.com/KaiyangZhou/CoOp) repository to download the datasets used for the few-shot image classification and domain generalization experiments. Note that downloading the class names and splits (e.g. `split_zhou_Caltech101.json`) is not required as they are already included in the `gallop/datasets` folder.  

For the out-of-distribution experiments, we use the following datasets curated by Huang et al. (2021):  

- [iNaturalist](https://github.com/visipedia/inat_comp)  
- [SUN](https://vision.princeton.edu/projects/2010/SUN/)  
- [Places](http://places2.csail.mit.edu/)  
- [Texture (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/)  

Please follow instructions from the [repository](https://github.com/deeplearning-wisc/large_scale_ood#out-of-distribution-dataset) to download the datasets.
The overall file structure is as follows:

The directory structure where all the datasets are stored should look like this:
```bash
$DATA_FOLDER/
└–––-Imagenet/
│    ├–––––train/
│    └––––val/
├––––sketch/
├––––caltech101/
│    └––––101_ObjectCategories/
├––––EuroSAT_RGB/
├––––dtddataset/
│    └–––dtd/
│        └––––images/
├––––fgvc-aircraft-2013b/
├––––flowers102/
├––––food-101/
│    └–––images/
├––––oxford-iiit-pet/
│    └–––images/
├––––stanford_cars/
├––––SUN397/
├––––UCF-101-midframes/
└––––ood_data/
     ├–––iNaturalist/
     ├–––SUN/
     ├–––Places/
     └–––dtd/
         └––––images/
```
## Training
We provide training scripts in the scripts folder. For instance, to launch the training on Imagenet with the ViT-B/16 backbone and with 16 shots do:
```bash
./scripts/run_imagenet.sh
```
## Few-shots classification results (ViT-B/16, 16 shots)
<img width="1368" height="397" alt="image" src="https://github.com/user-attachments/assets/ac4013f4-575d-4bd3-b02e-2985feed816e" />
Table 1: Comparison of ID Top-1 accuracy with SOTA prompt learning approaches across benchmark datasets.
## Imagenet out-of-detection results (ViT-B/16, 16 shots)
<img width="1375" height="807" alt="image" src="https://github.com/user-attachments/assets/45053ce7-b389-4ba8-829e-5894f0681302" />
Table 2: Comparison of OOD detection performance with SOTA approaches across four OOD datasets. We report FPR95↓ and
AUC↑. Lower FPR95 and higher AUC indicate better OOD detection.







