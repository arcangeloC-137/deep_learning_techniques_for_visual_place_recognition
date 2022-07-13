# ***Deep Learning Techniques for Visual Place Recognition***
## **Machine Learning and Deep Learning - Politecnico di Torino**

Visual Geolocalization (VG) or Visual Place Recognition (VPR) consists in determining the location where a given query photograph was taken. This task is performed using image matching and retrieval methods on a database of images with known GPS (Global Positioning System) coordinates.
Most recent works on VG still face several challenges in building models robust to night domain, occlusion, perspective changes, and require effective methodologies to combine features extracted from different models.

In this work we propose several approaches to face some of these issues: to begin with, data augmentation techniques are used to improve robustness against changes in perspectives and occlusions. Afterwards, a night domain study is performed in two different ways: the first via a 'smart' data augmentation, modifying images parameters like brightness and pixel colors, and the latter through the creation of synthetic
night images with the use of UNIT networks. This last technique in particular allows us to drastically improve the results on specific datasets. Eventually, multiscale testing and ensembles methods are adopted to further improve the results.

---

This repository contains the code for our project "*Image Retrieval for Visual Geolocalization*", for the Machine Learning and Deep Learning course @ Politecnico di Torino, A.Y. 2021/22. Please refer to the relative [report](https://github.com/arcangeloC-137/deep_learning_techniques_for_visual_place_recognition/blob/e3d87f3affe5ac569dafc9b720d79990d85e0e47/report/Project2_Group1_Report.pdf) for a complete description of the expertiment.

---

After cloning this repository, is recommended to first organize the code and the datasets in the following tree structure:

```
.
├── benchmarking_vg
└── datasets_vg
    └── datasets
        └── pitts30k
            └── images
                ├── train
                │   ├── database
                │   └── queries
                ├── val
                │   ├── database
                │   └── queries
                └── test
                    ├── database
                    └── queries
```
The [datasets_vg](https://github.com/gmberton/datasets_vg) can be used to download the datasets used for this project, while the [benchmarking_vg](https://github.com/gmberton/deep-visual-geo-localization-benchmark) can be used to download the vanilla version of the code.

## Running the Experiments
The work is composed by the following steps:
1. Preliminary baseline experiments and dataset visualization
2. Data Augmentation for Night Domain robustness:
   - 'Smart' Data Augmentation
   - Synthetic Images creation
3. Data Augmentation for Occlusions and Perspective Changes
4. Ensembles
5. Multi-scale Testing

For the sake of simplicity we provide a complete guide to run the the whole experiments in [deep_learning_techniques_for_visual_place_recognition.ipynb](https://github.com/arcangeloC-137/deep_learning_techniques_for_visual_place_recognition/blob/main/jupyter_notebook_guide/deep_learning_techniques_for_visual_place_recognition.ipynb).

---
Contributors to this project are: Atadjanov Olloshukur, Frigiola Arcangelo, and Scoleri Maria Rosa.
