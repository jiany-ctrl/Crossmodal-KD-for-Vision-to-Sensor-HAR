# Crossmodal KD 

Cross-Modal Knowledge Distillation For Vision-To-Sensor Action Recognition (ICASSP-2022): https://ieeexplore.ieee.org/document/9746752

# Introduction

Human activity recognition (HAR) based on multi-modal approach has been recently shown to improve the accuracy performance of HAR. However, restricted computational resources associated with wearable devices, i.e., smartwatch, failed to directly support such advanced methods. To tackle this issue, this study introduces an end-to-end Vision-to-Sensor Knowledge Distillation (VSKD) framework. In this VSKD framework, only time-series data, i.e., accelerometer data, is needed from wearable devices during the testing phase. Therefore, this framework will not only reduce the computational demands on wearable devices, but also produce a learning model that closely matches the performance of the computational expensive multi-modal approach. In order to retain the local temporal relationship and facilitate visual deep learning models, we first convert time-series data to two-dimensional images by applying the Gramian Angular Field (GAF) based encoding method. We adopted multi-scale TRN with BN-Inception and ResNet18 as the teacher and student network in this study, respectively. A novel loss function, named Distance and Angle-wised Semantic Knowledge loss (DASK), is proposed to mitigate the modality variations between the vision and the sensor domain. Extensive experimental results on UTD-MHAD, MMAct, and Berkeley-MHAD datasets demonstrate the competitiveness of the proposed VSKD model which can be deployed on wearable devices.



## Citation
If you find this repository useful, please consider citing the following paper:


```
@inproceedings{chen2022simkd,
  title={Knowledge Distillation with the Reused Teacher Classifier},
  author={Chen, Defang and Mei, Jian-Ping and Zhang, Hailin and Wang, Can and Feng, Yan and Chen, Chun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11933--11942},
  year={2022}
}
```
```
@inproceedings{chen2021cross,
  author    = {Defang Chen and Jian{-}Ping Mei and Yuan Zhang and Can Wang and Zhe Wang and Yan Feng and Chun Chen},
  title     = {Cross-Layer Distillation with Semantic Calibration},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  pages     = {7028--7036},
  year      = {2021},
}
```

