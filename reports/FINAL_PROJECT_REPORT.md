# Final Report - Poultry Disease Classification from Fecal Images
Run `20261115_145045` | selected model **EfficientNetB0** | split: REBUILT (group-aware)

## 1. Data and protocol
6,812 readable images indexed; split sizes train/val/test = 4785/1044/983.
Group-aware / leakage-audited (exact MD5 and perceptual-hash near-duplicates). Class weights (not oversampling) handle imbalance. Augmentation is applied to training batches only.

## 2. Validation benchmark
| Model          |   Macro_F1 |   Balanced_Acc |   Accuracy |   Recall_Newcastle_Disease |   Latency_ms |   Params_M |   Size_MB |
|:---------------|-----------:|---------------:|-----------:|---------------------------:|-------------:|-----------:|----------:|
| EfficientNetB0 |     0.9812 |         0.9828 |     0.9875 |                     0.9649 |      81.5597 |     4.3847 |   43.4146 |
| DenseNet169    |     0.9807 |         0.9842 |     0.9847 |                     0.9825 |     239.44   |    13.0778 |   92.7318 |
| ResNet50       |     0.9793 |         0.9878 |     0.9847 |                     1      |     214.208  |    24.1225 |  227.843  |
| MobileNetV2    |     0.9297 |         0.9178 |     0.9559 |                     0.7719 |      46.9299 |     2.5931 |   26.1676 |

## 3. Held-out test result (EfficientNetB0, evaluated once)
* Accuracy **98.47%** (95% CI 97.66-99.19)
* Macro-F1 **0.9765** (95% CI 0.9627-0.9882); balanced accuracy 98.15%
* Newcastle-disease recall 96.61% ; ECE 0.0065

## 4. Why numbers can be lower than published values
Paper-style protocol comparison was not run.

| Paper                | Data                            | Best reported                          | Comparability note                                                                                                                                  |
|:---------------------|:--------------------------------|:---------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 Machuve 2022       | 6,812 farm + 1,255 PCR-labelled | 98.24% (Xception, FT + frozen BN)      | Test set = farm-labelled images (expert annotation, not PCR). Original dataset paper.                                                               |
| 2 Degu 2023          | 8,067 -> 10,500 (ROS + aug)     | 98.70% (ResNet50 on YOLOv3 ROI)        | Oversampling/augmentation before training; order vs. split not stated in our summary - verify in paper.                                             |
| 3 Islam 2025         | 8,067 -> 12,000 (aug)           | 97.04% (RF-selected features + ANN)    | 5-fold CV; same augmentation-order question.                                                                                                        |
| 4 Alam 2026          | 4,090 (different, smaller set)  | 98% (tuned DenseNet169)                | Different dataset - not directly comparable.                                                                                                        |
| 5 Luo 2025           | 6,812 (aug to ~2,100/class)     | 98.3% (ResNet152 + SVM)                | Frozen CNN features + classical classifier.                                                                                                         |
| 6 Shanmugapriya 2026 | 6,812 + 14,618 (Nigeria)        | 92.0% (SPFRNN)                         | Two-level system + IoT simulation; different task.                                                                                                  |
| 7 Dhungana 2025      | 6,812 (70/20/10)                | 99.12% (YOLO11 crop + EfficientNet-B0) | Classifies YOLO-cropped regions; the report notes no field test.                                                                                    |
| 8 Anwarul 2025       | 8,067 -> 10,500                 | 98.4% (DeepFowl)                       | Oversampling algorithm; 120x120 input.                                                                                                              |
| 9 Lee 2026           | 8,420 (balanced ~2,100/class)   | 98.89% (Fecal-DANet)                   | Class counts imply augmentation (Zenodo NCD has only ~562 originals) - verify split order.                                                          |
| 10 Tasdelen 2025     | 6,812 -> 9,068                  | 97.1% (MobileNetV2, 96x96)             | 5-fold CV; frozen backbone.                                                                                                                         |
| 11 Hossain 2023      | 1,080 (270/class)               | 92.8% on heterogeneous farm test       | Drops from 99% validation to 92.8% on new farms/phones: the realistic figure.                                                                       |
| 12 Ahmed 2025        | 14,879 -> 20,000                | 99.67% (YOLOv8)                        | Merges two datasets that may overlap (8,067 = 1,255 + 6,812); the P/R/Acc/F1 columns in the report rise in equal 0.0011 steps - treat with caution. |
| 13 Al Huda 2024      | 6,812 (5448/682/682)            | 97.78% (EffNetV2B0 + DenseNet121)      | Closest protocol to ours.                                                                                                                           |
| 14 Harshitha 2024    | microscopy, binary              | 96.5% (VGG)                            | Different modality (microscope images) - not comparable.                                                                                            |

## 5. Novelty - poultry vs other-animal rejection
CLIP zero-shot gate (no training data) + Mahalanobis feature-space gate (poultry data only) + confidence gate.
Non-poultry rejection rate NOT measured (no evaluation folder). Genuine poultry test images passing the gate: 94.8%.

## 6. Limitations
Single data source (two Tanzanian regions); farm labels are expert-annotated, not PCR-verified; same-session/same-bird photos cannot be detected without metadata;
CLIP cannot reliably separate chicken from turkey/duck droppings; gate rejection rates are only as valid as the non-poultry evaluation images; no prospective field study.
Predictions must assist, not replace, veterinary diagnosis.
