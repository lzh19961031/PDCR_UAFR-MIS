# DermClinical: Clinical-Oriented Dataset and Evaluation for Computer-Aided Dermatological Diagnosis  

<!--- --->
Code for paper [DermClinical: Clinical-Oriented Dataset and Evaluation for Computer-Aided Dermatological Diagnosis] 



<div align="center"><img width="700" src="./images/dataset.jpg"/></div>


## Abstract:

Dermatosis is one of the most common diseases and requires a heavy workload of diagnosis for its vast number of cases. Despite the development of computer-aided diagnosis, especially deep learning based computational methods, misalignment exists between evaluation of benchmark studies in the literature and expectation of work flows in real clinical practice. In this paper, we systematically address the problem from perspectives of dataset, metric, as well as evaluation schemes, collectively denoted as an evaluation
framework DermClinical. Specifically, we build a so-far largest dataset, with a special focus on the long-tailed disease distribution by ensuring that all categories have sufficient samples for reliable evaluations. On this basis, we propose a novel metric which, as opposed to common classification based metrics, directly measures the labors saved by a computational model for doctors. Finally, we design a limited-data evaluation scheme to simulate situations when unseen categories are encountered in real practice. Extensive experiments are conducted to evaluate and compare current computational methods. The dataset, metric and code will be released to facilitate the research


## Contribution: 

- We build a new dataset on clinical images named “DermClinical-34k” from a variety of sources. The dataset, in a total of 34,671 images, is as far
as we know the largest among existing ones. To tackle the problem of limited samples or even potentially unseen for some tail categories in long-tailed distribution, two strategies are applied. Firstly, we combine clinically related categories if any individual one does not have enough samples (which is defined in this work as less than 200 samples, but can be also set to a stricter standard as per needs). Secondly, we introduce the concept of the “unknown” category which, on the one hand, merges the rest sample insufficient categories, and on the other hand, offers the opportunity for models to deal with unseen categories: cases for unseen categories are expected to be classified into the “unknown” category. In the end, the dataset contains 56 categories as well as the “unknown” category, none of which has a sample capacity less than 200.

- We propose a novel metric named “Overall-Recall-above-Precision-K (OR-K)” that directly measures the labors saved by a computational model for doctors. The metric is derived from the conventional pipeline where the computational model serves for preliminary screening and conclusions are double-checked by doctors. Since model performance is not uniform across categories, a central question is to differentiate a reliable category which has an acceptable error rate by doctors against the unreliable category - only predictions for reliable categories are trusted and corresponding labors are saved. The
differentiation is proposed to be based on the threshold over the precision for the category. In this paper, the threshold is set to be the same value K, but can be straightforwardly extended to varied per-category thresholds. Given the definition of the reliable and unreliable category, the metric then measures the saved labors by the ratio of samples predicted as reliable categories over all testing samples.

- We design a limited-data evaluation scheme besides the ordinary full-data evaluation scheme. This is to simulate situations for meeting unseen categories in daily clinical usage. To this end, we manually remove a certain
number of categories in the trainset, but still leave the corresponding categories in the testset. Note that we do not intend to investigate the zero-shot setting, but expect computational models to detect such out-of-scope cases and distinguish them as the “unknown” category instead of misclassification. In this way we aim at evaluation on the robustness of computational models when unseen categories are encountered in real practice.

- We conduct extensive experiments on both off-the-shelf computational methods in the literature which are dedicatedly designed for dermatological image classification, as well as general-purpose classification backbones equipped with our filtering strategy as post-processing. The results demonstrate the independence of the proposed OR-K metric in characterizing complementary aspects from classification based metrics, the importance of the concept of “unknown” category for avoiding risky predictions and saving more labors for doctors, and the challenges for the application of computational models in clinical practice, especially by out-of-scope samples simulated by the limited-data evaluation scheme. Further methodology innovations and investigations are worthy in the future.


## Experiments


### Benchmark

#### Full-Data scheme
- Performance comparison in the full-data evaluation scheme. 
<div align="center"><img width="700" src="./images/benchmark_fulldata.jpg"/></div>

#### Limited data scheme
- Performance comparison in the limited-data evaluation scheme.
<div align="center"><img width="700" src="./images/benchmark_limited_data.jpg"/></div>

### Investigations
#### Correlations between OR-K and Other Metrics
- Scatter diagrams to show the correlations between OR-K and F1, Mauc, Acc respectively. Correlation coefficient is also calculated.
<div align="center"><img width="400" src="./images/ORKandothermetric.jpg"/></div>

#### Influence of Setting of K
- Investigation in the variation of the OR-K values with respect to different threshold K. 
<div align="center"><img width="400" src="./images/differentK.jpg"/></div>



#### Ablation Studies for Models
- Performance comparison of different filtering strategies as well as vanilla classification without filtering. 
<div align="center"><img width="400" src="./images/ab_models.jpg"/></div>



## Usage: 
### 2D experiments:


#### Main requirements

  * **torch >= 1.1**
  * **torchvision >= 0.3**
  * **Python 3.6**
  * **16 NVIDIA GPUs**

#### Preparing Data
1. Download DermClinical-34k. Then use our DermClinical-34k dataset as the default structure:

```
data
└── Acne_Vulgaris(large)
    └── 0.jpg
    └── 1.jpg
    └── ...
└── Background
    └── 0.jpg
    └── 1.jpg
    └── ...
...
```

#### Training and validation

Run "run_training.py" for training or validation (using the *.yaml* configuration files in `./configs/`).

Here is an example for resnet101 backbone: 
```   
# General setting
epochs: 300                                       # the number of epoch for training 
batch_size: 4                                     # batch size
num_workers: 4
syncbn: True
mc: True
seed: 666

# Loss function
criterions: [['ce', 1]]         
loss_temperature: 1

# The information of model      
model:
    mode: 'resnet101'                                    # resnet101
    baseline_discard_only_low_confidence: False          # threshold for highest category
    baseline_discard_only_low_confidence_threshold: 0.9  

    baseline_discard_unknown: False                      # threshold for unknown category
    baseline_discard_unknown_threshold: 0.8

    input_channel: 3
    num_classes: 57                                      #number of category

    precision_threshold: 0.9                             #K
    recall_threshold: 0.6

# The information of dataset
dataset:
    mode: 'Recollected_dataset'
    root_path: './data/'
    input_size: [224, 224, 3]
    preload: False
    aug: False
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
    modality: '2D'

# The information of data augmentation
data_aug:
    affine:
      scale: [0.8, 1.2]
      translate: [0.1, 0.1]
      rotate: [-90, 90]
      shear: [-20, 20]
    flip: [0.5, 0.5]

# The information of validation
test:
  rank_metric: ['overall_recall_at_precision_K_top1']  # main metric used for ranking
  metric_used: ["overall_recall_at_precision_K_top1_test", "pre", "rec", "spe", "f1", "geo", "iba", "sup"]  # 'mDice', 'mIoU', 'fwIoU', 'Acc', 'Acc_class'
```

#### Testing

Similar with training, run "run_test.py" (using the *.yaml* configuration files in `./configs/`).


