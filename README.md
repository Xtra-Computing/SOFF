# The OARF Benchmark Suite: Characterization and Implications for Federated Learning Systems

This repository is the official implementation of The OARF Benchmark Suite: Characterization and Implications for Federated Learning Systems

## Requirements

To install requirements: enter a subdirectory (e.g. `sentiment`) and

```setup
pip install -r requirements.txt
```

To download the data required for training the model, please use these links:

* Sentiment Analysis:
    * [Amazon (subsampled)]()
    * [Imdb]()
* Chinese Character Recognition:
    * [CASIA]()
    * [HIT]()
    * Enter `chinese/` and run `bash preprocess.sh` to download and preprocess the datasets
* Federated Recommendation:
    * [MovieLens 1M]()
    * Enter `movie/` and run `bash preprocess.sh` to download and preprocess the dataset

## Training

To reproduce the results in the paper, enter a subdirectory (e.g. `sentiment`), run one of the following command:

| Experiments                                                          | Command                     |
| -------------------------------------------------------------------- |-----------------------------|
| **Sentimen Analysis**                                                |                             |
| &nbsp;&nbsp; Improvement of federated averaging                      |`./train_fedavg.sh`          |
| &nbsp;&nbsp; Communication cost                                      |`./train_comm_cost.sh`       |
| &nbsp;&nbsp; Differential privacy                                    |`./train_dp.sh`              |
| &nbsp;&nbsp; Secure multiparty compuatation                          |`./train_smc.sh`             |
| **Chinese Character Recognition**                                    |                             |
| &nbsp;&nbsp; Improvement of federated averaging                      |`bash train_fedavg.sh`       |
| &nbsp;&nbsp; Communication cost                                      |`bash train_comm_cost.sh`    |
| &nbsp;&nbsp; Differential privacy                                    |`bash train_dp.sh`           |
| &nbsp;&nbsp; Secure multiparty compuatation                          |`bash train_smc.sh`          |
| **Federated Recommendation**                                         |                             |
| &nbsp;&nbsp; Improvement of federated learning                       |`bash train.sh`              |

## Evaluation

The model is automatically evaluated after training. You can find the logs in the `results_*/` directories.

<!-- ## Pre-trained Models

You can download pretrained models here:

* [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. -->

## Results

Our model achieves the following performance on :

| Horizontal task name, data and setup              | Accuracy  |
| --------------------------------------------------|-----------|
| **Sentimen Analysis**                             |      85%  |
| &nbsp;&nbsp; imdb                                 |           |
| &nbsp;&nbsp; combined                             |           |
| &nbsp;&nbsp; fedavg                               |           |
| &nbsp;&nbsp; fedavg, with SMC                     |           |
| **Chinese Character Recognition**                 |           |
| &nbsp;&nbsp; CASIA-HWDB1.1                        |      93.7%|
| &nbsp;&nbsp; HIT-OR3C                             |      77.5%|
| &nbsp;&nbsp; combined                             |      94.8%|
| &nbsp;&nbsp; fedavg                               |      95.4%|
| &nbsp;&nbsp; fedavg, with SMC                     |      95.8%|
| --------------------------------------------------|-----------|
| Vertical task name, data and setup                | MSE       |
| --------------------------------------------------|-----------|
| **Federated Recommendation**                      |           |
| &nbsp;&nbsp; Rating                               |      0.755|
| &nbsp;&nbsp; Rating + Auxiliary                   |      0.720|