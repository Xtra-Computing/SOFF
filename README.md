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

## Training

To reproduce the results in the paper, enter a subdirectory (e.g. `sentiment`), run one of the following command:

| Experiments                                                          | Command                     |
| -------------------------------------------------------------------- |-----------------------------|
| **Sentimen Analysis**                                                |                             |
| &nbsp;&nbsp; Improvement of federated averaging                      |`./train_fedavg.sh`          |
| &nbsp;&nbsp; Communication cost                                      |`./train_comm_cost`          |
| &nbsp;&nbsp; Differential privacy                                    |`./train_dp`                 |
| &nbsp;&nbsp; Secure multiparty compuatation                          |`./train_smc`                |
| **Chinese Character Recognition**                                    |                             |
<!--TODO-->
| **Federated Recommendation**                                         |                             |
<!--TODO-->

## Evaluation

The model is automatically evaluated after training. You can find the logs in the `results_*/` directories.

<!-- ## Pre-trained Models

You can download pretrained models here:

* [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. -->

## Results

Our model achieves the following performance on :

| Task name, data and setup                         | Accuracy  |
| --------------------------------------------------|-----------|
| **Sentimen Analysis**                             |      85%  |
| &nbsp;&nbsp; imdb                                 |           |
| &nbsp;&nbsp; combined                             |           |
| &nbsp;&nbsp; fedavg                               |           |
| &nbsp;&nbsp; fedavg, with SMC                     |           |
| **Chinese Character Recognition**                 |           |
<!--TODO-->
| **Federated Recommendation**                      |           |
<!--TODO-->
