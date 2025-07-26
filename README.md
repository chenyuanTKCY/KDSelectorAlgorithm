<p align="center">
<img width="150" src="./fig/KD_Logo.png"/>
</p>


<h1 align="center">KDSelectorAlgorithm
</h1>
<h2 align="center"> A Framework of Knowledge-Enhanced and Data-Efficient Selector Learning for Anomaly Detection Model Selection in Time Series</h2>

We proposes a novel knowledge-enhanced and data-efficient framework for learning a neural network (NN)-based model selector in the context of time series anomaly detection (TSAD). It aims to address the limitations of existing model selection methods, which often fail to fully utilize the knowledge in historical data and are inefficient in terms of training speed.



## Framework

We introduce a novel neural network (NN)-based selector learning framework, which serves as the core component of our system. For a comprehensive understanding of its architecture and implementation, please refer to the detailed technical report available at
<!--
<iframe src="https://pdf-embed-api.com/?url=https://github.com/chenyuanTKCY/KDSelector/blob/master/app/fig/framework.pdf" width="100%" height="600px"></iframe> -->



## Installation

To install KDSelector from source, you will need the following tools:
- `git`
- `conda` (anaconda or miniconda)

#### Packages and tools setting

The following key tools and their versions are used in this project:
- **Python**
  - python==3.8.20

- **Machine Learning and Deep Learning**
  - scikit-learn==1.3.2
  - torch==1.13.

For the complete list of dependencies, please refer to the `environment.yml` and `requirements.txt` files.

#### Steps for installation

**Step 1:** Clone this repository using `git` and change into its root directory.

```bash
git clone https://anonymous.4open.science/r/KDSelectorAlgorithm-10E7.git
cd KDSelectorAlgorithm/
```

**Step 2:** Create and activate a `conda` environment named `KDSelector`.

```bash
conda env create --file environment.yml
conda activate KDSelector
```
> Note: If you plan to use GPU acceleration, please ensure that you have CUDA installed. You can refer to the [CUDA installation instructions](https://developer.nvidia.com/cuda-downloads) for guidance.

If you do not wish to create the `conda` environment, you can install only the dependencies listed in `requirements.txt` using the following command:
```
pip install -r requirements.txt
```
**Step 3:** Download the required datasets and weights from the following links (due to upload size limitations on GitHub, we host them on Google Drive):

- [Download datasets](https://drive.google.com/file/d/1PQKwu5lZTnTmsmms1ipko9KF712Oc5DE/view?usp=sharing)

  - Move the downloaded file `TSB.zip` into the `MSAD/data/` directory and unzip it.

- [Download datasets + preprocesed data](https://drive.google.com/file/d/1KBFzKE3Z-tUe_3KdI6gxnfjbMc0ampr6/view?usp=sharing)

  - Unzip the downloaded `data.zip` file and move its contents into the `MSAD/data/` directory. With this files you can skip the steps of generating the Oracles, and creating the proccesed datasets (windowed and features).

- [Download weights](https://drive.google.com/file/d/1qMfSTPXVT2XkwHkwuRE26mo19FKXrf60/view?usp=sharing)

  - Unzip the downloaded `weights.zip` file and move its contents (`supervised/` and `unsupervised/`) into the `MSAD/results/weights/` directory.

## Usage

Below, you will find a step-by-step guide on how to use our work. This includes the commands required to run the scripts along with a small explanation of what they do and the parameters they use. The values of the parameters in the scripts are just examples, and you can experiment with different values.

#### Compute Oracle

The Oracle is a hypothetical model that simulates the accuracy of a model on a given benchmark and evaluates its anomaly detection ability. You can simulate Oracle with different accuracy values, ranging from 1 (always selecting the best detector for a time series) to zero (always selecting a wrong detector). Additionally, you can simulate Oracle with different modes of randomness, namely:

1. **true**: When wrong, randomly select another detector.
2. **lucky**: When wrong, always select the second best detector (upper bound).
3. **unlucky**: When wrong, always select the worst detector (lower bound).
4. **best-k**: When wrong, always select the k-th best detector (e.g., best-2 is lucky).

To compute Oracle, run the following command:

```bash
python3 run_oracle.py --path=data/TSB/metrics/ --acc=1 --randomness=true
```

- path: Path to metrics (the results will be saved here).
- acc: The accuracy that you want to simulate (a float between 0 and 1).
- randomness: The randomness mode that you want to simulate (see possible modes above).

> The results are saved in _/MSAD/data/TSB/metrics/TRUE_ORACLE-100/_ (the name of the last folder should change depending on the parameters).


#### Compute Averaging Ensemble

The Averaging Ensemble, or Avg Ens, is used to ensemble the anomaly scores produced by all the detectors, by computing their average. Subsequently, the AUC-PR and the VUS-PR metrics are computed for the resulting score.

To compute Avg Ens, run the following command:

```bash
python3 run_avg_ens.py --n_jobs=16
```

- n_jobs: The number of threads to use for parallel computation (specify an appropriate value).

> This process may take some time :smile: (~16 mins in 16 cores and 32GB of RAM). The script will perform the following tasks:
>
> 1. Load all datasets from the TSB benchmark
> 2. Load all the scores for each time series and detector (~ 1800 \* 12 scores)
> 3. Compute the average score for each time series for 4 metrics (AUC-ROC, AUC-PR, VUS-ROC, VUS-PR)
> 4. Save the results in _/MSAD/data/TSB/metrics/AVG_ENS/_.

#### Data Preprocessing

Our models have been implemented to work with fixed-size inputs. Thus, before running any models, we first divide every time series in the TSB benchmark into windows. 

Note that you can add your own time series here and divide them into windows, but make sure to follow the same format.All the commands provided in this repository use a subsequence length of 128 as an example. However, you can modify this to suit your specific needs by selecting window sizes between 64, 128, 256, 512, 1024, and 2048.

To produce a windowed dataset, run the following command:

```bash
python3 create_windows_dataset.py --save_dir=data/ --path=data/TSB/data/ --metric_path=data/TSB/metrics/ --window_size=128 --metric=AUC_PR
```

- save_dir: Path to save the dataset.
- path: Path of the dataset to divide into windows.
- metric_path: Path to the metrics of the dataset provided (to produce the labels).
- window_size: Window size (if the window size is larger than the time series' length, that time series is skipped).
- metric: Metric to use for producing the labels (AUC-PR, VUS-PR, AUC-ROC, VUS-ROC).

The feature-based methods require a set of features to be computed first, turning the time series into tabular data. To achieve this, we use the TSFresh module, which computes a predefined set of features.

To compute the set of features for a segmented dataset, run the following command:

```bash
python3 generate_features.py --path=data/TSB_128/
```

- path: Path to the dataset for computing the features (the dataset should be segmented first into windows; see the command above). The resulting dataset is saved in the same directory (**MANDATORY**).

> Note: This process is memory-intensive, and we required a minimum of 512GB of RAM to run it. If you encounter memory issues, optimizing the TSFresh library is not within the scope of this project.


#### KDSelector Related Commands and Description

To train a model of KDSelector, run the following command:

```bash
python3 train_deep_model.py --path=data/TSB_128/ --split=0.7 --file=experiments/supervised_splits/split_TSB_128.csv --model=resnet --params=models/configuration/resnet_default.json --batch=64 --epochs=10 --eval-true --output_dim=64 --alpha=0.2 --lambda_CL=0.78 --temperature=0.22 --LLM_mode=eval --prune=0.8 --nbits=14 --nbins=8
```

- path: Path to the dataset to use.
- split: Split percentage for training and validation sets.
- seed: Seed for train/val split (optional).
- file: Path to a file that contains a specific split (to reproduce our results).
- model: Model to use (type of deep model architecture of KDSelector).
- params: A JSON file with the model's parameters.
- batch: Batch size.
- epochs: Number of training epochs.
- eval-true: Whether to evaluate the model on test data after training.
- output_dim: The output dimension of the multi-layer perceptron (MLP)
- alpha: The hyperparameter controls the ratio between soft labels and hard labels in the total loss function. 
- lambda_CL: The hyperparameter controls the ratio between text labels and other labels in the total loss function.
- temperature: The hyperparameter adjusts the temperature coefficient of the softmax function in the soft label module. 
- LLM_mode: Whether the model should fine-tune the large language model (LLM) during training. You can select either eval or train, with eval as default.
- prune: Adjust prune_ratio
- nbits: Adjust number of bits of hash_codes
- nbins: Adjust number of buckets when prune with IndexLSH

> This script will save the following:
>
> - training specific information into _/MSAD/results/done_training/resnet_default_128_11092023_162841.csv_ file.
> - TensorBoard data will be saved into _/MSAD/results/runs/resnet_default_128_11092023_162726/_.
> - The trained weights will be saved in _/MSAD/results/weights/resnet_default_128/_.
> - In case the 'eval-true' is selected, the results of the trained model on the test set will be saved in _/MSAD/results/raw_predictions/resnet_128_preds.csv_.

To evaluate a model on a folder of CSV files, run the following command:

```bash
python3 eval_deep_model.py --data=data/TSB_128/MGAB/ --model=resnet --model_path=results/weights/supervised/resnet_default_128/model_30012023_173428 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/
```

- data: Path to the time series data to predict.
- model: Model to use.
- model_path: Path to the trained model.
- params: A JSON file with the model's parameters.
- path_save: Path to save the results.

> The results of the above inference example are saved in _/MSAD/results/raw_predictions/resnet_128_preds.csv_.

To reproduce our specific results, run the following command:

```bash
python3 eval_deep_model.py --data=data/TSB_128/ --model=resnet --model_path=results/weights/supervised/resnet_default_128/model_30012023_173428 --params=models/configuration/resnet_default.json --path_save=results/raw_predictions/ --file=experiments/supervised_splits/split_TSB_128.csv
```

- file: Path to a file that contains a specific split (to reproduce our results).
