# Maria: Artifical Intelligence plays Super Mario Bros.

## Summary

Maria is an artificial intelligence (AI) suite for training and testing reinforcement learning models to play original Super Mario Bros. (SMB) levels. Maria utilizes the Proximal Policy Optimization (PPO) algorithm to train its models. You may use the pre-trained models provided in `crc/outputs` which were trained on supercomputers through Notre Dame's Center for Research Computing, or you may train your own custom models. Below are instructions for on how to setup your environment and start using Maria!

## Usage

### Environment Setup

To begin, you must have Python installed on your  machine. Python 3.10.5+ is recommended since that version was used to test Maria.

Next, clone the repository.
```bash
git clone https://github.com/eamontracey/maria
cd maria
```

Next, you must install the necessary dependencies/libraries. The use of a Python virtual environment is highly recommended.
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

Now, your environment is all set up!

### Training

Now that your environment has been configured with the necessary dependencies, you are ready to train a model. The training script is located at `src/train.py`. There are multiple training options.
```bash
python3 src/train.py --help
```
yields
```txt
usage: train.py [-h] --world WORLD --stage STAGE [--vectors VECTORS] [--steps STEPS] [--learning-rate LEARNING_RATE]

Train AI to play Super Mario Bros.

options:
  -h, --help            show this help message and exit
  --world WORLD         world (1-8)
  --stage STAGE         stage (1-4)
  --vectors VECTORS     number of vectors (processes)
  --steps STEPS         number of steps
  --learning-rate LEARNING_RATE
                        learning rate
```
The `--world` and `--stage` options allow you to select the SMB level on which to focus the training.

The `--steps` and `--learning-rate` options are settings to configure and optimize the PPO model. More steps and a lower learning rate will lead to a more optimized model at the expense of time. The default values are `--steps=100000` and `--learning-rate=0.0003`.

The `--vectors` option specifies how many processes to use. Maria supports multiprocessing acceleration to drastically speed up training times. It is highly recommended to use a value less than or equal to the number of CPUs available on your machine. The default value is `--vectors=4`.

An example command to train a model for 4-1 using one million steps and eight cores:
```bash
python3 src/train.py --world 4 --stage 1 --steps 1000000 --vectors 8
```
**Note**: Training accurate, optimized models will take *a long time*.

### Playinh