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

For example, train a model for 4-1 using one million steps and eight cores:
```bash
python3 src/train.py --world 4 --stage 1 --steps 1000000 --vectors 8
```
**Note**: Training accurate, optimized models will take *a long time*.

While the model trains, you will observe multiple outputs. If using a terminal, a progress bar indicating the completion of the training process will be printed to the terminal. Also, tensorboard-compatible files will be output to the `board` directory; these can be viewed using `tensorboard --logdir ./board` and display real-time information such as mean reward value as the model trains.

Most importantly, the model is saved to a `.ppo` file in the `models` directory. The file name includes the SMB level used for that model and the Unix epoch time at which the model is saved (to avoid name conflicts). Further, the training script uses a callback to evaluate the strength of the model. If the callback determines that the current model is strongest, then it saves the model to `models/best_model.zip`. These model files will be fed to the running script.

### Running

Once you have a trained model stored in a file, you are ready to run and test the model. The model running script is located at `src/maria.py`. There are multiple running options.
```bash
python3 src/maria.py --help
```
yields
```txt
usage: maria.py [-h] --model MODEL --world WORLD --stage STAGE [--runs RUNS] [--seed SEED]

Let artificial intelligence play Super Mario Bros.

options:
  -h, --help     show this help message and exit
  --model MODEL  path to the pre-trained model file
  --world WORLD  world (1-8)
  --stage STAGE  stage (1-4)
  --runs RUNS    number of runs (omit for infinity)
  --seed SEED    randomness seed (omit to use random seeds)
```
The `--model` option specifies the file path to the model.

The `--world` and `--stage` options allow you to select the SMB level on which to run the model.

The `--runs` and `--seed` options further configure how to run the given model. The `--runs` option specifies on how many Mario lives to run the model predictiomn scheme. The `--seed` options specifies a randomness seed. This is particularly useful if you want to keep track of and save specific run results. By default, the model will perform infinitely many runs with random seeds.

For example, test a pretrained 3-2-super model on level 3-1 with a set seed. 
```bash
python3 src/maria.py --world 3 --stage 2 --model crc/outputs/3-2-super/models/best_model.zip --seed 31
```