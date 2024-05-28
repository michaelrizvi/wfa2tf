# Learning WFAs with Transformers

This repo contains the code necessary to reproduce the experiments from the paper 
"Learning WFAs with Transformers". 

## Getting Started

### Dependencies

This project uses Python 3.8. To install Python 3.8 follow the instructions below
*For Mac: Type `brew install python@3.8` into your terminal

### Installing
It is recommended to run this code in a virtual environment. I used `venv` for this project. 
To setup the virtual environment and download all the necessary packages, follow the steps below

First, load the Python module you want to use:
```
module load python/3.8
```
Or use `python3.8` instead in the following commands. Then, create a virtual environment in your home directory:

```
python -m venv $HOME/<env>
```
Where `<env>` is the name of your environment. Finally, activate the environment:

```
source $HOME/<env>/bin/activate
```

Now to install the packages simply run
```
pip install -r requirements.txt
```

### Executing program
To run any training script, simply launch using the `python` command or using the editor/IDE of your choice. For example to run the `train_counting.py` experiment:
```
python train_counting.py
```

## Authors

Michael Rizvi-Martel (correspondence to michael.rizvi-martel@mila.quebec)
Maude Lizaire
Clara Lacroce
Guillaume Rabusseau

## License

This project is licensed under the MIT License - see the LICENSE.md file for details
