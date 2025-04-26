# Alpha Go Everywhere: Machine Learning and International Stock Returns

This is the README file for the project Alpha Go Everywhere: Machine Learning and International Stock Returns (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3489679) accepted by Review of Asset Pricing Studies. It provides an overview of the project structure and instructions on how to use and contribute to the codebase.

## Table of Contents

- [Project Structure](#project-structure)
- [Usage](#usage)
- [Data](#data)
- [Computing Environment](#computing-environment)

## Project Structure

The project is organized as follows:

- `Load_Data.py`: Necessary functions for loading or preprocessing data
- `SetUp.py`: Variables names
- `Local{US}_Factor{GapQ}.py`: Create Local{US} factor{GapQ}
- `Merge_Factor+GapQ.py`: Merge US factors, US gaps, and local factors
- `International_Pool.py`: Intergrate all standardlised market data into one international data
- `ML{NN}_Market.py`: Train various ML{NN} models for each market
- `ML{NN}_Market_USmodel.py`: Predict the international markets using the USA model, No further training.
- `ML{NN}_Market_Enhanced.py`: Train market-specific models enhanced by the USA factors, USA characteristics gaps, and local factors, as additional features.


## Usage

To use the project, follow these steps:

1. Run Rank_Norm.py to rank-normalize the data, like GKX's paper.
1. Run Local{US}_Factor{GapQ}.py to create Local{US} factor{GapQ}.
2. Run Merge_Factor+GapQ.py to merge US factors, US gaps, and local factors.
3. Run International_Pool.py to integrate all standardlised market data into one international data.
4. Run ML{NN}_Market.py to train various ML{NN} models for each market.
5. Run ML{NN}_Market_USmodel.py to predict the international markets using the USA model. 
6. Run ML{NN}_Market_Enhanced.py to train market-specific models enhanced by the USA factors, USA characteristics gaps, and local factors, as additional features.

## Data
US data from CRSP and China data from CSMAR, Other markets data from DataStream.

## Computing Environment
To run the reproducibility check, the following computing environment and package(s) are required:
- Environment: These experiments were conducted on a system equipped with an Nvidia A100 GPU with 40 GB of GPU memory, an AMD EPYC 7713 64-Core Processor @ 1.80GHz with 128 cores, and 1.0TB of RAM, running Ubuntu 20.04.4 LTS. 

- Package(s): 
    - Python 3.8.18
    - PyTorch 2.0.1+cu117
    - numpy 1.22.3
    - pandas 2.0.3