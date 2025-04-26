# <span style="color: #ff4500">🚀 Alpha Go Everywhere: Machine Learning and International Stock Returns 🚀</span>

This is the README file for the project **Alpha Go Everywhere: Machine Learning and International Stock Returns** ([SSRN link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3489679)) accepted by *Review of Asset Pricing Studies*. It provides an overview of the project structure and instructions on how to use and contribute to the codebase.

---

## 📑 Table of Contents

- [Project Structure](#project-structure)
- [Usage](#usage)
- [Data](#data)
- [Computing Environment](#computing-environment)

---

## 🏗️ Project Structure

The project is organized as follows (**key scripts** highlighted):

- ❗️ `<span style="background-color: #ffff00">Rank_Norm.py</span>`: Rank-normalize the data, like GKX's paper.  
- 📂 `Load_Data.py`: Necessary functions for loading or preprocessing data  
- ⚙️ `SetUp.py`: Variable definitions  
- 🛠️ `Local{US}_Factor{GapQ}.py`: Create Local{US} factor{GapQ}  
- 🔗 `Merge_Factor+GapQ.py`: Merge US factors, US gaps, and local factors  
- 🌐 `International_Pool.py`: Integrate all standardized market data into one dataset  
- 🤖 `ML{NN}_Market.py`: Train various ML{NN} models for each market  
- 🗽 `ML{NN}_Market_USmodel.py`: Predict international markets using the USA model (no further training)  
- 🚀 `ML{NN}_Market_Enhanced.py`: Train enhanced market-specific models with USA factors, gaps, and local features  

---

## 🚀 Usage

To use the project, follow these steps:

1. **Run** `<span style="background-color: #ffff00">Rank_Norm.py</span>` to rank-normalize the predictors (as in GKX’s paper).  
2. **Run** `Local{US}_Factor{GapQ}.py` to create Local{US} factor{GapQ}.  
3. **Run** `Merge_Factor+GapQ.py` to merge US factors, gaps, and local factors.  
4. **Run** `International_Pool.py` to integrate all standardized market data into one international dataset.  
5. **Run** `ML{NN}_Market.py` to train ML models for each market.  
6. **Run** `ML{NN}_Market_USmodel.py` to predict international markets using the USA model.  
7. **Run** `ML{NN}_Market_Enhanced.py` to train enhanced models with additional features.  

---

## 🗄️ Data

- **US data** from **CRSP**  
- **China data** from **CSMAR**  
- **Other markets** data from **DataStream**  

---

## 💻 Computing Environment

To run the reproducibility checks, the following environment and packages are **required**:

- **Hardware**  
  - Nvidia A100 GPU (40 GB)  
  - AMD EPYC 7713 64-Core @ 1.80 GHz (128 cores)  
  - 1.0 TB RAM  
  - Ubuntu 20.04.4 LTS  

- **Software**  
  - 🐍 Python 3.8.18  
  - 🔥 PyTorch 2.0.1+cu117  
  - 📊 numpy 1.22.3  
  - 📑 pandas 2.0.3  

---

**Happy coding!** 🎉  
