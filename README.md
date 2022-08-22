# RewardRepair
[Neural Program Repair with Execution-based Backpropagation (ICSE 2022)](http://arxiv.org/pdf/2105.04123)

```
@inproceedings{Ye2021RewardRepair,
 title = {Neural Program Repair with Execution-based Backpropagation},
 year = {2022},
 author = {He Ye and Matias Martinez and Martin Monperrus},
 url = {http://arxiv.org/pdf/2105.04123},
 doi = {10.1145/3510003.3510222},
 booktitle = {Proceedings of the International Conference on Software Engineering},
}

```

# Trained Model
We share our trained model in Zenodo:
```
RewardRepair trained model: https://doi.org/10.5281/zenodo.5997686
```
 


## Folder Structure
 ```bash
 ├── data: csv data used for training
 │ 
 ├── model: the trained model of RewardRepair
 │
 ├── train.py: script to  train RewardRepair
 │
 ├── test.py: script to test RewardRepair
 │
 ├── result: raw experiment results
 │
 ├── ComparisonData.csv: the file to compare with the state-of-the-art
 
```

## Prerequisites

* JDK 1.8
* Pytorch==1.7.1
* transformers>=4.10.0
* OS: Linux and Mac
* Add submodule Bears and copy the folder myscript to Bears


```
git submodule add https://github.com/bears-bugs/bears-benchmark.git
cp -rf myscript ./bears-benchmar/
pip install transformers
pip install sentencepiece
```

Get the CoCoNut java dataset from here: https://github.com/lin-tan/CoCoNut-Artifact/releases/tag/training_data_1.0.0
use the script to extract data: extract_CoCoNut_data.py
```
Data availables here: 
https://zenodo.org/record/7009192#.YwPS4-xBxbg 
```

Get Megadiff dataset from here https://github.com/monperrus/megadiff
use the script to extract data: extract_Megadiff_data.py

Get CodRep dataset from here https://github.com/KTH/codrep-2019


To run our script
```
python3 train.py
```

To evaluate our model on other bugs (make sure the script takes the correct test file patch)
```
python3 test.py
```



## RQ2: Details reults are available in RewardRepair/ResultForRQ2/
| Benchmarks | Top30 | Top100 | Top200 |
| :---: | :---: | :---: |:---: |
| QuixBugs | 44.1% | 35.7% | 31.5%|
| Defects4J | 45.7% | 38.0% |33.6%|

Note: Above are the number reported in the paper. The actual compilable rate should be higher. Because we found many uncompilable cases due to space in  operators > =, < =, = =, ! =.
This could be easily avoided by added tokens in training or post processing in test.py.
