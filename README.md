# Multiword expressions (MWE)-aware dependency parsing  

## About
This repository contains the implementation of MWE-aware dependency parsing, where each MWE is a syntactic unit in a dependency tree. Currently, we support functional (e.g., "a number of", "even though"), verbal (e.g., "pick up"), and adjectival MWEs (e.g., "out of business").

## Codebase
This repository is forked from [HMTL (Hierarchical Multi-Task Learning model)](https://github.com/huggingface/hmtl).

## Dependencies
- Python 3.6
- PyTorch 0.4.1
- AllenNLP 0.7.0

## The way to develop a virtual environment
- We recommend you to build your virtual environment with pipenv instead of poetry.

## Data
We provide users with a sample configuration file (configs/sample.json) and sample data for dependency parsing and MWE recognition (data/sample_{parsing | vmwe}.conll). 

## Training and Inference
```
$> python train.py --config_file_path configs/sample.json --serialization_dir sample_001

$> python evaluate.py --serialization_dir sample_001
```

## References
Please consider citing the following paper if you find this repository useful.
```
Akihiko Kato, Hiroyuki Shindo, and Yuji Matsumoto. 2019. 複単語表現を考慮した依存構造コーパスの構築と解析 (Construction and Analysis of Multiword Expression-aware Dependency Corpus). In 自然言語処理 (Natural Language Processing), 26(4), pp.663-688
```
