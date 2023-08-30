# SGGRL: Multi-Modal Representation Learning for Molecular Property Prediction: Sequence, Graph, Geometry

## Framework

![method](https://cdn.jsdelivr.net/gh/Vencent-Won/GraphBed/img/SGGRL-framework.png)



## Usage:

1. Process Data
```
python build_corpus.py --in_path {data_path} --out_path {save_path}
python build_vocab.py --corpus_path {corpus_path} --out_path {save_path}
python data_3d.py --dataset {dataset name}
```
2. Molecular Property Prediction
```
python main.py --dataset {dataset name} --task_type {reg/class}
```

