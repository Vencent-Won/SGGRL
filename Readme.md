# SGGRL: Multi-Modal Representation Learning for Molecular Property Prediction: Sequence, Graph, Geometry

## Framework

![method](https://cdn.jsdelivr.net/gh/Vencent-Won/GraphBed/img/SGGRL-framework.png)

## Enviroment
- paddle-bfloat==0.1.7
- paddlepaddle==2.5.1
- torch==1.13.0
- torch-cluster==1.6.0+pt113cu117
- torch-geometric==2.2.0
- torch-scatter==2.1.0+pt113cu117
- torch-sparse==0.6.15+pt113cu117
- torch-spline-conv==1.2.1+pt113cu117
- rdkit==2023.3.1

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

