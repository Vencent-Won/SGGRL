SGGRL:Multi-Modal Representation Learning for Molecular Property Prediction: Sequence, Graph, Geometry

Usage:

1. Process Data\
'''python
python build_corpus.py --in_path {data_path} --out_path {save_path}
python build_vocab.py --corpus_path {corpus_path} --out_path {save_path}
python data_3d.py --dataset {dataset name}
'''
2. Molecular Property Prediction
'''python
python main.py --dataset {dataset name}
'''


