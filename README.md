# EHGNN

An implementation for the paper--Efficient Learning for Billion-scale Heterogeneous Information Networks.

### Dataset

The three datasets used in the paper (PubMed, Yelp and DBLP) can be downloaded from [here](https://drive.google.com/drive/folders/186u90Y0gzmdI-R6gQEOis1Nip6G759rv?usp=drive_link). In addition, the OGB-MAG240M dataset can be found [here](https://ogb.stanford.edu/docs/lsc/mag240m/). Please place the downloaded datasets in the `../data`.

### Usage

To conduct the experiments, please execute `main.py` in each folder (Node Classification, Link Prediction, and MAG240M). Hyperparameters can be explored within the `main.py`, and here are the ones we used.

| Task | Dataset | $\alpha$ | K   | learning rate | dropout | hidden dimension | layers | batch size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Node Classification | PubMed | 0.7 | 20  | 1e-3 | 0.4 | 256 | 4   | 3000 |
|     | Yelp | 0.7 | 20  | 3e-4 | 0.5 | 256 | 4   | 3000 |
|     | DBLP | 0.7 | 20  | 5e-4 | 0.5 | 512 | 5   | 3000 |
|     | OGB-MAG240M | 0.7 | 25  | 3e-4 | 0.4 | 512 | 5   | 5000 |
| Link Prediction | PubMed | 0.1 | 20  | 3e-4 | 0.5 | 256 | 4   | 40  |
|     | Yelp | 0.1 | 20  | 3e-4 | 0.5 | 256 | 4   | 100 |
|     | DBLP | 0.7 | 20  | 5e-4 | 0.5 | 512 | 5   | 1000 |
