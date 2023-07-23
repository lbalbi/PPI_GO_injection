This repo has the code required to train and evaluate several neural models based on two approaches that attempt to inject avaliable domain knowledge on PPI data to solve a multi-label classification task for the PPI dataset "ogbn-proteins" made avaliable by the OGB project (see: https://github.com/snap-stanford/ogb/tree/master ) .


We propose the building of enriched torch-based datasets that make up graphs constructed over: approach i) "Hollistic-Graph" - a hypergraph that combines the PPI data with the Gene Ontology through protein annotations; and approach ii) "Dual-Graph" - a GO graph that includes protein annotations to the GO tree.
For access to the datasets see: https://bitbucket.org/laurabalbi/datasets-ppi-go/downloads/

i) "Hollistic-Graph" Datasets:

dataset_SGOFULL_NR is an enriched PPI dataset with non-redundant annotations to the Full GO and split between train/validation/test by protein species;

dataset_RGOFULL_NR is an enriched PPI dataset with non-redundant annotations to the Full GO and randomly split between train/validation/test;

dataset_SGOFULL_TRAIN is an enriched PPI dataset with training proteins' non-redundant annotations to the Full GO and split between train/validation/test by protein species;

dataset_RGOFULL_TRAIN is an enriched PPI dataset with training proteins' non-redundant annotations to the Full GO and randomly split between train/validation/test;

dataset_SGOGEN is an enriched PPI dataset with annotations to the GO Slim Generic and split between train/validation/test by protein species;

dataset_RGOGEN is an enriched PPI dataset with annotations to the GO Slim Generic and randomly split between train/validation/test;


Graph-based DL approaches over the graphs in i) will receive the GO as part of the training data, this way directly receiving both PPI and Protein Function information through graph structure exploration and training over a global, combined representation of this knowledge.

DL approaches that see the inclusion of the graphs in ii) will receive both a PPI graph and a GO annotated graph, exploring both protein caracterization aspects in separate, but training over both representations.

