This repository contains several datasets that make up graphs that include contextual knowledge on PPI data from a benchmark dataset. 
These approaches attempt to solve the node classification task for the PPI dataset "ogbn-proteins" made avaliable by the OGB project.


Each folder contains files required to build datasets that have graphs constructed over either i) a hypergraph that combines the PPI data with the Gene Ontology through protein annotations; and ii) a GO graph that includes protein annotations to the GO tree.

Graph-based DL approaches over the graphs in i) will receive the GO as part of the training data, this way directly receiving both PPI and Protein Function information through graph structure exploration and training over a global, combined representation of this knowledge.

DL approaches that see the inclusion of the graphs in ii) will receive both a PPI graph and a GO annotated graph, exploring both protein caracterization aspects in separate, but training over both representations.

