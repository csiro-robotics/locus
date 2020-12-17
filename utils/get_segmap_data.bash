#!/bin/bash
echo "Downloading pre-trained model into '~/.segmap/'"
`wget -P ~/.segmap/ http://robotics.ethz.ch/~asl-datasets/segmap/segmap_data/default_training.ini`
`wget -P ~/.segmap/trained_models/segmap64/ http://robotics.ethz.ch/~asl-datasets/segmap/segmap_data/trained_models/segmap64/checkpoint`
`wget -P ~/.segmap/trained_models/segmap64/ http://robotics.ethz.ch/~asl-datasets/segmap/segmap_data/trained_models/segmap64/graph.pb`
`wget -P ~/.segmap/trained_models/segmap64/ http://robotics.ethz.ch/~asl-datasets/segmap/segmap_data/trained_models/segmap64/model.ckpt.index`
`wget -P ~/.segmap/trained_models/segmap64/ http://robotics.ethz.ch/~asl-datasets/segmap/segmap_data/trained_models/segmap64/model.ckpt.meta`
`wget -P ~/.segmap/trained_models/segmap64/ http://robotics.ethz.ch/~asl-datasets/segmap/segmap_data/trained_models/segmap64/model.ckpt.data-00000-of-00001`
echo "Finished downloading pre-trained model"