# Beyond the Doors of Perception: Vision Transformers Represent Relations Between Objects

## Reproducibility
### Data Generation
`data.py` is used to generate training data for discrimination and RMTS tasks. Once these are generated, run `data.py --create_das_dataset` to generate DAS counterfactual datasets. Add `--match_to_sample` to generate RMTS counterfactual datasets.

### Model Training
`sweep.py` is used to kick off model training. Add `--auxiliary_loss` to induce disentangled object representations.

### Perceptual Stage 
`das.py` runs DAS over all layers of a specified model to find a specified subspace. 

### Relational Stage
`das.py` also returns novel representation analysis results.

`linear_probe_intervention.py` trains linear probes over RMTS models and runs the additive intervention on them.

### Analyses
Graphing files are found in `analysis`
