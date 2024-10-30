# Beyond the Doors of Perception: Vision Transformers Represent Relations Between Objects
This repository contains the code and data necessary to reproduce the results in Beyond the Doors of Perception: Vision Transformers Represent Relations Between Objects (NeurIPS 2024). Original model weights are available upon request.

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

For examples of calling these scripts, see the `CCV` directory.

### Analyses
Graphing files are found in `analysis`

## Dependencies

This repository requires using our local copy of Pyvene, which is included here.  Additionally, you must use our fork of Transformer_lens, found here: `https://github.com/mlepori1/TransformerLens.git`.
