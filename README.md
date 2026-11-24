# Context-enriched RCA experiments

Standalone reproduction code comparing a local
Random Forest baseline and remote decision models on RCAEval RE2. Models receive
application descriptions, service roles, and baseline/incident metric summaries.

Start with [REPRODUCE.md](REPRODUCE.md) for installation, data preparation,
reference-split verification, server setup and execution. Run all commands
from this directory.

```sh
python prepare_dataset.py
python -m Context_enriched_experiment.run --models rf --output-dir results/context_rf
# After deploying the remote model servers and configuring their URLs:
python -m Context_enriched_experiment.run --models all --output-dir results/context_all
```

See the [experiment documentation](Context_enriched_experiment/README.md) for
prompt construction and reporting, and [dataset_readme.md](dataset_readme.md)
for data schemas and feature definitions. Server URLs are configured in
`rca_experiments/config.py`. Server implementations and weights are external.
