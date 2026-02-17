## Environment Setup

### 1. Installation
Follow the instructions in the [official InterAct repository](https://github.com/wzyabcas/InterAct/tree/main) to set up the environment.

### 2. Dataset Preparation
Download the **OMOMO** dataset and ensure the directory structure looks like this:

```text
.
└── omomo/
```

## Hand Correction

Run the optimization process for hand-object interaction:

```
python hoi_correction/optimize.py

```

## Run Evaluation
Run the evaluation script on the OMOMO dataset:

```
python visualization/omomo_eval.py

```


## Run Visualization

Generate visualizations for the results:

```
python visualization/omomo_vis.py

```

