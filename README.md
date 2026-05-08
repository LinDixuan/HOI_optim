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

Run the wrist correction and optimization process for hand-object interaction:

```
python hoi_correction/correct_wrist.py --dataset omomo

python hoi_correction/finetune_stage2.py 

```



## Run Visualization

Generate visualizations for the results:

```
python visualization/optim_vis.py

```

Generate the visualization for a single sequence:

```
python visualization/optim_vis.py --seq_name <sequence_name>

```
