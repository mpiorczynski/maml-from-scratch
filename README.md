<style>
  /* Three image containers (use 25% for four, and 50% for two, etc) */
.column {
  float: left;
  width: 40%;
  padding: 5px;
}

/* Clear floats after image containers */
.row::after {
  content: "";
  clear: both;
  display: table;
}
</style>
<div class="row">
  <div class="column">
    <img src="assets/maml-teaser.png" width="100%"/>
  </div>
  <div class="column">
    <img src="assets/maml-algorithm.png" width="72.5%"/>
  </div>
</div>


## Install
```bash
git clone https://github.com/mpiorczynski/maml-from-scratch.git
cd maml-from-scratch
conda create -n maml-from-scratch python=3.9
conda activate maml-from-scratch
pip install -r requirements.txt
```

Stronly inspired by `higher` package MAML [example](https://github.com/facebookresearch/higher/blob/main/examples/maml-omniglot.py).

## Usage
```
./run.sh maml.py \
  --task-name 'sinusoid' \
  --k-shot 10 --k-query 10 \
  --meta-batch-size 16 --num-epochs 10 \
  --metch-optimizer 'adam' \
  --inner-steps 1 --inner-learning-rate 0.01 \
  --seed 42 \
  --use-wandb \
  --log-interval 100 
```

## Citations
```bibtex
@inproceedings{finn2017model,
  title={Model-agnostic meta-learning for fast adaptation of deep networks},
  author={Finn, Chelsea and Abbeel, Pieter and Levine, Sergey},
  booktitle={International conference on machine learning},
  pages={1126--1135},
  year={2017},
  organization={PMLR}
}
```