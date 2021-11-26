# NSR
## Dependence
- python3
- sklearn
- numpy


## Algorithms
- NN-MU_lp
- NN-MU_l1
- NN-KSVD

## running code
### Setting
Add the path of `NSR/nsr/`

### `NN_MU_lp` (p=0.9, alpha=0.35)
1. Move directory `nsr/`
2. Run the following command

```python3
python3 demo_nsr.py
```

### `NN_MU_lp` with Hoyer's projection function (p=0.9, alpha=0.35)
1. Move directory `nsr/`
2. Run the following command

```python3
python3 demo_nsr_lp_proj.py
```