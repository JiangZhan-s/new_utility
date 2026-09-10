# Run the generator-aware BAQP model

## 1. Pull and run tests

```bash
git pull
python -m unittest discover -s tests -v
```

The new tests are in `tests/test_quality_mechanism.py`.

## 2. Pure mechanism diagnostic, K=6

Use the IMFL-AIGC CIFAR-10 reference gradient-gap ratio:

```bash
python -m baqp.sweep_quality \
  --dataset cifar10 \
  --data-root data \
  --clients 6 \
  --alpha 0.3 \
  --split-seed 2026 \
  --generator-gap-ratios 0 0.30857142857142855 0.4
```

The `0` case is the perfect-AIGC ablation. The `0.30857...` case equals `0.54/1.75`, the direct gradient-gap ratio corresponding to the CIFAR-10 values reported by IMFL-AIGC.

## 3. Large-client heuristic diagnostic

```bash
python -m baqp.heuristic_quality \
  --dataset cifar10 \
  --data-root data \
  --clients 20 50 \
  --alpha 0.3 \
  --split-seed 2026 \
  --budgets 0.05 0.10 0.20 0.40 \
  --generator-gap-ratio 0.30857142857142855
```

This is a multi-start add/drop/swap heuristic. It is a structural diagnostic and does not claim global optimality for K=20/50.

## 4. Measure the exact synthetic cache quality and build contracts only

```bash
python -m baqp.experiment_quality \
  --dataset cifar10 \
  --data-root data \
  --aigc-root aigc_imgs \
  --clients 6 \
  --alpha 0.3 \
  --split-seed 2026 \
  --budget-fraction 0.4 \
  --model cnn \
  --generator-gap-source measured \
  --quality-device cpu \
  --quality-samples-per-class 512 \
  --prepare-only
```

Inspect `metadata.json -> generator_quality` and the `continuous/contract.json` allocation before spending GPU time.

## 5. Fixed-compute CNN comparison

Run inside the normal Slurm GPU allocation:

```bash
python -m baqp.experiment_quality \
  --dataset cifar10 \
  --data-root data \
  --aigc-root aigc_imgs \
  --clients 6 \
  --alpha 0.3 \
  --split-seed 2026 \
  --budget-fraction 0.4 \
  --model cnn \
  --rounds 100 \
  --local-epochs 2 \
  --batch 32 \
  --generator-gap-source measured \
  --methods fedavg_all no_aigc_budget three_state continuous selected_no_aigc selected_full
```

The decisive accuracy comparison is `continuous` versus `selected_full`: they use the same selected clients and the same number of local optimizer updates. The only intended systematic difference is the real/synthetic mixture induced by `u`.

Do not claim that the theory proves CNN test accuracy. The theoretical statement is that the generator-aware profile tightens the same CE-risk/accuracy certificate under the theorem assumptions; the CNN ordering remains an empirical result.
