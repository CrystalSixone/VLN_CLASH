# Code for VLN-CLASH

---

**Note:** This codebase is currently under development. As the associated paper is still under review, we are in the process of cleaning and releasing the code progressively.
Thank you for your understanding and interest.

---

## 1. Fine-tuning

To fine-tune the model, run:

```bash
bash run_r2r/main.bash train
```

---

## 2. Validation

To evaluate the model on the validation splits, run:

```bash
bash run_r2r/main.bash eval val_seen_unseen 
```

This command evaluates both the `val-seen` and `val-unseen` splits simultaneously and records the results separately.

---

## 3. Inference

To generate test predictions, run:

```bash
bash run_r2r/main.bash infer
```

This will generate the `pred.json` file for the `test-unseen` split, which can be submitted to the online VLN-CE leaderboard for fair comparison.
