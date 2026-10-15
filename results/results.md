# LoRA study results (test set, 2,000 texts)

| Method | Trainable params | Trainable % | Test accuracy | Test macro F1 | Gap to full FT (acc) | Stored per task | Train time |
|---|---|---|---|---|---|---|---|
| Head only (frozen encoder) | 595K | 0.89% | 54.8% | 0.309 | -36.8 pt | 2.3 MB | 27.1 min |
| LoRA r=2 (attention q,v) | 632K | 0.95% | 83.1% | 0.767 | -8.4 pt | 2.4 MB | 28.5 min |
| LoRA r=8 (attention q,v) | 743K | 1.12% | 86.1% | 0.801 | -5.4 pt | 2.8 MB | 27.3 min |
| LoRA r=32 (attention q,v) | 1.19M | 1.77% | 89.3% | 0.837 | -2.2 pt | 4.5 MB | 27.2 min |
| LoRA r=8 (all linear layers) | 1.26M | 1.88% | 90.3% | 0.851 | -1.2 pt | 4.8 MB | 43.7 min |
| Full fine-tuning | 66.96M | 100.00% | 91.5% | 0.870 | — | 255.4 MB | 63.7 min |
