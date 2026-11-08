# ResNet on CIFAR-10 | Group 9 weekly progress

**2–8 November 2026**

### 周子旺

- Configured GitHub Actions to check that each pull request commit starts with the student ID and full name associated with its author’s GitHub account.
- Added `scripts/prepare_data.py` to download CIFAR-10, check the official splits and compute a per-pixel RGB mean from the training images. Added `src/data.py` to load training batches with random crops and horizontal flips and test batches with a single view. Checked the command-line help and Git exclusions for local data files.

### 李霄汉

- Implemented CIFAR-10 ResNet-20 in `src/resnet.py` with parameter-free Option A shortcuts and 269,722 trainable parameters.
- Verified the model on an RTX 5080 Laptop GPU using eight synthetic 32 × 32 RGB images. The check completed a forward pass, a backward pass and one SGD parameter update.
