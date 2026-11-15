# ResNet on CIFAR-10 | Group 9 weekly progress

**2–8 November 2026**

### 周子旺

- Configured GitHub Actions to check that each pull request commit starts with the student ID and full name associated with its author’s GitHub account.
- Added `scripts/prepare_data.py` to download CIFAR-10, check the official splits and compute a per-pixel RGB mean from the training images. Added `src/data.py` to load training batches with random crops and horizontal flips and test batches with a single view. Checked the command-line help and Git exclusions for local data files.

### 李霄汉

- Implemented CIFAR-10 ResNet-20 in `src/resnet.py` with parameter-free Option A shortcuts and 269,722 trainable parameters.
- Verified the model on an RTX 5080 Laptop GPU using eight synthetic 32 × 32 RGB images. The check completed a forward pass, a backward pass and one SGD parameter update.

**9–15 November 2026**

### 周子旺

- On the basis of existing model implementations, a unified implementation of Plain-20, Plain-56, ResNet-20, and ResNet-56 was completed. Plain and ResNet with the same depth share the same learnable layer structure, with residual connections added or omitted to form comparisons. Weight initialization rules were clarified, and an automated random seed setting interface was added to model creation, ensuring that paired models at the same depth start training from identical initial parameters. Checks were completed for model layer counts, parameter quantities, feature dimensions, forward and backward computations, paired initialization, and compatibility with existing ResNet-20 weights.
- Develop independent training and evaluation programs to achieve batch data augmentation, round-by-round error rate recording, training node saving and restoration, and a visualization progress window. On the RTX 3060 Laptop GPU, with a random seed of 1, complete controlled experiments on CIFAR-10 with four models each undergoing 90 epochs and 35,190 parameter updates. After each epoch, evaluate the current model on all 50,000 training images and 10,000 test images. Compile and submit the training configurations, complete 90-epoch records, and six independent result images for the four models, separately presenting comparisons between the training and test sets.

### 李霄汉

- Based on the four models determined by the group, write scripts for model loading, training, evaluation, and batch running, integrate CIFAR-10 data preparation and preprocessing processes, and unify experimental settings such as batch size, optimizer, learning rate adjustment, and random seeds. Implement full evaluation for each round, record the training process, save weights, and restore breakpoints, and write instructions for experimental setup and operation methods.
- Complete forward and backward computation verification of four models on RTX 5080 Laptop GPU, as well as checking the consistency of initial parameters for models at the same depth. Complete the first round of CIFAR-10 training and full evaluation of the training and testing sets for four models, record model configuration, error rate, loss, runtime, and video memory usage, and submit the training code, experimental instructions, and first round records to GitHub.

**Group Conclusion**
- This week, the group conducted a comparative experiment on whether increasing network depth leads to degradation and whether residual connections can improve this phenomenon. The following comparison uses random seed 1 and four models trained for 90 rounds and completed 35190 parameter updates on common nodes. The training error rate is obtained by evaluating all training images with fixed current weights, while the testing error rate is obtained by evaluating all test images.
- After the ordinary network was deepened from 20 layers to 56 layers, the training error rate increased from 2.642% to 11.194%, and the testing error rate increased from 9.540% to 15.580%. The performance of deeper ordinary networks also deteriorates on the training set, and this control experiment reproduced the degradation phenomenon caused by increasing depth.
The following two graphs show the error rates of a regular network on the training and testing sets, respectively. The images show the 82nd to 90th rounds after the first reduction in learning rate. Each image compares two models, and the measured values of the 90th round are marked at the end.
<img width="1728" height="1080" alt="01_plain_train" src="https://github.com/user-attachments/assets/87c59b39-1798-45bb-a7af-44797299a7d9" />
<img width="1728" height="1080" alt="02_plain_test" src="https://github.com/user-attachments/assets/bc5b760c-49a2-4658-9a16-9eee3a9b06c3" />

- After introducing residual connections, the network deepened from 20 layers to 56 layers, reducing the training error rate from 1.974% to 0.440% and the testing error rate from 8.500% to 7.430%. When both layers are 56, the training error rate and testing error rate of ResNet-56 are lower than those of Plain-56. The results of this experiment have verified the improvement effect of residual connections on deep network optimization, which enables increasing depth to bring better training and testing performance.
- The following two graphs show the error rates of the residual network on the training and testing sets, respectively, using the same observation interval from rounds 82-90.
<img width="1728" height="1080" alt="03_resnet_train" src="https://github.com/user-attachments/assets/95c94b81-36a9-4a1f-a190-2c26d450ae54" />
<img width="1728" height="1080" alt="04_resnet_test" src="https://github.com/user-attachments/assets/43042477-88a7-4910-b88e-199c924c1e6e" />
<img width="1728" height="1080" alt="05_depth56_train" src="https://github.com/user-attachments/assets/235c7624-d08b-4f5c-bf48-9fd7fc7c3eb5" />
<img width="1728" height="1080" alt="06_depth56_test" src="https://github.com/user-attachments/assets/c73758d9-1810-45a1-a330-360705bee56a" />

