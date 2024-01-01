# DDIM Implementation with Classifier-Free Guidance

Example 32x32 images of cars generated using 24-step DDIM rollouts with classifier free guidance by a model trained on CIFAR-10:

![Example Generated CIFAR-10 Images](example.png)

**Example Training Command:**
```bash
python src/train.py configs/cifar10.yaml
```

**Example Inference Command:**
```bash
python src/infer.py configs/cifar10.yaml --iter-num 1000000 --cond 1
```

**References:**
- https://www.tonyduan.com/diffusion
- https://arxiv.org/abs/2006.11239
- https://arxiv.org/abs/2010.02502
- https://arxiv.org/abs/2207.12598
