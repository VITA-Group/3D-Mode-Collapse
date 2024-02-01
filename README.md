# Taming Mode Collapse in Score Distillation for Text-to-3D Generation

Peihao Wang, Dejia Xu, Zhiwen Fan, Dilin Wang, Sreyas Mohan, Forrest Iandola, Rakesh Ranjan, Yilei Li, Qiang Liu, Zhangyang Wang, Vikas Chandra

[[Project Page]](https://vita-group.github.io/3D-Mode-Collapse/) | [[Paper]](https://arxiv.org/abs/2401.00909)

![](teaser.gif)

## Quick Start

1. Clone `threestudio` with the support of `ProlificDreamer`:
```
git clone https://github.com/threestudio-project/threestudio.git
```

2. Follow the instructions [here](https://github.com/threestudio-project/threestudio?tab=readme-ov-file#installation) to configure running environments and commands [here](https://github.com/threestudio-project/threestudio?tab=readme-ov-file#supported-models) to start your first text-to-3D experiments.

3. Experiencing Janus problem? Our paper suggests simply specifying the argument `system.guidance.guidance_scale_lora` can potentially relieve Janus problem. For example,
```
python launch.py --config configs/prolificdreamer.yaml --train --gpu 0
system.prompt_processor.prompt=<your_prompt> system.guidance.guidance_scale_lora=0.5
```

*Note:* The value of this hyperparameter may vary in different scenarios.

Check our [paper](https://arxiv.org/abs/2401.00909) for theoretical arguments on this surprising finding.

## Gaussian Example

To demonstrate the principle of our algorithm, we provide an example script in `gaussian_example.py` to visualize the training trajectory of matching two Gaussian distribution using various score distillation schemes.

<img src="materials/trajectory_sds.gif" alt="SDS" width="33%"/> <img src="materials/trajectory_vsd.gif" alt="VSD" width="33%"/> <img src="materials/trajectory_esd.gif" alt="ESD" width="33%"/>

To reproduce these results, you may try the following commands:

```
# SDS
python gaussian_example.py --method sds --save_video
# VSD
python gaussian_example.py --method vsd --save_video
# ESD (Ours)
python gaussian_example.py --method esd --lambda_coeff 1.0 --save_video
```

It is also recommended to play with `--lambda_coeff` to see how this hyperparameter affect the matching results.

## Citation

If you find this work or our work helpful for your own research, please cite our paper.

```
@article{wang2023esd,
  title={Taming Mode Collapse in Score Distillation for Text-to-3D Generation},
  author={Wang, Peihao and Xu, Dejia and Fan, Zhiwen and Wang, Dilin and Mohan, Sreyas and Iandola, Forrest and Ranjan, Rakesh and Li, Yilei and Liu, Qiang and Wang, Zhangyang and Chandra, Vikas},
  journal={arXiv preprint: 2401.00909},
  year={2023}
}}
```
