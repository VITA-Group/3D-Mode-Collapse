# Taming Mode Collapse in Score Distillation for Text-to-3D Generation

Peihao Wang, Dejia Xu, Zhiwen Fan, Dilin Wang, Sreyas Mohan, Forrest Iandola, Rakesh Ranjan, Yilei Li, Qiang Liu, Zhangyang Wang, Vikas Chandra

[[Project Page]](https://vita-group.github.io/3D-Mode-Collapse/) | [[Paper]](/)

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

Check our [paper](/) for theoretical arguments on this surprising finding.

## More Implementation Options

Coming soon ...

## Citation

If you find this work or our work helpful for your own research, please cite our paper.

```
@inproceedings{wang2023taming,
title={Taming Mode Collapse in Score Distillation for Text-to-3D Generation},
author={Wang, Peihao and Xu, Dejia and Fan, Zhiwen and Wang, Dilin and Mohan, Sreyas and Iandola, Forrest and Ranjan, Rakesh and Li, Yilei and Liu, Qiang and Wang, Zhangyang and Chandra, Vikas},
year={2023}
}
```
