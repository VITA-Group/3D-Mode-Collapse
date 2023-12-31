# Taming Mode Collapse in Score Distillation for Text-to-3D Generation

Peihao Wang, Dejia Xu, Zhiwen Fan, Dilin Wang, Sreyas Mohan, Forrest Iandola, Rakesh Ranjan, Yilei Li, Qiang Liu, Zhangyang Wang, Vikas Chandra

[[Project Page]](https://vita-group.github.io/3D-Mode-Collapse/) | [[Paper]](/)

![](figures/teaser.gif)

## Abstract

Despite the remarkable performance of score distillation in text-to-3D generation, such techniques notoriously suffer from view inconsistency issues, also known as "Janus" artifact, where the generated objects fake each view with multiple front faces. Although empirically effective methods have approached this problem via score debiasing or prompt engineering, a more rigorous perspective to explain and tackle this problem remains elusive. In this paper, we reveal that the existing score distillation-based text-to-3D generation frameworks degenerate to maximal likelihood seeking on each view independently and thus suffer from the mode collapse problem, manifesting as the Janus artifact in practice. To tame mode collapse, we improve score distillation by re-establishing in entropy term in the corresponding variational objective, which is applied to the distribution of rendered images. Maximizing the entropy encourages diversity among different views in generated 3D assets, thereby mitigating the Janus problem. Based on this new objective, we derive a new update rule for 3D score distillation, dubbed Entropic Score Distillation (ESD). We theoretically reveal that ESD can be simplified and implemented by just adopting the classifier-free guidance trick upon variational score distillation.
Although embarrassingly straightforward, our extensive experiments successfully demonstrate that ESD can be an effective treatment for Janus artifacts in score distillation.

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
