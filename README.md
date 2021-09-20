# DCDL-Pytorch

This repo provide a pytorch version for "Deep Conditional Distribution Learningfor Age Estimation" in TIFS2021. 

## Abstract

Age estimation is a challenging task not only because face appearance is affected by illumination, pose, and expression, but also because there exists age label ambiguity among different demographic groups. In this work, we first revisit different label distribution learning (LDL) based age estimation methods and propose a more general formulation, which can unify individual LDL-based age estimation methods, as well as the traditional regression, classification, and ranking based age estimation methods. Based on such a general formulation, we propose a novel deep conditional distribution learning (DCDL) method, which can flexibly leverage a varying number of auxiliary face attributes to achieve adaptive age-related feature learning and improve age estimation robustness against the challenges above. Experimental results on multiple age estimation datasets (MORPH II, AgeDB, FG-NET, MegaAge-Asian, CLAP2016, UTK-Face, and LFW+) show that the proposed approach outperforms the state-of-the-art age estimation methods by a large margin. In addition, the proposed approach can generalize well to other human attributes estimation tasks, like height, weight, and body mass index (BMI) estimation.

## DataSet

- Age-Estimation：MORPH II, AgeDB, FG-NET, MegaAge-Asian, CLAP2016, UTK-Face, and LFW+.
- Height, weight and BMI estimation: VIP and VIPL-Mumo-3K-Demo (VIPL3KFace, which contains age, gender, height, weight attributes, and will be released later.)

## Train
#### Pretrain with IMDB-WIKI
```
  python main_wiki.py -i [IMDB-WIKI path] -e 100 -lr 0.01 -b 32 
```
#### Train DCDL model on MORPH II
```
  python main_dcdl_5folder.py -i [MORPH II path] -e 100 -lr 0.001 -b 32 -lw [wiki pretrained model's path] -n [folder_num]
```

## Citation

If you find this work useful, please cite our paper with the following bibtex:
```
@ARTICLE{SunTIFS2021,
  author={Haomiao Sun, Hongyu Pan, Hu Han, Shiguang Shan},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Deep Conditional Distribution Learning for Age Estimation}, 
  year={2021},
  volume={},
  number={},
  pages={},
  doi={10.1109/TIFS.2021.3114066}
}
```
---
Other code will be released later.
