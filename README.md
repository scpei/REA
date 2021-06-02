# REA: Robust Cross-lingual Entity Alignment Between Knowledge Graphs

Our proposed method REA (Robust Entity Alignment) consists of two components: noise detection and noise-aware entity alignment. 

The noise detection is designed by following the adversarial training principle. 
The noise-aware entity alignment is devised by leveraging graph neural network based knowledge graph encoder as the core. In order to mutually boost the performance of the two components, we propose a unified reinforced training strategy to combine them.

REA is a plug-and-play strategy to mitigate the effect of noise in the given labeled entity pairs for entity alignment problem. The idea also can be easily developed for other alignment algorithms.

Contact: Shichao Pei (shichao.pei@kaust.edu.sa)

## Environment

* python>=3.5
* tensorflow>=1.10.1
* scipy>=1.1.0
* networkx>=2.2

## Usage

```
python3 train.py --lang zh_en
```

Datasets are from [JAPE](https://github.com/nju-websoft/JAPE).

## Reference
Please refer to our paper. 

    @inproceedings{pei2020rea,
      title={Rea: Robust cross-lingual entity alignment between knowledge graphs},
      author={Pei, Shichao and Yu, Lu and Yu, Guoxian and Zhang, Xiangliang},
      booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
      pages={2175--2184},
      year={2020}
    }

