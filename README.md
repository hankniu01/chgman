# chgman

Dataset and code for parper: **Composition-based Heterogeneous Graph Multi-channel Attention Network for Multi-aspect Multi-sentiment Classification**

## Requirement

- Python 3.8.3
- Pytorch 1.6.0
- Huggingface transformers 4.5.1

## Usage

For Training model for each dataset:

```bash
sh run.sh
```

If you want to get the trained models, please contact me by email.

## MAMS dataset

For MAMS dataset, please refer to https://github.com/siat-nlp/MAMS-for-ABSA


## Citation

If this work is helpful, please cite as:
```
@inproceedings{niu-etal-2022-composition,
    title = "Composition-based Heterogeneous Graph Multi-channel Attention Network for Multi-aspect Multi-sentiment Classification",
    author = "Niu, Hao  and
      Xiong, Yun  and
      Gao, Jian  and
      Miao, Zhongchen  and
      Wang, Xiaosu  and
      Ren, Hongrun  and
      Zhang, Yao  and
      Zhu, Yangyong",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.594",
    pages = "6827--6836",
    abstract = "Aspect-based sentiment analysis (ABSA) has drawn more and more attention because of its extensive applications. However, towards the sentence carried with more than one aspect, most existing works generate an aspect-specific sentence representation for each aspect term to predict sentiment polarity, which neglects the sentiment relationship among aspect terms. Besides, most current ABSA methods focus on sentences containing only one aspect term or multiple aspect terms with the same sentiment polarity, which makes ABSA degenerate into sentence-level sentiment analysis. In this paper, to deal with this problem, we construct a heterogeneous graph to model inter-aspect relationships and aspect-context relationships simultaneously and propose a novel Composition-based Heterogeneous Graph Multi-channel Attention Network (CHGMAN) to encode the constructed heterogeneous graph. Meanwhile, we conduct extensive experiments on three datasets: MAMSATSA, Rest14, and Laptop14, experimental results show the effectiveness of our method.",
}
```


