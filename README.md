# Re-implementation of Llama2Vec with BMRETRIEVER recipe

## Environment
```bash
conda create -n llama2vec python=3.10
conda activate llama2vec
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Performance
|          | **NFCorpus** | **SciFact** | **SciDocs** | **Trec-COVID** | **BIOSSES** | **Avg. Retr.** | **Avg. All** | 
|----------|--------------|-------------|-------------|----------------|-------------|----------------|----------------|
|Llama2Vec| 0.382 | 0.754 | 0.181 | 0.832 | 0.852(N2N) | 0.537 | 0.600 |
|BMRETRIEVER-7B| 0.364 | 0.778 | 0.201 | 0.861 | 0.847 | 0.551 | 0.610 |
|BM-Llama2Vec| 0.371 | 0.731 | 0.186 | 0.811 | 0.834(N2N) | 0.525 | 0.587 |

## In-domain Experiment

To evaluate the in-domain efficacy of the Llama2Vec approach, we also conducted experiments on the same biomedical fine-tuning datasets used by BMRETRIEVER. However, our results did not show any significant performance improvements when applying Llama2Vec under these in-domain settings. As it remains unclear whether this is due to Llama2Vec’s inherent limitations in in-domain performance or suboptimal hyperparameter/training configurations, we plan to re-run experiments with adjusted settings in the future to verify.

## Citation
```
@misc{li2023makinglargelanguagemodels,
      title={Making Large Language Models A Better Foundation For Dense Retrieval}, 
      author={Chaofan Li and Zheng Liu and Shitao Xiao and Yingxia Shao},
      year={2023},
      eprint={2312.15503},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2312.15503}, 
}

@article{xu2024bmretriever,
  title={Bmretriever: Tuning large language models as better biomedical text retrievers},
  author={Xu, Ran and Shi, Wenqi and Yu, Yue and Zhuang, Yuchen and Zhu, Yanqiao and Wang, May D and Ho, Joyce C and Zhang, Chao and Yang, Carl},
  journal={arXiv preprint arXiv:2404.18443},
  year={2024}
}
```


