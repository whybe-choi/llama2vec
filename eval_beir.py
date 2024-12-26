import os
import pathlib
import logging
from typing import List, Dict, Union

import torch
from transformers import AutoTokenizer, AutoModel
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES


logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)


class Llama2VecRetriever:
    def __init__(self, model_path: str = "BAAI/LLARA-beir", max_length: int = 512):
        self.model_path = model_path
        self.max_length = max_length
        self.model = AutoModel.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model.cuda()
        self.model.eval()

        self.prefix = '"'
        self.prefix_ids = self.tokenizer(self.prefix, return_tensors=None)["input_ids"]

        self.suffix_query = '", predict the following passage within eight words: <s9><s10><s11><s12><s13><s14><s15><s16>'  # prompt for EBAR
        self.suffix_query_ids = self.tokenizer(
            self.suffix_query, return_tensors=None, add_special_tokens=False
        )["input_ids"]
        self.query_max_len = (
            self.max_length - len(self.prefix_ids) - len(self.suffix_query_ids)
        )

        self.suffix_passage = '", summarize the above passage within eight words: <s1><s2><s3><s4><s5><s6><s7><s8>'  # prompt for EBAE
        self.suffix_passage_ids = self.tokenizer(
            self.suffix_passage, return_tensors=None, add_special_tokens=False
        )["input_ids"]
        self.passage_max_len = (
            self.max_length - len(self.prefix_ids) - len(self.suffix_passage_ids)
        )

    def get_query_inputs(self, queries: List[str]) -> dict:
        query_inputs = []
        for query in queries:
            inputs = self.tokenizer(
                query,
                return_tensors=None,
                truncation=True,
                max_length=self.query_max_len,
                add_special_tokens=False,
            )
            inputs["input_ids"] = (
                self.prefix_ids + inputs["input_ids"] + self.suffix_query_ids
            )
            inputs["attention_mask"] = [1] * len(inputs["input_ids"])
            query_inputs.append(inputs)

        return self.tokenizer.pad(
            query_inputs,
            padding="max_length",
            max_length=self.max_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

    def get_passage_inputs(self, passages: List[str]) -> dict:
        passage_inputs = []
        for passage in passages:
            inputs = self.tokenizer(
                passage,
                return_tensors=None,
                truncation=True,
                max_length=self.passage_max_len,
                add_special_tokens=False,
            )
            inputs["input_ids"] = (
                self.prefix_ids + inputs["input_ids"] + self.suffix_passage_ids
            )
            inputs["attention_mask"] = [1] * len(inputs["input_ids"])
            passage_inputs.append(inputs)

        return self.tokenizer.pad(
            passage_inputs,
            padding="max_length",
            max_length=self.max_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

    @torch.no_grad()
    def encode_queries(
        self, queries: List[str], batch_size: int = 8, **kwargs
    ) -> torch.Tensor:
        embeddings = []
        for start_idx in range(0, len(queries), batch_size):
            batch_queries = queries[start_idx : start_idx + batch_size]
            batch_inputs = self.get_query_inputs(batch_queries)
            batch_inputs = {k: v.cuda() for k, v in batch_inputs.items()}

            outputs = self.model(
                **batch_inputs, return_dict=True, output_hidden_states=True
            )
            batch_embeddings = outputs.hidden_states[-1][:, -8:, :]
            batch_embeddings = torch.mean(batch_embeddings, dim=1)
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, dim=-1)
            embeddings.append(batch_embeddings.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        logger.info(f"Embeddings shape: {embeddings.shape}")
        return embeddings

    @torch.no_grad()
    def encode_corpus(
        self,
        corpus: Union[List[Dict[str, str]], Dict[str, List]],
        batch_size: int = 8,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(corpus, dict):
            passages = [doc.strip() for doc in corpus["text"]]
        else:
            passages = [doc["text"].strip() for doc in corpus]

        embeddings = []
        for start_idx in range(0, len(passages), batch_size):
            batch_passages = passages[start_idx : start_idx + batch_size]
            batch_inputs = self.get_passage_inputs(batch_passages)
            batch_inputs = {k: v.cuda() for k, v in batch_inputs.items()}

            outputs = self.model(
                **batch_inputs, return_dict=True, output_hidden_states=True
            )
            batch_embeddings = outputs.hidden_states[-1][:, -8:, :]
            batch_embeddings = torch.mean(batch_embeddings, dim=1)
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, dim=-1)
            embeddings.append(batch_embeddings.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        logger.info(f"Embeddings shape: {embeddings.shape}")

        return embeddings


# Download and load dataset for biomedical information retrieval
# "nfcorpus", "scifact", "scidocs", "trec-covid"
dataset = "nfcorpus"
url = (
    f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
)
output_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "eval_datasets")
data_path = util.download_and_unzip(url, output_dir)

# Load corpus, queries, and qrels
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

model_path = "BAAI/LLARA-beir"
logging.info(f"Loading model: {model_path}")

model = DRES(Llama2VecRetriever(model_path=model_path), batch_size=8)
retriever = EvaluateRetrieval(model, score_function="dot")
results = retriever.retrieve(corpus, queries)

# Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K where k = [1,3,5,10,100,1000]
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
