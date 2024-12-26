import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from scipy.stats import spearmanr


class TestDatasetForEmbedding(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512, use_next_prompt=True):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_next_prompt = use_next_prompt
        self.prefix = '"'
        self.suffix = [
            '", summarize the above passage within eight words: <s1><s2><s3><s4><s5><s6><s7><s8>',
            '", predict the following passage within eight words: <s9><s10><s11><s12><s13><s14><s15><s16>',
        ]
        self.prefix_ids = self.tokenizer(
            self.prefix,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )["input_ids"]
        self.suffix_ids = self.tokenizer(
            self.suffix,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
            add_special_tokens=False,
        )["input_ids"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]

        if self.use_next_prompt:
            suffix_ids = self.suffix_ids[1]  # Predict the next passage
        else:
            suffix_ids = self.suffix_ids[0]  # Summarize the passage

        inputs = self.tokenizer(
            text,
            return_tensors=None,
            truncation=True,
            max_length=self.max_length - len(suffix_ids) - len(self.prefix_ids),
            add_special_tokens=False,
        )
        inputs["input_ids"] = self.prefix_ids + inputs["input_ids"] + suffix_ids
        inputs["attention_mask"] = [1] * len(inputs["input_ids"])

        return self.tokenizer.pad(
            inputs,
            padding="max_length",
            max_length=self.max_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )


def get_embeddings(model, dataloader, desc):
    embeddings = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            inputs = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**inputs, return_dict=True, output_hidden_states=True)
            embedding = outputs.hidden_states[-1][:, -8:, :]  # Use last 8 tokens
            embedding = torch.mean(embedding, dim=1)  # Mean pooling
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
            embeddings.append(embedding.cpu())

    return torch.cat(embeddings, dim=0)


def compute_spearman_correlation(predictions, ground_truths):
    return spearmanr(predictions, ground_truths).correlation


def main():
    model_name = "BAAI/LLARA-beir"

    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.cuda()

    dataset = load_dataset("tabilab/biosses", split="train")

    sentence1 = dataset["sentence1"]
    sentence2 = dataset["sentence2"]
    scores = dataset["score"]  # Ground-truth scores (0~4)

    # In the paper, SELF-SELF(S2S) is used for paraphrased documents, and NEXT-NEXT(N2N) is used for parapharased short texts.
    # The sentence pairs in BIOSSES were selected from citing sentences, i.e. sentences that have a citation to a reference article.
    # so we use NEXT-NEXT(N2N)

    use_next_prompt = True
    print(f"Using {'NEXT-NEXT' if use_next_prompt else 'SELF-SELF'} prompt")

    dataset1 = TestDatasetForEmbedding(
        sentence1, tokenizer, use_next_prompt=use_next_prompt
    )
    dataset2 = TestDatasetForEmbedding(
        sentence2, tokenizer, use_next_prompt=use_next_prompt
    )

    dataloader1 = DataLoader(dataset1, batch_size=8, shuffle=False)
    dataloader2 = DataLoader(dataset2, batch_size=8, shuffle=False)

    embeddings1 = get_embeddings(model, dataloader1, desc="Embedding Sentence1")
    embeddings2 = get_embeddings(model, dataloader2, desc="Embedding Sentence2")

    cosine_similarities = (
        torch.nn.functional.cosine_similarity(embeddings1, embeddings2).cpu().numpy()
    )

    # Compute Spearman correlation
    spearman_corr = compute_spearman_correlation(cosine_similarities, scores)

    print(f"Spearman Correlation: {spearman_corr:.4f}")


if __name__ == "__main__":
    main()
