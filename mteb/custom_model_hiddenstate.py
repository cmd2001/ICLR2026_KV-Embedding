import mteb
from mteb.encoder_interface import PromptType
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
import tqdm
import torch
import torch.nn.functional as F



# CACHE_DIR = "../models_storage"
model_name = "meta-llama/Llama-3.1-8B-Instruct"
batch_size = 8
max_length = 512

tot_layers = 32
n_layers = 32 # use KV cache from the last n layers
selected_layers = -1 # -1 means last n_layers, otherwise use the list

class MTEBHiddenAsEmbedding:
    """
    Use transformer hidden states as sentence embeddings.

    Config knobs:
      pos_agg:   "mean" | "last" | "cls"      (aggregate across sequence)
      layer_agg: "mean" | "sum" | "flatten" | "last"    (aggregate across layers)
    Output shape depends on agg choices.
    """

    def __init__(
        self,
        model, tokenizer, selected_layers,
        pos_agg="mean",
        layer_agg="mean",
        normalize_embeddings=False,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.selected_layers = list(selected_layers)
        self.pos_agg = pos_agg
        self.layer_agg = layer_agg
        self.normalize_embeddings = normalize_embeddings

    @torch.inference_mode()
    def encode(self, sentences, batch_size=1, convert_to_numpy=True, show_progress_bar=False):
        if isinstance(sentences, str):
            sentences = [sentences]

        outs = []
        iterator = range(0, len(sentences), batch_size)
        if show_progress_bar:
            iterator = tqdm.tqdm(iterator)

        for i in iterator:
            batch = sentences[i:i + batch_size]
            embs = self._encode_batch(batch)
            if self.normalize_embeddings:
                embs = F.normalize(embs, p=2, dim=-1)
            if convert_to_numpy:
                embs = embs.detach().cpu().float().numpy()
            outs.append(embs)

        if convert_to_numpy:
            return np.concatenate(outs, axis=0)
        return torch.cat(outs, dim=0)

    def _encode_batch(self, texts):
        tok = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        primary = next(self.model.parameters()).device
        input_ids = tok["input_ids"].to(primary)
        attention_mask = tok["attention_mask"].to(primary)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            output_attentions=False,
        )
        hidden_states = out.hidden_states  # tuple of length (num_layers + 1)

        layer_vecs = []

        for li in self.selected_layers:
            # hidden_states[0] is embeddings; [1..L] are transformer blocks.
            # Assume li is 0-based layer index (0..L-1) to match KV indices.
            hs = hidden_states[li + 1]       # (B, T, D)
            B, T, D = hs.shape

            attn_T = attention_mask[:, :T].to(device=hs.device)
            mask = attn_T.unsqueeze(-1).to(dtype=hs.dtype)  # (B, T, 1)

            def pos_pool(x):
                # x: (B, T, D) on hs.device
                if self.pos_agg == "last":
                    lengths = attn_T.sum(dim=1)                   # (B,)
                    idx = (lengths - 1).clamp_min(0).long()      # (B,)
                    idx3 = idx.view(B, 1, 1).expand(B, 1, D)     # (B,1,D)
                    return x.gather(1, idx3).squeeze(1)          # (B,D)
                elif self.pos_agg == "cls":
                    first_idx = attn_T.argmax(dim=1).long()       # (B,)
                    idx3 = first_idx.view(B, 1, 1).expand(B, 1, D)
                    return x.gather(1, idx3).squeeze(1)           # (B,D)
                else:  # "mean"
                    denom = mask.sum(dim=1).clamp_min(1e-6)       # (B,1)
                    return (x * mask).sum(dim=1) / denom          # (B,D)

            h_layer = pos_pool(hs)          # (B, D)
            layer_vecs.append(h_layer.to("cpu"))

        # Layer aggregation on CPU
        if self.layer_agg == "flatten":
            sent_vec = torch.cat(layer_vecs, dim=-1)               # (B, L*D_selected)
        elif self.layer_agg == "sum":
            sent_vec = torch.stack(layer_vecs, dim=0).sum(dim=0)   # (B, D)
        elif self.layer_agg == "last":
            sent_vec = layer_vecs[-1]                              # (B, D)
        else:  # "mean"
            sent_vec = torch.stack(layer_vecs, dim=0).mean(dim=0)  # (B, D)

        return sent_vec  # on CPU


class CustomModel:
    def __init__(self, model_name=model_name, tot_layers=tot_layers, n_layers=4, selected_layers=-1):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if isinstance(selected_layers, int) and selected_layers == -1:
            # last n_layers, consistent with your KV code, 0-based indices
            selected_layers = list(range(tot_layers - n_layers, tot_layers))
        self.selected_layers = selected_layers

        # swap to hidden-state embedder here
        self.hidden_embedder = MTEBHiddenAsEmbedding(
            self.model,
            self.tokenizer,
            self.selected_layers,
            pos_agg="mean",
            layer_agg="mean",
            normalize_embeddings=True,
        )

    def encode(self, sentences, batch_size=1, convert_to_numpy=True, **_):
        return self.hidden_embedder.encode(
            sentences,
            batch_size=batch_size,
            convert_to_numpy=convert_to_numpy,
        )


# evaluating the model:
model = CustomModel()
tasks = mteb.get_tasks(tasks=["AmazonCounterfactualClassification", "DBpediaClassification", "TweetTopicSingleClassification", "FinancialPhrasebankClassification"])
evaluation = mteb.MTEB(tasks=tasks)
evaluation.run(model)