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
n_layers = 4 # use KV cache from the last n layers
selected_layers = -1 # -1 means last n_layers, otherwise use the list

class MTEBKVAsEmbedding:
    """
    Use the attention KV cache itself as sentence embeddings.

    Config knobs:
      kv_part:   "k" | "v" | "kv_cat"
      pos_agg:   "mean" | "last" | "cls"      (aggregate across sequence)
      head_agg:  "mean" | "flatten"           (aggregate across heads)
      layer_agg: "mean" | "sum" | "flatten" | "last"    (aggregate across layers)
    Output shape depends on agg choices (documented below).
    """

    def __init__(
        self,
        model, tokenizer, selected_layers,
        kv_part="kv_cat",           # V often works well; try "kv_cat" too
        pos_agg="mean",
        head_agg="flatten",
        layer_agg="last",
        normalize_embeddings=False
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.selected_layers = list(selected_layers)
        self.kv_part = kv_part
        self.pos_agg = pos_agg
        self.head_agg = head_agg
        self.layer_agg = layer_agg
        self.normalize_embeddings = normalize_embeddings

    @torch.inference_mode()
    def encode(self, sentences, batch_size=1, convert_to_numpy=True, show_progress_bar=False):
        if isinstance(sentences, str):
            sentences = [sentences]

        outs = []
        for i in tqdm.tqdm(range(0, len(sentences), batch_size)):
            batch = sentences[i:i+batch_size]
            embs = self._encode_batch(batch)
            if self.normalize_embeddings:
                embs = F.normalize(embs, p=2, dim=-1)
            if convert_to_numpy:
                embs = embs.detach().cpu().float().numpy()
            outs.append(embs)

        if convert_to_numpy:
            import numpy as np
            return np.concatenate(outs, axis=0)
        return torch.cat(outs, dim=0)

    def _encode_batch(self, texts):
        tok = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        # Let HF dispatch handle sharding; inputs can be on the "primary" device (or CPU).
        # Moving to the first module's device is fine:
        primary = next(self.model.parameters()).device
        input_ids = tok["input_ids"].to(primary)
        attention_mask = tok["attention_mask"].to(primary)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=False,
            output_attentions=False,
        )
        pkv = out.past_key_values

        layer_vecs = []

        for li in self.selected_layers:
            kv_tuple = pkv[li]
            k = kv_tuple[0]
            v = kv_tuple[1]

            # Normalize layout to (B, H, T, D)
            def to_bhtd(t):
                # If it’s (B, T, H, D), transpose to (B, H, T, D)
                return t.transpose(1, 2) if (t.dim() == 4 and t.shape[1] == attention_mask.shape[1]) else t
            k = to_bhtd(k)
            v = to_bhtd(v)

            # Everything below must run on the SAME device as the tensors we pool from:
            # choose the ref tensor depending on kv_part
            ref = v if self.kv_part != "k" else k
            B, H, T, D = ref.shape

            # Build mask on ref.device and ref.dtype
            attn_T = attention_mask[:, :T].to(device=ref.device)
            mask = attn_T.unsqueeze(1).unsqueeze(-1).to(dtype=ref.dtype)  # (B,1,T,1)

            def pos_pool(x):
                # x: (B, H, T, D) on ref.device
                if self.pos_agg == "last":
                    lengths = attn_T.sum(dim=1)                      # (B,) on ref.device
                    idx = (lengths - 1).clamp_min(0).long()          # (B,)
                    # build gather index on same device
                    idx4 = idx.view(B, 1, 1, 1).expand(B, H, 1, D)   # (B,H,1,D)
                    return x.gather(2, idx4).squeeze(2)              # (B,H,D)
                elif self.pos_agg == "cls":
                    first_idx = attn_T.argmax(dim=1).long()          # (B,) on ref.device
                    idx4 = first_idx.view(B, 1, 1, 1).expand(B, H, 1, D)
                    return x.gather(2, idx4).squeeze(2)              # (B,H,D)
                else:  # "mean"
                    denom = mask.sum(dim=2).clamp_min(1e-6)          # (B,1,1)
                    return (x * mask).sum(dim=2) / denom             # (B,H,D)

            if self.kv_part == "k":
                h_hd = pos_pool(k)
            elif self.kv_part == "kv_cat":
                h_hd = torch.cat([pos_pool(k), pos_pool(v)], dim=-1)  # (B,H,2D)
            else:
                h_hd = pos_pool(v)

            # Head aggregation
            if self.head_agg == "flatten":
                h_layer = h_hd.reshape(B, -1)   # (B, H*D) or H*2D
            else:
                h_layer = h_hd.mean(dim=1)      # (B, D) or (B, 2D)

            # IMPORTANT: with model parallel, different layers are on different GPUs.
            # Move each layer vector to CPU (or a single chosen device) BEFORE stacking.
            layer_vecs.append(h_layer.to("cpu"))

        # Layer aggregation on a common device (CPU here)
        if self.layer_agg == "flatten":
            sent_vec = torch.cat(layer_vecs, dim=-1)              # (B, L*feat)
        elif self.layer_agg == "sum":
            sent_vec = torch.stack(layer_vecs, dim=0).sum(dim=0)  # (B, feat)
        elif self.layer_agg == "last":
            sent_vec = layer_vecs[-1]                              # (B, feat)
        else:  # "mean"
            sent_vec = torch.stack(layer_vecs, dim=0).mean(dim=0) # (B, feat)

        return sent_vec  # on CPU

class CustomModel:
    def __init__(self, model_name=model_name, tot_layers=tot_layers, n_layers=4, selected_layers=-1):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR, device_map="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR, trust_remote_code=True, padding_side='left')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if type(selected_layers) == int and selected_layers == -1:
            selected_layers = list(range(tot_layers - n_layers, tot_layers))
        self.selected_layers = selected_layers
        # add:
        self.kv_embedder = MTEBKVAsEmbedding(
            self.model, self.tokenizer, self.selected_layers,
            kv_part="kv_cat", pos_agg="mean", head_agg="mean", layer_agg="mean", normalize_embeddings=True
        )
    def encode(self, sentences, batch_size=1, convert_to_numpy=True, **_):
        return self.kv_embedder.encode(sentences, batch_size=batch_size, convert_to_numpy=convert_to_numpy)

# evaluating the model:
model = CustomModel()
tasks = mteb.get_tasks(tasks=["AmazonCounterfactualClassification", "DBpediaClassification", "TweetTopicSingleClassification",  "FinancialPhrasebankClassification"])
evaluation = mteb.MTEB(tasks=tasks)
evaluation.run(model)