from pathlib import Path
from typing import Iterable, List, Optional

from sentence_transformers import SentenceTransformer
import torch

from MAR.Utils.offline_embeddings import load_query_embeddings

def get_sentence_embedding(sentence):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(sentence)
    return torch.tensor(embeddings)

class SentenceEncoder(torch.nn.Module):
    def __init__(self, device=None, query_embeddings_csv: Optional[str] = None):
        super().__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=self.device)
        embeddings_dir = Path(__file__).resolve().parents[2] / "Datasets" / "embeddings"
        query_path = query_embeddings_csv or str(embeddings_dir / "query_embeddings.csv")
        self.offline_query_embeddings = load_query_embeddings(
            query_path, device=torch.device(self.device), dtype=torch.float32
        )
        
    def forward(
        self,
        sentence,
        query_ids: Optional[Iterable[object]] = None,
        dataset_name: Optional[str] = None,
    ):
        if isinstance(sentence, str):
            sentences: List[str] = [sentence]
        else:
            sentences = list(sentence)

        if len(sentences) == 0:
            return torch.tensor([]).to(self.device)

        if query_ids is None or dataset_name is None:
            embeddings = self.model.encode(sentences, convert_to_tensor=True, device=self.device)
            # sentence-transformers may run under torch.inference_mode(), producing inference tensors.
            # Convert to a normal tensor so downstream trainable modules can save it for backward.
            return embeddings.clone()

        ids = list(query_ids)
        if len(ids) != len(sentences):
            embeddings = self.model.encode(sentences, convert_to_tensor=True, device=self.device)
            return embeddings.clone()

        dataset_key = str(dataset_name).strip().lower() if dataset_name is not None else ""
        if not dataset_key:
            embeddings = self.model.encode(sentences, convert_to_tensor=True, device=self.device)
            return embeddings.clone()

        embeddings_out: List[Optional[torch.Tensor]] = [None] * len(sentences)
        missing_texts: List[str] = []
        missing_idx: List[int] = []
        for idx, (query_id, text) in enumerate(zip(ids, sentences)):
            cached = None
            try:
                query_id_int = int(query_id)
            except (TypeError, ValueError):
                query_id_int = None
            if query_id_int is not None:
                cached = self.offline_query_embeddings.get((dataset_key, query_id_int))
            if cached is not None:
                embeddings_out[idx] = cached
            else:
                missing_idx.append(idx)
                missing_texts.append(text)

        if missing_texts:
            computed = self.model.encode(missing_texts, convert_to_tensor=True, device=self.device)
            if computed.dim() == 1:
                computed = computed.unsqueeze(0)
            for offset, idx in enumerate(missing_idx):
                embeddings_out[idx] = computed[offset]

        stacked = torch.stack([item for item in embeddings_out if item is not None], dim=0)
        # Ensure ordering matches input by re-gathering in index order.
        if stacked.size(0) != len(sentences):
            ordered = [embeddings_out[idx] for idx in range(len(sentences))]
            stacked = torch.stack(ordered, dim=0)
        return stacked.clone()
