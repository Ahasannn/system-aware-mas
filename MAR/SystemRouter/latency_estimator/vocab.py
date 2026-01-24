from __future__ import annotations

from typing import Dict, Iterable, List


class CategoricalVocab:
    def __init__(self, tokens: Iterable[str], *, unk_token: str = "<unk>") -> None:
        unique_tokens: List[str] = []
        seen = set()
        for token in tokens:
            if token in seen:
                continue
            unique_tokens.append(token)
            seen.add(token)
        self.tokens = list(unique_tokens)
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.unk_token = unk_token
        if self.unk_token not in self.token_to_id:
            self.token_to_id[self.unk_token] = len(self.tokens)
            self.tokens.append(self.unk_token)
        self.unk_id = self.token_to_id[self.unk_token]

    def encode(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_id)

    def decode(self, idx: int) -> str:
        if idx < 0 or idx >= len(self.tokens):
            return self.unk_token
        return self.tokens[idx]

    def __len__(self) -> int:
        return len(self.tokens)

    def to_dict(self) -> Dict[str, object]:
        return {"tokens": list(self.tokens), "unk_token": self.unk_token}

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "CategoricalVocab":
        tokens = data.get("tokens", [])
        unk_token = str(data.get("unk_token", "<unk>"))
        return cls(list(tokens), unk_token=unk_token)
