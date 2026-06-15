# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import transformers

from sam_audio.model.config import T5EncoderConfig


class TextEmbeddingDiskCache:
    def __init__(
        self,
        cache_dir: str | Path,
        namespace: str,
        max_entries: int = 100,
    ):
        self.namespace = self._safe_namespace(namespace)
        self.cache_dir = Path(cache_dir) / self.namespace
        self.index_path = self.cache_dir / "index.json"
        self.max_entries = max_entries

    @staticmethod
    def _safe_namespace(namespace: str) -> str:
        return "".join(
            char if char.isalnum() or char in ("-", "_", ".") else "_"
            for char in namespace
        )

    @staticmethod
    def _key(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def get(
        self,
        prompt: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        key = self._key(prompt)
        path = self.cache_dir / f"{key}.pt"
        if not path.exists():
            return None

        try:
            data = torch.load(path, map_location="cpu")
            features = data.get("features")
            mask = data.get("mask")
            if (
                data.get("prompt") != prompt
                or not isinstance(features, torch.Tensor)
                or not isinstance(mask, torch.Tensor)
                or features.dim() != 2
                or mask.dim() != 1
                or features.size(0) != mask.size(0)
            ):
                raise ValueError(f"Invalid cached text embedding entry: {path}")

            return (
                features.to(device=device, dtype=dtype),
                mask.to(device=device, dtype=torch.bool),
            )
        except Exception as exc:
            print(f"Failed to load SAM Audio text embedding cache entry: {exc!r}")
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            return None

    def put(
        self,
        prompt: str,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            key = self._key(prompt)
            path = self.cache_dir / f"{key}.pt"

            fd, tmp_name = tempfile.mkstemp(
                dir=self.cache_dir,
                prefix=f".{key}.",
                suffix=".tmp",
            )
            os.close(fd)
            try:
                torch.save(
                    {
                        "prompt": prompt,
                        "features": features.detach().cpu(),
                        "mask": mask.detach().cpu(),
                    },
                    tmp_name,
                )
                os.replace(tmp_name, path)
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)

            self._record_key(key)
        except Exception as exc:
            print(f"Failed to write SAM Audio text embedding cache entry: {exc!r}")

    def _read_index(self) -> list[str]:
        if self.index_path.exists():
            try:
                with self.index_path.open() as fin:
                    data = json.load(fin)
                if isinstance(data, list):
                    return [
                        item
                        for item in data
                        if isinstance(item, str)
                        and (self.cache_dir / f"{item}.pt").exists()
                    ]
            except Exception as exc:
                print(
                    f"Failed to read SAM Audio text embedding cache index: {exc!r}"
                )

        return [
            path.stem
            for path in sorted(
                self.cache_dir.glob("*.pt"),
                key=lambda item: item.stat().st_mtime,
            )
        ]

    def _write_index(self, entries: list[str]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            dir=self.cache_dir,
            prefix=".index.",
            suffix=".tmp",
            delete=False,
        ) as fout:
            json.dump(entries, fout)
            tmp_name = fout.name
        os.replace(tmp_name, self.index_path)

    def _record_key(self, key: str) -> None:
        entries = self._read_index()
        if key not in entries:
            entries.append(key)

        while len(entries) > self.max_entries:
            evicted = entries.pop(0)
            try:
                (self.cache_dir / f"{evicted}.pt").unlink()
            except FileNotFoundError:
                pass

        self._write_index(entries)


class T5TextEncoder(torch.nn.Module):
    def __init__(self, cfg: T5EncoderConfig):
        super().__init__()
        self.model = transformers.T5EncoderModel.from_pretrained(cfg.name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.name)
        self.pad_mode = cfg.pad_mode
        self.max_length = cfg.max_length
        self.disk_cache: Optional[TextEmbeddingDiskCache] = None

    def configure_disk_cache(
        self,
        cache_dir: str | Path,
        namespace: str,
        max_entries: int = 100,
    ) -> None:
        self.disk_cache = TextEmbeddingDiskCache(
            cache_dir=cache_dir,
            namespace=namespace,
            max_entries=max_entries,
        )

    def forward(
        self,
        texts: list[str],
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        if self.disk_cache is not None:
            cached = self._forward_with_disk_cache(
                texts,
                device=device,
                dtype=dtype,
                verbose=verbose,
            )
            if cached is not None:
                return cached

        return self._encode_uncached(texts, device=device)

    def _encode_uncached(
        self,
        texts: list[str],
        *,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=self.pad_mode,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        res = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )["last_hidden_state"]

        return res, attention_mask.bool()

    def _forward_with_disk_cache(
        self,
        texts: list[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        verbose: bool = False,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not texts:
            return None

        assert self.disk_cache is not None

        unique_texts = list(dict.fromkeys(texts))
        encoded_by_text: dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        missing_texts: list[str] = []

        for text in unique_texts:
            cache_start = time.perf_counter()
            cached = self.disk_cache.get(text, device=device, dtype=dtype)
            cache_lookup_ms = int((time.perf_counter() - cache_start) * 1000)
            if cached is None:
                if verbose:
                    print(
                        "SAM Audio text embedding cache miss "
                        f"namespace={self.disk_cache.namespace} "
                        f"key={TextEmbeddingDiskCache._key(text)[:12]} "
                        f"cache_lookup_ms={cache_lookup_ms} "
                        f"prompt={text!r}"
                    )
                missing_texts.append(text)
            else:
                if verbose:
                    print(
                        "SAM Audio text embedding cache hit "
                        f"namespace={self.disk_cache.namespace} "
                        f"key={TextEmbeddingDiskCache._key(text)[:12]} "
                        f"cache_load_ms={cache_lookup_ms} "
                        f"prompt={text!r}"
                    )
                encoded_by_text[text] = cached

        if missing_texts:
            embedding_start = time.perf_counter()
            features, masks = self._encode_uncached(missing_texts, device=device)
            text_embedding_compute_ms = int(
                (time.perf_counter() - embedding_start) * 1000
            )
            if verbose:
                print(
                    "SAM Audio text embedding cache computed missing text embeddings "
                    f"namespace={self.disk_cache.namespace} "
                    f"count={len(missing_texts)} "
                    f"text_embedding_compute_ms={text_embedding_compute_ms} "
                    f"prompts={missing_texts!r}"
                )
            for idx, text in enumerate(missing_texts):
                valid_len = int(masks[idx].sum().item())
                text_features = features[idx, :valid_len].detach()
                text_mask = masks[idx, :valid_len].detach()
                cache_store_start = time.perf_counter()
                self.disk_cache.put(text, text_features, text_mask)
                cache_store_ms = int(
                    (time.perf_counter() - cache_store_start) * 1000
                )
                if verbose:
                    print(
                        "SAM Audio text embedding cache stored embedding "
                        f"namespace={self.disk_cache.namespace} "
                        f"key={TextEmbeddingDiskCache._key(text)[:12]} "
                        f"cache_store_ms={cache_store_ms} "
                        f"prompt={text!r}"
                    )
                encoded_by_text[text] = (text_features, text_mask)

        max_len = max(encoded_by_text[text][0].size(0) for text in texts)
        hidden_size = encoded_by_text[texts[0]][0].size(1)
        text_features = torch.zeros(
            len(texts),
            max_len,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        text_mask = torch.zeros(
            len(texts),
            max_len,
            device=device,
            dtype=torch.bool,
        )

        for idx, text in enumerate(texts):
            features, mask = encoded_by_text[text]
            length = features.size(0)
            text_features[idx, :length] = features
            text_mask[idx, :length] = mask

        return text_features, text_mask
