# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from core.audio_visual_encoder import PEAudioFrame, PEAudioFrameTransform
from torchdiffeq import odeint

from sam_audio.model.align import AlignModalities
from sam_audio.model.base import BaseModel
from sam_audio.model.codec import DACVAE
from sam_audio.model.config import SAMAudioConfig
from sam_audio.model.text_encoder import T5TextEncoder
from sam_audio.model.transformer import DiT
from sam_audio.model.vision_encoder import PerceptionEncoder
from sam_audio.processor import Batch
from sam_audio.ranking import create_ranker

DFLT_ODE_OPT = {"method": "midpoint", "options": {"step_size": 2 / 32}}


class SinusoidalEmbedding(torch.nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        inv_freq = torch.exp(
            -math.log(theta) * torch.arange(half_dim).float() / half_dim
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, pos=None):
        if pos is None:
            seq_len, device = x.shape[1], x.device
            pos = torch.arange(seq_len, device=device)

        emb = torch.einsum("i, j -> i j", pos, self.inv_freq)
        emb = torch.cat((emb.cos(), emb.sin()), dim=-1)
        return emb


class EmbedAnchors(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, out_dim: int):
        super().__init__()
        self.embed = torch.nn.Embedding(
            num_embeddings + 1, embedding_dim, padding_idx=num_embeddings
        )
        self.gate = torch.nn.Parameter(torch.tensor([0.0]))
        self.proj = torch.nn.Linear(embedding_dim, out_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        anchor_ids: Optional[torch.Tensor] = None,
        anchor_alignment: Optional[torch.Tensor] = None,
    ):
        if anchor_ids is None:
            return x

        embs = self.embed(anchor_ids.gather(1, anchor_alignment))
        proj = self.proj(embs)
        return x + self.gate.tanh() * proj


@dataclass
class SeparationResult:
    target: torch.Tensor
    residual: torch.Tensor
    noise: torch.Tensor


class SAMAudio(BaseModel):
    config_cls = SAMAudioConfig
    revision = None

    def __init__(self, cfg: SAMAudioConfig):
        super().__init__()
        self.audio_codec = DACVAE(cfg.audio_codec)
        self.text_encoder = T5TextEncoder(cfg.text_encoder)
        self.vision_encoder = PerceptionEncoder(cfg.vision_encoder)
        self.transformer = DiT(cfg.transformer)
        self.proj = torch.nn.Linear(cfg.in_channels, cfg.transformer.dim)
        self.align_masked_video = AlignModalities(
            cfg.vision_encoder.dim, cfg.transformer.dim
        )
        self.embed_anchors = EmbedAnchors(
            cfg.num_anchors, cfg.anchor_embedding_dim, cfg.transformer.dim
        )
        self.memory_proj = torch.nn.Linear(cfg.text_encoder.dim, cfg.transformer.dim)
        self.timestep_emb = SinusoidalEmbedding(cfg.transformer.dim)
        self.visual_ranker = create_ranker(cfg.visual_ranker)
        self.text_ranker = create_ranker(cfg.text_ranker)
        if cfg.span_predictor is not None:
            self.span_predictor = PEAudioFrame.from_config(
                cfg.span_predictor, pretrained=True
            )
            self.span_predictor_transform = PEAudioFrameTransform.from_config(
                cfg.span_predictor
            )

    @property
    def sample_rate(self):
        return self.audio_codec.sample_rate

    def align_inputs(
        self,
        noisy_audio,
        audio_features: torch.Tensor,
        masked_video_features: Optional[torch.Tensor] = None,
        anchor_ids: Optional[torch.Tensor] = None,
        anchor_alignment: Optional[torch.Tensor] = None,
    ):
        x = torch.cat(
            [
                noisy_audio,
                torch.zeros_like(audio_features),
                audio_features,
            ],
            dim=2,
        )

        projected = self.proj(x)
        aligned = self.align_masked_video(projected, masked_video_features)
        aligned = self.embed_anchors(aligned, anchor_ids, anchor_alignment)
        return aligned

    def forward(
        self,
        noisy_audio: torch.Tensor,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        time: torch.Tensor,
        masked_video_features: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        anchor_ids: Optional[torch.Tensor] = None,
        anchor_alignment: Optional[torch.Tensor] = None,
        audio_pad_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for the model.  Represents one function evaluation of the ODE.
        In the below descriptions, B is batch size, T is sequence length, C is channel size.
        Note that the size of C and T may vary across arguments (ex. text_features vs. audio_features),
        it is used only to designate a Channel or time/sequence-length dimension respectively.

        Args:
            noisy_audio (torch.Tensor): Noisy audio input tensor (being denoised).
            audio_features (torch.Tensor): Clean audio features [B x T x C].
            text_features (torch.Tensor): Encoded text features tensor [B x T x C].
            time (torch.Tensor): Timestep tensor for positional encoding [B].
            masked_video_features (Optional[torch.Tensor], optional): Masked video features tensor. [B x C x T].
            text_mask (Optional[torch.Tensor], optional): Padding mask for text features. [B x T].
            anchor_ids (Optional[torch.Tensor], optional): Anchor IDs tensor. Defaults to None [B x T].
            anchor_alignment (Optional[torch.Tensor], optional): Anchor alignment tensor. B x T.
            audio_pad_mask (Optional[torch.Tensor], optional): Padding mask for audio input. [B x T].

        Returns:
            torch.Tensor
        """
        aligned_inputs = self.align_inputs(
            noisy_audio,
            audio_features,
            masked_video_features=masked_video_features,
            anchor_ids=anchor_ids,
            anchor_alignment=anchor_alignment,
        )

        memory = timestep_emb = self.timestep_emb(time, pos=time).unsqueeze(1)
        if text_features is not None:
            memory = self.memory_proj(text_features) + timestep_emb

        return self.transformer(
            aligned_inputs,
            time,
            padding_mask=audio_pad_mask,
            memory=memory,
            memory_padding_mask=text_mask,
        )

    def _get_audio_features(self, audios: torch.Tensor):
        with torch.autograd.profiler.record_function("sam_audio/audio_codec_encode"):
            audio_features = self.audio_codec(audios).transpose(1, 2)
        return torch.cat([audio_features, audio_features], dim=2)

    def _get_video_features(self, video, audio_features):
        B, T, _ = audio_features.shape
        if video is None:
            return audio_features.new_zeros(B, self.vision_encoder.dim, T)
        else:
            return self.vision_encoder(video).transpose(1, 2)

    def _repeat_for_reranking(self, tensor, candidates):
        if candidates > 1:
            B = tensor.size(0)
            rest = tensor.shape[1:]
            return (
                tensor.unsqueeze(1)
                .expand(B, candidates, *rest)
                .reshape(B * candidates, *rest)
            )
        else:
            return tensor

    def _unrepeat_from_reranking(self, tensor, candidates):
        return tensor[::candidates]

    def _get_forward_args(self, batch: Batch, candidates: int = 1):
        with torch.autograd.profiler.record_function("sam_audio/get_audio_features"):
            audio_features = self._get_audio_features(batch.audios)
        with torch.autograd.profiler.record_function("sam_audio/text_encoder"):
            text_features, text_mask = self.text_encoder(batch.descriptions)
        with torch.autograd.profiler.record_function("sam_audio/get_video_features"):
            masked_video_features = self._get_video_features(
                batch.masked_video, audio_features
            )

        return {
            "audio_features": self._repeat_for_reranking(audio_features, candidates),
            "text_features": self._repeat_for_reranking(text_features, candidates),
            "text_mask": self._repeat_for_reranking(text_mask, candidates),
            "masked_video_features": self._repeat_for_reranking(
                masked_video_features, candidates
            ),
            "anchor_ids": self._repeat_for_reranking(batch.anchor_ids, candidates),
            "anchor_alignment": self._repeat_for_reranking(
                batch.anchor_alignment, candidates
            ),
            "audio_pad_mask": self._repeat_for_reranking(
                batch.audio_pad_mask, candidates
            ),
        }

    def predict_spans(
        self, batch: Batch, audio_features: torch.Tensor, audio_pad_mask: torch.Tensor
    ) -> Batch:
        input = self.span_predictor_transform(text=batch.descriptions).to(
            audio_features.device
        )
        output = self.span_predictor(
            input_features=audio_features[:, :, :128],
            padding_mask=audio_pad_mask,
            return_spans=True,
            **input,
        )
        anchors = [[["+"] + anchor for anchor in anchors] for anchors in output.spans]
        batch.process_anchors(anchors)
        return batch

    @torch.inference_mode()
    def separate(
        self,
        batch: Batch,
        noise: Optional[torch.Tensor] = None,
        ode_opt: Dict[str, Any] = DFLT_ODE_OPT,
        reranking_candidates: int = 1,
        predict_spans: bool = False,
    ) -> SeparationResult:
        # Encode audio
        with torch.autograd.profiler.record_function("sam_audio/get_forward_args"):
            forward_args = self._get_forward_args(
                batch, candidates=reranking_candidates
            )

        if predict_spans and hasattr(self, "span_predictor") and batch.anchors is None:
            batch = self.predict_spans(
                batch=batch,
                audio_features=self._unrepeat_from_reranking(
                    forward_args["audio_features"], reranking_candidates
                ),
                audio_pad_mask=self._unrepeat_from_reranking(
                    forward_args["audio_pad_mask"], reranking_candidates
                ),
            )

        audio_features = forward_args["audio_features"]
        B, T, C = audio_features.shape
        C = C // 2  # we stack audio_features, so the actual channels is half

        if noise is None:
            with torch.autograd.profiler.record_function("sam_audio/noise_init"):
                noise = torch.randn_like(audio_features)

        def vector_field(t, noisy_audio):
            res = self.forward(
                noisy_audio=noisy_audio,
                time=t.expand(noisy_audio.size(0)),
                **forward_args,
            )
            return res

        with torch.autograd.profiler.record_function("sam_audio/odeint"):
            states = odeint(
                vector_field,
                noise,
                torch.tensor([0.0, 1.0], device=noise.device),
                **ode_opt,
            )
        generated_features = states[-1].transpose(1, 2)
        # generated_features has shape [B, 2C, T].  Reshape to stack along the batch dimension
        with torch.autograd.profiler.record_function("sam_audio/audio_codec_decode"):
            wavs = self.audio_codec.decode(
                generated_features.reshape(2 * B, C, T)
            ).view(B, 2, -1)

        bsz = wavs.size(0) // reranking_candidates
        sizes = self.audio_codec.feature_idx_to_wav_idx(batch.sizes)
        with torch.autograd.profiler.record_function("sam_audio/unbatch_outputs"):
            target_wavs = self.unbatch(
                wavs[:, 0].view(bsz, reranking_candidates, -1), sizes
            )
            residual_wavs = self.unbatch(
                wavs[:, 1].view(bsz, reranking_candidates, -1), sizes
            )

        if (
            reranking_candidates > 1
            and batch.masked_video is not None
            and self.visual_ranker is not None
        ):
            scores = self.visual_ranker(
                extracted_audio=target_wavs,
                videos=batch.masked_video,
                sample_rate=self.audio_codec.sample_rate,
            )
            idxs = scores.argmax(dim=1)
        elif reranking_candidates > 1 and self.text_ranker is not None:
            input_audio = [
                audio[:, :size].expand(reranking_candidates, -1)
                for audio, size in zip(batch.audios, sizes, strict=False)
            ]
            scores = self.text_ranker(
                extracted_audio=target_wavs,
                input_audio=input_audio,
                descriptions=batch.descriptions,
                sample_rate=self.audio_codec.sample_rate,
            )
            idxs = scores.argmax(dim=1)
        else:
            idxs = torch.zeros(bsz, dtype=torch.long, device=noise.device)

        return SeparationResult(
            target=[wav[idx] for wav, idx in zip(target_wavs, idxs, strict=False)],
            residual=[
                wavs[idx] for wavs, idx in zip(residual_wavs, idxs, strict=False)
            ],
            noise=noise,
        )

    def unbatch(self, wavs: torch.Tensor, sizes: torch.Tensor, time_dim: int = -1):
        result = []
        for row, size in zip(wavs, sizes, strict=False):
            result.append(row.narrow(dim=time_dim, start=0, length=size))
        return result

    def load_state_dict(self, state_dict, strict=True):
        if strict:
            missing_keys, unexpected_keys = super().load_state_dict(
                state_dict, strict=False
            )
            # We load this directly from HF, not in checkpoint
            skip_regex = re.compile(
                "(^text_encoder|^visual_ranker|^text_ranker|^span_predictor)"
            )
            missing_keys = [x for x in missing_keys if not re.search(skip_regex, x)]
            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                raise RuntimeError(
                    f"Missing keys: {missing_keys}, unexpected_keys: {unexpected_keys}"
                )


class SamAudioModelTextOnly(SAMAudio):
    """
    A memory-optimized version of SAMAudio that strictly handles Audio and Text.

    This class:
    1. Does NOT initialize vision_encoder, rankers, or span predictors in __init__.
    2. Overrides load_state_dict to ignore those keys from the checkpoint.
    3. Overrides _get_video_features to return empty embeddings without using a model.
    """

    def __init__(self, cfg: SAMAudioConfig):
        # We explicitly call the grandparent (BaseModel) init, bypassing SAMAudio.__init__
        # This prevents the heavy components from being initialized even for a split second.
        super(SAMAudio, self).__init__()

        # --- Initialize only the core components ---
        self.audio_codec = DACVAE(cfg.audio_codec)
        self.text_encoder = T5TextEncoder(cfg.text_encoder)

        # We DO NOT initialize self.vision_encoder.
        # However, we save the dimension for the zero-tensor generation.
        self.vision_encoder = None
        self._vision_encoder_dim = cfg.vision_encoder.dim

        self.transformer = DiT(cfg.transformer)
        self.proj = torch.nn.Linear(cfg.in_channels, cfg.transformer.dim)

        # We keep alignment to ensure tensor shapes match the transformer input expectations
        self.align_masked_video = AlignModalities(
            cfg.vision_encoder.dim, cfg.transformer.dim
        )
        self.embed_anchors = EmbedAnchors(
            cfg.num_anchors, cfg.anchor_embedding_dim, cfg.transformer.dim
        )
        self.memory_proj = torch.nn.Linear(cfg.text_encoder.dim, cfg.transformer.dim)
        self.timestep_emb = SinusoidalEmbedding(cfg.transformer.dim)

        # Explicitly set heavy optional components to None
        self.visual_ranker = None
        self.text_ranker = None
        self.span_predictor = None
        self.span_predictor_transform = None

    def _get_video_features(self, video, audio_features):
        """
        Override: Returns zero-tensors instead of running a vision encoder.
        """
        B, T, _ = audio_features.shape
        # Create zeros matching [Batch, VisionDim, Time]
        return audio_features.new_zeros(B, self._vision_encoder_dim, T)

    def load_state_dict(self, state_dict, strict=True):
        """
        Override: Filters out keys for components we deleted so we don't get errors
        or load them into memory.
        """
        # We pass strict=False to the parent so it doesn't crash on missing keys immediately.
        # We will handle the "real" missing keys check manually below.
        missing_keys, unexpected_keys = super(BaseModel, self).load_state_dict(
            state_dict, strict=False
        )

        # Updated Regex: Includes ^vision_encoder now
        skip_regex = re.compile(
            "(^vision_encoder|^text_encoder|^visual_ranker|^text_ranker|^span_predictor)"
        )

        # Check if we are missing keys that we ACTUALLY care about (not the ones we skipped)
        real_missing_keys = [x for x in missing_keys if not re.search(skip_regex, x)]

        if len(real_missing_keys) > 0:
            raise RuntimeError(
                f"Missing keys: {real_missing_keys}\n(Unexpected keys are ignored)"
            )

        # If strict=True was passed to this function, we theoretically should error on
        # unexpected_keys (the weights for the vision encoder present in the file),
        # but the purpose of this class is to ignore them.


class SamAudioModelTextOnlyOptimized(SamAudioModelTextOnly):
    """
    Compile-friendlier text-only SAM Audio model.

    This preserves the same learned modules and parameter names as
    ``SamAudioModelTextOnly`` so existing checkpoints still load, but it restructures
    the denoising path into tensor-only helpers and replaces the generic fixed-step
    midpoint ODE solve with an equivalent explicit midpoint loop when the requested
    solver configuration matches the default inference path.
    """

    _ODE_INTERVAL = (0.0, 1.0)

    def _prepare_forward_args(self, batch: Batch, candidates: int):
        forward_args = self._get_forward_args(batch, candidates=candidates)
        return (
            forward_args,
            forward_args["audio_features"],
            forward_args["text_features"],
            forward_args["text_mask"],
            forward_args["masked_video_features"],
            forward_args["anchor_ids"],
            forward_args["anchor_alignment"],
            forward_args["audio_pad_mask"],
        )

    def _supports_explicit_midpoint(self, ode_opt: Dict[str, Any]) -> tuple[bool, float, int]:
        method = ode_opt.get("method")
        options = ode_opt.get("options") or {}
        step_size = options.get("step_size")

        if method != "midpoint" or step_size is None:
            return False, 0.0, 0

        start, end = self._ODE_INTERVAL
        total_span = end - start
        raw_steps = total_span / float(step_size)
        rounded_steps = round(raw_steps)
        if rounded_steps <= 0 or not math.isclose(raw_steps, rounded_steps):
            return False, 0.0, 0

        unsupported_keys = set(ode_opt) - {"method", "options"}
        unsupported_option_keys = set(options) - {"step_size"}
        if unsupported_keys or unsupported_option_keys:
            return False, 0.0, 0

        return True, float(step_size), int(rounded_steps)

    def _denoiser_step(
        self,
        noisy_audio: torch.Tensor,
        time: torch.Tensor,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: torch.Tensor,
        masked_video_features: torch.Tensor,
        anchor_ids: torch.Tensor,
        anchor_alignment: torch.Tensor,
        audio_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(
            noisy_audio=noisy_audio,
            audio_features=audio_features,
            text_features=text_features,
            time=time,
            masked_video_features=masked_video_features,
            text_mask=text_mask,
            anchor_ids=anchor_ids,
            anchor_alignment=anchor_alignment,
            audio_pad_mask=audio_pad_mask,
        )

    def _solve_fixed_midpoint(
        self,
        noise: torch.Tensor,
        *,
        step_size: float,
        num_steps: int,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: torch.Tensor,
        masked_video_features: torch.Tensor,
        anchor_ids: torch.Tensor,
        anchor_alignment: torch.Tensor,
        audio_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        state = noise
        batch_size = state.size(0)
        current_time = self._ODE_INTERVAL[0]
        half_step = 0.5 * step_size

        for _ in range(num_steps):
            time_start = state.new_full((batch_size,), current_time)
            k1 = self._denoiser_step(
                state,
                time_start,
                audio_features,
                text_features,
                text_mask,
                masked_video_features,
                anchor_ids,
                anchor_alignment,
                audio_pad_mask,
            )
            midpoint_state = state + half_step * k1
            time_mid = state.new_full((batch_size,), current_time + half_step)
            k2 = self._denoiser_step(
                midpoint_state,
                time_mid,
                audio_features,
                text_features,
                text_mask,
                masked_video_features,
                anchor_ids,
                anchor_alignment,
                audio_pad_mask,
            )
            state = state + step_size * k2
            current_time += step_size

        return state

    def _finalize_outputs(
        self,
        batch: Batch,
        denoised_features: torch.Tensor,
        noise: torch.Tensor,
        reranking_candidates: int,
    ) -> SeparationResult:
        B, T, C2 = denoised_features.shape
        C = C2 // 2

        with torch.autograd.profiler.record_function("sam_audio/audio_codec_decode"):
            wavs = self.audio_codec.decode(
                denoised_features.transpose(1, 2).reshape(2 * B, C, T)
            ).view(B, 2, -1)

        bsz = wavs.size(0) // reranking_candidates
        sizes = self.audio_codec.feature_idx_to_wav_idx(batch.sizes)
        with torch.autograd.profiler.record_function("sam_audio/unbatch_outputs"):
            target_wavs = self.unbatch(
                wavs[:, 0].view(bsz, reranking_candidates, -1), sizes
            )
            residual_wavs = self.unbatch(
                wavs[:, 1].view(bsz, reranking_candidates, -1), sizes
            )

        if (
            reranking_candidates > 1
            and batch.masked_video is not None
            and self.visual_ranker is not None
        ):
            scores = self.visual_ranker(
                extracted_audio=target_wavs,
                videos=batch.masked_video,
                sample_rate=self.audio_codec.sample_rate,
            )
            idxs = scores.argmax(dim=1)
        elif reranking_candidates > 1 and self.text_ranker is not None:
            input_audio = [
                audio[:, :size].expand(reranking_candidates, -1)
                for audio, size in zip(batch.audios, sizes, strict=False)
            ]
            scores = self.text_ranker(
                extracted_audio=target_wavs,
                input_audio=input_audio,
                descriptions=batch.descriptions,
                sample_rate=self.audio_codec.sample_rate,
            )
            idxs = scores.argmax(dim=1)
        else:
            idxs = torch.zeros(bsz, dtype=torch.long, device=noise.device)

        return SeparationResult(
            target=[wav[idx] for wav, idx in zip(target_wavs, idxs, strict=False)],
            residual=[
                wavs[idx] for wavs, idx in zip(residual_wavs, idxs, strict=False)
            ],
            noise=noise,
        )

    @torch.inference_mode()
    def separate(
        self,
        batch: Batch,
        noise: Optional[torch.Tensor] = None,
        ode_opt: Dict[str, Any] = DFLT_ODE_OPT,
        reranking_candidates: int = 1,
        predict_spans: bool = False,
    ) -> SeparationResult:
        with torch.autograd.profiler.record_function("sam_audio/get_forward_args"):
            (
                forward_args,
                audio_features,
                text_features,
                text_mask,
                masked_video_features,
                anchor_ids,
                anchor_alignment,
                audio_pad_mask,
            ) = self._prepare_forward_args(batch, candidates=reranking_candidates)

        if predict_spans and hasattr(self, "span_predictor") and batch.anchors is None:
            batch = self.predict_spans(
                batch=batch,
                audio_features=self._unrepeat_from_reranking(
                    forward_args["audio_features"], reranking_candidates
                ),
                audio_pad_mask=self._unrepeat_from_reranking(
                    forward_args["audio_pad_mask"], reranking_candidates
                ),
            )

        if noise is None:
            with torch.autograd.profiler.record_function("sam_audio/noise_init"):
                noise = torch.randn_like(audio_features)

        use_explicit_midpoint, step_size, num_steps = self._supports_explicit_midpoint(
            ode_opt
        )

        if use_explicit_midpoint:
            with torch.autograd.profiler.record_function("sam_audio/odeint"):
                denoised_features = self._solve_fixed_midpoint(
                    noise,
                    step_size=step_size,
                    num_steps=num_steps,
                    audio_features=audio_features,
                    text_features=text_features,
                    text_mask=text_mask,
                    masked_video_features=masked_video_features,
                    anchor_ids=anchor_ids,
                    anchor_alignment=anchor_alignment,
                    audio_pad_mask=audio_pad_mask,
                )
        else:
            def vector_field(t, noisy_audio):
                return self._denoiser_step(
                    noisy_audio,
                    t.expand(noisy_audio.size(0)),
                    audio_features,
                    text_features,
                    text_mask,
                    masked_video_features,
                    anchor_ids,
                    anchor_alignment,
                    audio_pad_mask,
                )

            with torch.autograd.profiler.record_function("sam_audio/odeint"):
                states = odeint(
                    vector_field,
                    noise,
                    torch.tensor(list(self._ODE_INTERVAL), device=noise.device),
                    **ode_opt,
                )
            denoised_features = states[-1]

        return self._finalize_outputs(
            batch=batch,
            denoised_features=denoised_features,
            noise=noise,
            reranking_candidates=reranking_candidates,
        )


__all__ = ["SAMAudio", "SamAudioModelTextOnly", "SamAudioModelTextOnlyOptimized"]
