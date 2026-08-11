"""
Experimental MiniMax H3 context windows for ComfyUI.

Phase 1 + ref2va scope:
- static contiguous context windows
- joint video/audio slicing
- H3's non-uniform video-token timing is used to map video windows to audio
- absolute target video/audio RoPE positions are preserved across windows
- all ref2va reference blocks are retained in every window
- FL2VA keyframes, FreeNoise, causal anchors, and non-static schedules are intentionally unsupported
"""

import math
from dataclasses import dataclass

import torch

import comfy.conds
import comfy.context_windows as cw
import comfy.ldm.minimax.model as h3_model
import comfy.model_management
import comfy.patcher_extension
import comfy.utils


def _video_indices_to_audio_indices(video_indices, audio_t):
    """Map H3 video latent tokens to every audio latent bin they temporally overlap."""
    video_indices = [int(i) for i in video_indices]
    if not video_indices or audio_t <= 0:
        return []
    if any(i < 0 for i in video_indices):
        raise ValueError("MiniMax H3 context indices must be non-negative.")

    max_idx = max(video_indices)
    starts = h3_model._video_t_grid(max_idx + 1, 0.0).tolist()
    spans = h3_model._video_t_spans(max_idx + 1)

    selected = set()
    for idx in video_indices:
        start = starts[idx]
        end = start + spans[idx]
        # Audio latent j occupies [j, j+1). Exclude boundary-only contacts.
        a_start = max(math.floor(start + 1e-9), 0)
        a_stop = min(math.ceil(end - 1e-9), int(audio_t))
        selected.update(range(a_start, a_stop))
    return sorted(selected)


@dataclass
class _MiniMaxH3WindowingState(cw.WindowingState):
    modality_dims: tuple[int, int] = (2, 3)

    def prepare_window(self, window, model):
        if not self.is_multimodal or len(self.latents) < 2:
            return window

        video_total = self.latents[0].shape[2]
        audio_total = self.latents[1].shape[3]
        audio_indices = _video_indices_to_audio_indices(window.index_list, audio_total)

        # Used only for fuse-weight shape. The actual audio indices are derived from H3 time.
        audio_overlap = max(round(window.context_overlap * audio_total / video_total), 0)

        audio_window = cw.IndexListContextWindow(
            audio_indices,
            dim=3,
            total_frames=audio_total,
            context_overlap=audio_overlap,
        )
        return cw.IndexListContextWindow(
            window.index_list,
            dim=2,
            total_frames=video_total,
            modality_windows={1: audio_window},
            context_overlap=window.context_overlap,
        )

    def slice_for_window(self, window, retain_index_list, device=None):
        # H3 T2VA/ref2va has no context-window guide-frame injection.
        video_window = window
        audio_window = window.get_window_for_modality(1)
        return [
            video_window.get_tensor(self.latents[0], device, dim=2),
            audio_window.get_tensor(self.latents[1], device, dim=3),
        ], [0, 0]

    def strip_guide_frames(self, out_per_modality, guide_frame_counts, window):
        return


class _MiniMaxH3ContextHandler(cw.IndexListContextHandler):
    def _build_window_state(self, x_in, conds, model):
        latent_shapes = self._get_latent_shapes(conds)
        if latent_shapes is None or len(latent_shapes) != 2:
            raise ValueError(
                "MiniMax H3 context windows require the packed video+audio latent "
                "with exactly two latent_shapes entries."
            )

        latents = list(comfy.utils.unpack_latents(x_in, latent_shapes))
        if latents[0].ndim != 5 or latents[1].ndim != 4:
            raise ValueError(
                "Unexpected MiniMax H3 latent shapes: "
                f"video={tuple(latents[0].shape)}, audio={tuple(latents[1].shape)}."
            )

        return _MiniMaxH3WindowingState(
            latents=latents,
            guide_latents=[None, None],
            guide_entries=[None, None],
            keyframe_idxs=[None, None],
            latent_shapes=latent_shapes,
            dim=2,
            is_multimodal=True,
            temporal_downscale_ratio=getattr(model.latent_format, "temporal_downscale_ratio", 1),
        )

    def execute(self, calc_cond_batch, model, conds, x_in, timestep, model_options):
        self._model = model
        self.set_step(timestep, model_options)

        window_state = self._build_window_state(x_in, conds, model)
        context_windows = self.get_context_windows(model, window_state.latents[0], model_options)
        enumerated = list(enumerate(context_windows))
        total_windows = len(enumerated)
        dims = window_state.modality_dims

        accum = [[torch.zeros_like(mod) for _ in conds] for mod in window_state.latents]
        if self.fuse_method.name == cw.ContextFuseMethods.RELATIVE:
            counts = [
                [torch.ones(cw.get_shape_for_dim(mod, dims[i]), device=mod.device) for _ in conds]
                for i, mod in enumerate(window_state.latents)
            ]
        else:
            counts = [
                [torch.zeros(cw.get_shape_for_dim(mod, dims[i]), device=mod.device) for _ in conds]
                for i, mod in enumerate(window_state.latents)
            ]
        biases = [
            [([0.0] * mod.shape[dims[i]]) for _ in conds]
            for i, mod in enumerate(window_state.latents)
        ]

        for callback in comfy.patcher_extension.get_all_callbacks(
            cw.IndexListCallbacks.EXECUTE_START, self.callbacks
        ):
            callback(self, model, x_in, conds, timestep, model_options)

        for enum_window in enumerated:
            results = self.evaluate_context_windows(
                calc_cond_batch,
                model,
                x_in,
                conds,
                timestep,
                [enum_window],
                model_options,
                window_state=window_state,
                total_windows=total_windows,
            )
            for result in results:
                for mod_idx in range(2):
                    mod_out = [
                        result.sub_conds_out[ci][mod_idx] for ci in range(len(conds))
                    ]
                    modality_window = result.window.get_window_for_modality(mod_idx)
                    self.combine_context_window_results(
                        window_state.latents[mod_idx],
                        mod_out,
                        result.sub_conds,
                        modality_window,
                        result.window_idx,
                        total_windows,
                        timestep,
                        accum[mod_idx],
                        counts[mod_idx],
                        biases[mod_idx],
                    )

        try:
            result_out = []
            for ci in range(len(conds)):
                finalized = []
                for mod_idx in range(2):
                    if self.fuse_method.name != cw.ContextFuseMethods.RELATIVE:
                        accum[mod_idx][ci] /= counts[mod_idx][ci]
                    finalized.append(accum[mod_idx][ci])
                packed, _ = comfy.utils.pack_latents(finalized)
                result_out.append(packed)
            return result_out
        finally:
            for callback in comfy.patcher_extension.get_all_callbacks(
                cw.IndexListCallbacks.EXECUTE_CLEANUP, self.callbacks
            ):
                callback(self, model, x_in, conds, timestep, model_options)

    def combine_context_window_results(
        self,
        x_in,
        sub_conds_out,
        sub_conds,
        window,
        window_idx,
        total_windows,
        timestep,
        conds_final,
        counts_final,
        biases_final,
    ):
        dim = window.dim
        if self.fuse_method.name == cw.ContextFuseMethods.RELATIVE:
            for pos, idx in enumerate(window.index_list):
                bias = 1 - abs(
                    idx - (window.index_list[0] + window.index_list[-1]) / 2
                ) / ((window.index_list[-1] - window.index_list[0] + 1e-2) / 2)
                bias = max(1e-2, bias)
                for i in range(len(sub_conds_out)):
                    bias_total = biases_final[i][idx]
                    prev_weight = bias_total / (bias_total + bias)
                    new_weight = bias / (bias_total + bias)
                    idx_window = tuple([slice(None)] * dim + [idx])
                    pos_window = tuple([slice(None)] * dim + [pos])
                    conds_final[i][idx_window] = (
                        conds_final[i][idx_window] * prev_weight
                        + sub_conds_out[i][pos_window] * new_weight
                    )
                    biases_final[i][idx] = bias_total + bias
        else:
            weights = cw.get_context_weights(
                window.context_length,
                x_in.shape[dim],
                window.index_list,
                self,
                sigma=timestep,
                context_overlap=window.context_overlap,
            )
            weights_tensor = cw.match_weights_to_dim(
                weights, x_in, dim, device=x_in.device
            )
            for i in range(len(sub_conds_out)):
                window.add_window(conds_final[i], sub_conds_out[i] * weights_tensor)
                window.add_window(counts_final[i], weights_tensor)

        for callback in comfy.patcher_extension.get_all_callbacks(
            cw.IndexListCallbacks.COMBINE_CONTEXT_WINDOW_RESULTS, self.callbacks
        ):
            callback(
                self,
                x_in,
                sub_conds_out,
                sub_conds,
                window,
                window_idx,
                total_windows,
                timestep,
                conds_final,
                counts_final,
                biases_final,
            )


def _set_h3_absolute_target_positions(layout, video_indices, audio_indices):
    """Replace only the target AV temporal coordinates; refs keep their normal layout."""
    audio_seg = next((seg for seg in layout.segments if seg[2] == "audio"), None)
    video_seg = next((seg for seg in layout.segments if seg[2] == "video"), None)
    if audio_seg is None or video_seg is None:
        raise RuntimeError("MiniMax H3 PackedLayout did not contain target audio/video segments.")

    aa, ab, _ = audio_seg
    va, vb, _ = video_seg
    audio_t = len(audio_indices)
    video_t = len(video_indices)

    if ab - aa != audio_t * 2:
        raise RuntimeError("MiniMax H3 audio layout/window length mismatch.")

    if audio_t:
        target_cursor = float(layout.position_ids[aa, 0])
    elif video_t:
        target_cursor = float(layout.position_ids[va, 0])
    else:
        return

    audio_idx = torch.tensor(audio_indices, dtype=torch.float64)
    layout.position_ids[aa:aa + audio_t, 0] = target_cursor + audio_idx
    layout.position_ids[aa + audio_t:ab, 0] = target_cursor + audio_idx

    if video_t:
        frame_rows = (vb - va) // video_t
        global_grid = h3_model._video_t_grid(
            max(video_indices) + 1, target_cursor
        )[video_indices]
        video_pos = layout.position_ids[va:vb].view(video_t, frame_rows, 3)
        video_pos[:, :, 0] = global_grid[:, None]


def _h3_context_layout_wrapper(
    executor,
    x,
    timestep,
    context,
    transformer_options={},
    minimax_payload=None,
    **kwargs,
):
    window = transformer_options.get("context_window")
    if window is None or not getattr(window, "modality_windows", None):
        return executor(
            x,
            timestep,
            context,
            transformer_options,
            minimax_payload=minimax_payload,
            **kwargs,
        )

    payload = dict(minimax_payload or {})
    if payload.get("keyframes"):
        raise NotImplementedError(
            "MiniMax H3 context windows currently support T2VA and ref2va only. "
            "FL2VA first/last-frame anchors are not implemented in this phase."
        )

    video_x, audio_x = x[0], x[1]
    latent_t = video_x.shape[2]
    lat_h = ((video_x.shape[3] + 1) // 2) * 2
    lat_w = ((video_x.shape[4] + 1) // 2) * 2
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]

    audio_window = window.get_window_for_modality(1)
    video_indices = list(window.index_list)
    audio_indices = list(audio_window.index_list)

    if len(video_indices) != latent_t or len(audio_indices) != audio_t:
        raise RuntimeError(
            "MiniMax H3 context-window metadata does not match sliced latent shapes: "
            f"video {len(video_indices)} != {latent_t}, audio {len(audio_indices)} != {audio_t}."
        )

    # refs=... deliberately includes every ref2va block in every window.
    layout = h3_model.PackedLayout(
        text_len,
        latent_t,
        lat_h,
        lat_w,
        audio_t,
        refs=payload.get("refs"),
    )
    _set_h3_absolute_target_positions(layout, video_indices, audio_indices)
    payload["layout"] = layout

    return executor(
        x,
        timestep,
        context,
        transformer_options,
        minimax_payload=payload,
        **kwargs,
    )


def _decoded_frames_to_video_latents(frame_count):
    frame_count = max(5, int(frame_count))
    remainder = (frame_count - 5) % 17
    if remainder:
        frame_count += 17 - remainder
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


class MiniMaxH3ContextWindows:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "context_length": (
                    "INT",
                    {
                        "default": 73,
                        "min": 5,
                        "max": 3600,
                        "step": 17,
                        "tooltip": "Context length in decoded video frames; H3 uses a 17*n+5 frame grid.",
                    },
                ),
                "context_overlap": (
                    "INT",
                    {
                        "default": 17,
                        "min": 0,
                        "max": 3600,
                        "step": 1,
                        "tooltip": "Approximate overlap in decoded frames.",
                    },
                ),
                "fuse_method": (
                    cw.ContextFuseMethods.LIST_STATIC,
                    {"default": cw.ContextFuseMethods.PYRAMID},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/patch/minimax"
    EXPERIMENTAL = True

    def patch(self, model, context_length, context_overlap, fuse_method):
        model = model.clone()

        latent_context_length = _decoded_frames_to_video_latents(context_length)
        if context_overlap <= 0:
            latent_context_overlap = 0
        else:
            latent_context_overlap = max(round(context_overlap * 5 / 17), 1)
            latent_context_overlap = min(
                latent_context_overlap, latent_context_length - 1
            )

        model.model_options["context_handler"] = _MiniMaxH3ContextHandler(
            context_schedule=cw.get_matching_context_schedule(
                cw.ContextSchedules.STATIC_STANDARD
            ),
            fuse_method=cw.get_matching_fuse_method(fuse_method),
            context_length=latent_context_length,
            context_overlap=latent_context_overlap,
            context_stride=1,
            closed_loop=False,
            dim=2,
            freenoise=False,
            causal_window_fix=False,
        )
        cw.create_prepare_sampling_wrapper(model)
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "MiniMaxH3_context_layout",
            _h3_context_layout_wrapper,
        )
        return (model,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ContextWindows": MiniMaxH3ContextWindows,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ContextWindows": "MiniMax H3 Context Windows (Experimental)",
}
