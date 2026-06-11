#!/usr/bin/env python3
"""
Export the GroundingDINO SwinT-OGC model to a 6-input ONNX
that DeepStream can drive.

Unlike NVIDIA's TAO export, the upstream model tokenizes the caption and builds
the BERT self-attention mask + position ids *inside* forward() with data-dependent
ops (torch.nonzero, python loops). Those don't trace into a fixed ONNX. So we:

  1. precompute the text tensors in Python with the upstream recipe
     (generate_masks_with_special_tokens_and_transfer_map), padded to a fixed
     max_text_len=256, and
  2. wrap the model with a forward() that takes those tensors as explicit inputs,
     reproducing GroundingDINO.forward from the BERT call onward.

Tracing runs on CPU, which makes MultiScaleDeformableAttention fall back to its
pure-PyTorch (grid_sample) path — no custom CUDA `_C` op, fully ONNX-able (opset 17).

Inputs of the exported graph (batch dim dynamic = "N"):
    image          [N,3,544,960] float32   (ImageNet-normalized, planar RGB)
    input_ids      [N,256]       int64
    token_type_ids [N,256]       int64
    attention_mask [N,256,256]   bool       BERT text self-attention (block diagonal)
    position_ids   [N,256]       int64
    text_token_mask[N,256]       bool       padding mask (True = real token)
Outputs:
    pred_logits    [N,900,256]   float32    (raw; sigmoid applied at decode)
    pred_boxes     [N,900,4]     float32    (normalized cxcywh)
"""
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "GroundingDINO"))

from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.misc import NestedTensor, inverse_sigmoid
from groundingdino.models.GroundingDINO.bertwarper import (
    generate_masks_with_special_tokens_and_transfer_map,
)

MAX_TOKENS = 256


def load_model(config_path, ckpt_path):
    args = SLConfig.fromfile(config_path)
    args.device = "cpu"
    # gradient checkpointing only affects training memory; off => cleaner trace
    args.use_checkpoint = False
    args.use_transformer_ckpt = False
    model = build_model(args)
    sd = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(clean_state_dict(sd["model"]), strict=False)
    print(f"loaded checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")
    if missing:
        print("  missing (first 10):", missing[:10])
    model.eval()
    return model


def build_text_inputs(model, caption, L=MAX_TOKENS):
    """Tokenize + build the fixed-length text tensors exactly like upstream."""
    tok = model.tokenizer
    enc = tok([caption], padding="longest", return_tensors="pt")
    ids = enc["input_ids"][0].tolist()[:L]            # CLS ... SEP
    n = len(ids)

    input_ids = torch.zeros(1, L, dtype=torch.long)
    input_ids[0, :n] = torch.tensor(ids, dtype=torch.long)
    token_type_ids = torch.zeros(1, L, dtype=torch.long)
    text_token_mask = torch.zeros(1, L, dtype=torch.bool)
    text_token_mask[0, :n] = True

    attn, position_ids, _ = generate_masks_with_special_tokens_and_transfer_map(
        {"input_ids": input_ids}, model.specical_tokens, tok
    )
    return (input_ids, token_type_ids, attn.bool(),
            position_ids.long(), text_token_mask, n)


class GDINOExport(nn.Module):
    """GroundingDINO.forward, but text tensors come in as explicit inputs."""

    def __init__(self, model, H, W):
        super().__init__()
        self.m = model
        self.H, self.W = H, W

    def forward(self, image, input_ids, token_type_ids,
                attention_mask, position_ids, text_token_mask):
        m = self.m
        # ---- text branch (attention_mask = 3D self-attn block-diagonal) ----
        bert_out = m.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        encoded_text = m.feat_map(bert_out["last_hidden_state"])
        text_dict = {
            "encoded_text": encoded_text,
            "text_token_mask": text_token_mask,
            "position_ids": position_ids,
            "text_self_attention_masks": attention_mask,
        }

        # ---- image branch (no padding: full-frame mask = all False) ----
        B = image.shape[0]
        mask = torch.zeros((B, self.H, self.W), dtype=torch.bool, device=image.device)
        samples = NestedTensor(image, mask)
        features, poss = m.backbone(samples)

        srcs, masks = [], []
        for l, feat in enumerate(features):
            src, mk = feat.decompose()
            srcs.append(m.input_proj[l](src))
            masks.append(mk)
        if m.num_feature_levels > len(srcs):
            _len = len(srcs)
            for l in range(_len, m.num_feature_levels):
                src = m.input_proj[l](features[-1].tensors if l == _len else srcs[-1])
                mk = F.interpolate(mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = m.backbone[1](NestedTensor(src, mk)).to(src.dtype)
                srcs.append(src); masks.append(mk); poss.append(pos_l)

        hs, reference, hs_enc, ref_enc, init_box_proposal = m.transformer(
            srcs, masks, None, poss, None, None, text_dict
        )

        outputs_coord_list = []
        for layer_ref_sig, layer_bbox_embed, layer_hs in zip(reference[:-1], m.bbox_embed, hs):
            delta = layer_bbox_embed(layer_hs)
            coord = (delta + inverse_sigmoid(layer_ref_sig)).sigmoid()
            outputs_coord_list.append(coord)
        pred_boxes = torch.stack(outputs_coord_list)[-1]

        outputs_class = torch.stack(
            [cls_embed(layer_hs, text_dict)
             for cls_embed, layer_hs in zip(m.class_embed, hs)]
        )
        pred_logits = outputs_class[-1]
        return pred_logits, pred_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        HERE, "..", "GroundingDINO", "groundingdino", "config",
        "GroundingDINO_SwinT_OGC.py"))
    ap.add_argument("--ckpt", default=os.path.join(
        HERE, "..", "weights", "groundingdino_swint_ogc.pth"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "onnx", "gdino_orig.onnx"))
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--caption", default="car . person .")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--verify", action="store_true",
                    help="run onnxruntime and compare against torch on the same inputs")
    args = ap.parse_args()

    torch.manual_seed(0)
    model = load_model(args.config, args.ckpt)
    wrapper = GDINOExport(model, args.height, args.width).eval()

    input_ids, token_type_ids, attn, position_ids, text_token_mask, n = \
        build_text_inputs(model, args.caption)
    print(f"caption='{args.caption}' -> {n} tokens; "
          f"attn {tuple(attn.shape)} pos {tuple(position_ids.shape)}")

    image = torch.randn(1, 3, args.height, args.width)
    inputs = (image, input_ids, token_type_ids, attn, position_ids, text_token_mask)

    with torch.no_grad():
        pl, pb = wrapper(*inputs)
    print(f"torch outputs: pred_logits {tuple(pl.shape)} pred_boxes {tuple(pb.shape)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    input_names = ["image", "input_ids", "token_type_ids",
                   "attention_mask", "position_ids", "text_token_mask"]
    output_names = ["pred_logits", "pred_boxes"]
    dyn = {k: {0: "N"} for k in input_names + output_names}

    torch.onnx.export(
        wrapper, inputs, args.out,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dyn, opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"wrote {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")

    if args.verify:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        feeds = {
            "image": image.numpy(),
            "input_ids": input_ids.numpy(),
            "token_type_ids": token_type_ids.numpy(),
            "attention_mask": attn.numpy(),
            "position_ids": position_ids.numpy(),
            "text_token_mask": text_token_mask.numpy(),
        }
        o_pl, o_pb = sess.run(output_names, feeds)
        # masked logits are -inf; compare only finite entries
        finite = np.isfinite(pl.numpy()) & np.isfinite(o_pl)
        dl = np.abs(pl.numpy()[finite] - o_pl[finite]).max()
        db = np.abs(pb.numpy() - o_pb).max()
        print(f"[verify] max|Δpred_logits|={dl:.3e}  max|Δpred_boxes|={db:.3e}")
        # CPU onnxruntime vs torch differ only by fp rounding / constant folding.
        assert dl < 5e-2 and db < 1e-3, "ONNX/torch parity failed"
        print("[verify] ONNX matches torch ✓")


if __name__ == "__main__":
    main()
