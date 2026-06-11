#!/usr/bin/env python3
"""
Verify the NVIDIA TAO Grounding-DINO accuracy through the DeepStream packing.
Builds the 6 TAO inputs (image + text tensors) for a frame, packs them exactly
like the C++ preprocess lib, and either:
  * runs the packed ONNX in onnxruntime (FP32) -> the model's reference detections, or
  * dumps packed.bin for trtexec (to test the FP32 / FP16 engines on the same input).

TAO input format (note: opposite of the upstream model):
  attention_mask [256]      = padding mask (1=real token)
  text_token_mask[256,256]  = block-diagonal self-attention mask
"""
import os, sys, json, argparse
import numpy as np, cv2

H, W, L = 544, 960, 256
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SPECIALS = {101, 102, 1012, 1029}     # [CLS] [SEP] . ?


def preprocess(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    chw = np.transpose((rgb / 255.0 - MEAN) / STD, (2, 0, 1))[None]
    return np.ascontiguousarray(chw, np.float32)


def build_text(caption):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    ids = tok([caption], return_tensors="np")["input_ids"][0].tolist()[:L]   # CLS..SEP
    n = len(ids)
    input_ids = np.zeros(L, np.int64); input_ids[:n] = ids
    attention_mask = np.zeros(L, np.int64); attention_mask[:n] = 1            # padding
    token_type = np.zeros(L, np.int64)
    position_ids = np.zeros(L, np.int64)
    text_mask = np.eye(L, dtype=bool)                                        # block-diagonal
    prev, spans = 0, []
    for col in range(L):
        if input_ids[col] not in SPECIALS:
            continue
        if col == 0 or col == L - 1:
            text_mask[col, col] = True; position_ids[col] = 0
        else:
            text_mask[prev + 1:col + 1, prev + 1:col + 1] = True
            position_ids[prev + 1:col + 1] = np.arange(0, col - prev)
            if col - (prev + 1) > 0:
                spans.append((prev + 1, col))
        prev = col
    phrases = [p.strip() for p in caption.replace(",", ".").split(".") if p.strip()]
    return input_ids, attention_mask, position_ids, token_type, text_mask, spans, phrases


def pack(layout, img, input_ids, attention_mask, position_ids, token_type, text_mask):
    by = {t["name"]: t for t in layout["tensors"]}
    packed = np.zeros((1, layout["total"], 1, 1), np.float32); flat = packed.reshape(-1)
    src = {"inputs": img, "input_ids": input_ids, "attention_mask": attention_mask,
           "position_ids": position_ids, "token_type_ids": token_type,
           "text_token_mask": text_mask}
    for name, t in by.items():
        flat[t["offset"]:t["offset"] + t["count"]] = src[name].astype(np.float32).reshape(-1)
    return packed


def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1); I = iw*ih
    U = a[2]*a[3] + b[2]*b[3] - I; return I/U if U > 0 else 0


def decode(pl, pb, spans, phrases, thr, fw, fh):
    dets = []
    for q in range(pl.shape[0]):
        bs, bp = 0.0, -1
        for p, (s0, s1) in enumerate(spans):
            s = sigmoid(pl[q, s0:s1]).max()
            if s > bs: bs, bp = s, p
        if bp < 0 or bs < thr: continue
        cx, cy, w, h = pb[q]
        if w*h > 0.92: continue
        dets.append([(cx-w/2)*fw, (cy-h/2)*fh, w*fw, h*fh, bs, phrases[bp] if bp < len(phrases) else "?"])
    dets.sort(key=lambda d: -d[4])
    keep = []
    for d in dets:
        if all(iou(d, k) <= 0.5 for k in keep): keep.append(d)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="onnx/gdino_single_input.onnx")
    ap.add_argument("--layout", default="assets/pack_layout.json")
    ap.add_argument("--video", default="data/sample_720p.mp4")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--caption", default="car . person .")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--dump", default="", help="save packed.bin here and skip onnxruntime")
    ap.add_argument("--load-json", default="", help="decode a trtexec --exportOutput json instead")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video); cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, bgr = cap.read(); cap.release(); fh, fw = bgr.shape[:2]
    img = preprocess(bgr)
    input_ids, am, pos, tt, tm, spans, phrases = build_text(a.caption)
    layout = json.load(open(a.layout))
    packed = pack(layout, img, input_ids, am, pos, tt, tm)
    print(f"frame {a.frame} caption='{a.caption}' phrases={phrases} spans={spans}")

    if a.dump:
        packed.tofile(a.dump); np.save(a.dump + ".spans.npy", np.array(spans))
        print(f"wrote {a.dump}"); return

    if a.load_json:
        raw = open(a.load_json).read().replace("-inf", "-1e30").replace("inf", "1e30").replace("nan", "0")
        o = {t["name"]: np.array(t["values"], np.float32) for t in json.loads(raw)}
        pl = o["pred_logits"].reshape(900, 256); pb = o["pred_boxes"].reshape(900, 4)
    else:
        import onnxruntime as ort
        sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
        pl, pb = sess.run(["pred_logits", "pred_boxes"], {"packed": packed})
        pl, pb = pl[0], pb[0]

    keep = decode(pl, pb, spans, phrases, a.thr, fw, fh)
    print(f"{len(keep)} detections @ thr={a.thr}:")
    for x, y, w, h, s, ph in keep:
        print(f"  {ph:8s} {s:.3f}  box=({x:.0f},{y:.0f},{w:.0f},{h:.0f})")


if __name__ == "__main__":
    main()
