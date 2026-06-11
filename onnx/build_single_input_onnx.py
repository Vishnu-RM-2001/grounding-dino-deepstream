#!/usr/bin/env python3
"""
Pack the 6 Grounding-DINO inputs into one fp32 'packed' input and split them back
inside the graph with a small Slice/Reshape/Cast preamble, so nvinfer can take the
single per-frame tensor that nvdspreprocess provides.

Two models are supported; they differ only in the order of the text inputs and in
which mask is 2-D (block-diagonal). Pick with --variant (default: auto-detect):

  tao       inputs, input_ids, attention_mask[256], position_ids, token_type_ids,
            text_token_mask[256,256]
  gdino_b image, input_ids, token_type_ids, attention_mask[256,256], position_ids,
            text_token_mask[256]

The chosen order MUST match src/gdino_layout.h / writeTextRegion for that variant.
"""
import argparse, json, os
import numpy as np
import onnx
import onnx_graphsurgeon as gs

PACK_ORDERS = {
    "tao":      ["inputs", "input_ids", "attention_mask",
                 "position_ids", "token_type_ids", "text_token_mask"],
    "gdino_b": ["image", "input_ids", "token_type_ids",
                 "attention_mask", "position_ids", "text_token_mask"],
}


def detect_variant(by_name):
    # the gdino_b export names the image "image" and makes attention_mask the 2-D mask;
    # the TAO export names it "inputs" and makes text_token_mask the 2-D mask.
    if "image" in by_name:
        return "gdino_b"
    if "inputs" in by_name:
        return "tao"
    raise SystemExit(f"cannot auto-detect variant; inputs: {list(by_name)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="original onnx")
    ap.add_argument("--out", dest="out", required=True, help="single-input onnx")
    ap.add_argument("--layout", required=True, help="layout json to write")
    ap.add_argument("--variant", choices=["tao", "gdino_b", "auto"], default="auto")
    ap.add_argument("--packed-name", default="packed")
    args = ap.parse_args()

    graph = gs.import_onnx(onnx.load(args.inp))
    by_name = {t.name: t for t in graph.inputs}
    variant = detect_variant(by_name) if args.variant == "auto" else args.variant
    PACK_ORDER = PACK_ORDERS[variant]
    print(f"variant: {variant}  pack_order: {PACK_ORDER}")
    missing = [n for n in PACK_ORDER if n not in by_name]
    if missing:
        raise SystemExit(f"inputs not found in model: {missing}; "
                         f"present: {[t.name for t in graph.inputs]}")

    # Resolve concrete shapes with batch forced to 1.
    layout, offset, sizes = [], 0, []
    for name in PACK_ORDER:
        v = by_name[name]
        shape = [1 if (isinstance(d, str) or d in (-1, 0, None)) else int(d)
                 for d in v.shape]
        n = int(np.prod(shape))
        layout.append({"name": name, "shape": shape, "count": n,
                       "offset": offset, "elem_type": int(v.dtype_onnx)
                       if hasattr(v, "dtype_onnx") else None})
        sizes.append(n)
        offset += n
    total = offset

    # Build preamble: packed[N,total,1,1] -> Split -> per-input Reshape (+Cast).
    # NB1: shape is 4-D NCHW (C=total,H=1,W=1).
    # NB2: batch dim is DYNAMIC ("N"). Gst-nvinfer only enumerates an engine's
    # layers when input tensor[0] has a wildcard dim (its getImplicitLayersInfo()
    # path is a no-op); a fully-fixed input makes nvinfer report "layers num: 0".
    packed = gs.Variable(args.packed_name, dtype=np.float32, shape=["N", total, 1, 1])
    split_sizes = gs.Constant("split_sizes", np.array(sizes, dtype=np.int64))
    split_outs = [gs.Variable(f"chunk_{i}", dtype=np.float32, shape=["N", s, 1, 1])
                  for i, s in enumerate(sizes)]
    nodes = [gs.Node("Split", "split_packed",
                     inputs=[packed, split_sizes], outputs=split_outs,
                     attrs={"axis": 1})]

    new_inputs_handled = []
    for i, name in enumerate(PACK_ORDER):
        v = by_name[name]                       # original input Variable
        info = layout[i]
        tgt_shape = info["shape"]
        tgt_np = v.dtype                         # numpy dtype of original input
        chunk = split_outs[i]

        # dynamic batch: reshape with -1 in the batch dim so it follows N
        dyn_shape = [-1] + list(tgt_shape[1:])
        shape_c = gs.Constant(f"shape_{name}",
                              np.array(dyn_shape, dtype=np.int64))
        if np.dtype(tgt_np) == np.float32:
            # image: Reshape straight into the original input tensor
            v.inputs.clear()
            nodes.append(gs.Node("Reshape", f"reshape_{name}",
                                 inputs=[chunk, shape_c], outputs=[v]))
        else:
            reshaped = gs.Variable(f"{name}_f32", dtype=np.float32,
                                   shape=["N"] + list(tgt_shape[1:]))
            nodes.append(gs.Node("Reshape", f"reshape_{name}",
                                 inputs=[chunk, shape_c], outputs=[reshaped]))
            v.inputs.clear()
            nodes.append(gs.Node("Cast", f"cast_{name}", inputs=[reshaped],
                                 outputs=[v],
                                 attrs={"to": onnx.helper.np_dtype_to_tensor_dtype(
                                     np.dtype(tgt_np))}))
        new_inputs_handled.append(name)

    graph.nodes.extend(nodes)
    graph.inputs = [packed]
    graph.cleanup().toposort()

    onnx.save(gs.export_onnx(graph), args.out)
    with open(args.layout, "w") as f:
        json.dump({"packed_name": args.packed_name, "variant": variant,
                   "total": total, "pack_order": PACK_ORDER, "tensors": layout}, f, indent=2)

    print(f"Wrote {args.out}")
    print(f"Packed input '{args.packed_name}': [1, {total}] float32  "
          f"({total*4/1e6:.2f} MB/frame)")
    for t in layout:
        print(f"  {t['name']:16s} off={t['offset']:>9d} count={t['count']:>8d} "
              f"shape={t['shape']}")
    print(f"Layout -> {args.layout}")


if __name__ == "__main__":
    main()
