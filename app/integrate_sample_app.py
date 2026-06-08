#!/usr/bin/env python3
"""
Build the app by editing NVIDIA's stock deepstream-preprocess-test sample, so no
NVIDIA source needs to live in this repo. Reads the sample from the installed
DeepStream tree, applies the edits below, links against libgdino_common.so, builds.

Run inside the DeepStream 9.0 container:
  python3 app/integrate_sample_app.py --cuda-ver 13.1
"""
import argparse
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_SAMPLE = ("/opt/nvidia/deepstream/deepstream/sources/apps/"
                  "sample_apps/deepstream-preprocess-test")

CPP = "deepstream_preprocess_test.cpp"
MK = "Makefile"


# ---- edit 1: include -------------------------------------------------------
INC_ANCHOR = '#include "gstnvdsinfer.h"\n'
INC_NEW = (INC_ANCHOR +
           '#include "gdino_prompt_store.h"\n'
           '#include <glib-unix.h>\n')

# ---- edit 2: helper (inserted after the stock class-name table) ------------
HELPER_ANCHOR = ('gchar pgie_classes_str[4][32] = {"Vehicle", "TwoWheeler", "Person",\n'
                 '                                 "RoadSign"};\n')
HELPER_NEW = HELPER_ANCHOR + r'''
/* Label a detection with its phrase (class_id indexes the live prompt). */
static void gdino_stamp_label(NvDsObjectMeta *om)
{
  auto snap = gdino::PromptStore::instance().current();
  const char *lbl = "object";
  if (snap && om->class_id >= 0 &&
      om->class_id < (int)snap->text.phrases.size())
    lbl = snap->text.phrases[om->class_id].c_str();
  g_strlcpy(om->obj_label, lbl, MAX_LABEL_SIZE);
  if (om->text_params.display_text)
    g_free(om->text_params.display_text);
  om->text_params.display_text = g_strdup_printf("%s %.2f", lbl, om->confidence);
  om->text_params.x_offset = (int)om->rect_params.left;
  om->text_params.y_offset = (int)(om->rect_params.top > 12 ? om->rect_params.top - 12 : 0);
  om->text_params.font_params.font_name = (char *)"Serif";
  om->text_params.font_params.font_size = 11;
  om->text_params.font_params.font_color = {1, 1, 1, 1};
  om->text_params.set_bg_clr = 1;
  om->text_params.text_bg_clr = {0, 0, 0, 0.6};
  om->rect_params.border_width = 3;
  om->rect_params.border_color = {0, 1, 0, 1};
  if (g_getenv("GDINO_LOG"))
    g_print("[gdino] %s %.2f\n", lbl, om->confidence);
}

/* Quit the main loop on Ctrl+C / SIGTERM so the pipeline shuts down cleanly. */
static gboolean gdino_quit(gpointer loop)
{
  g_print("\n[gdino] signal received - stopping pipeline...\n");
  g_main_loop_quit((GMainLoop *)loop);
  return G_SOURCE_REMOVE;
}
'''

# ---- edit 3: call the helper for each object -------------------------------
CALL_ANCHOR = ('      obj_meta = (NvDsObjectMeta *)(l_obj->data);\n'
               '      num_rects++;\n')
CALL_NEW = CALL_ANCHOR + '      gdino_stamp_label(obj_meta);\n'

# ---- edit 4: env-selectable sink -------------------------------------------
SINK_ANCHOR = '''  if (prop.integrated)
  {
    sink = gst_element_factory_make("nv3dsink", "nvvideo-renderer");
  } else {
#ifdef __aarch64__
    sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
#else
    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
#endif
  }
'''
SINK_NEW = '''  /* env-selectable sink: GDINO_OUT (jpegs) | GDINO_SINK | on-screen renderer */
  GstElement *oconv = NULL, *ocaps = NULL, *jenc = NULL;
  const char *gdino_out = g_getenv("GDINO_OUT");
  const char *gdino_sink = g_getenv("GDINO_SINK");
  if (gdino_out) {
    oconv = gst_element_factory_make("nvvideoconvert", "gdino-oconv");
    ocaps = gst_element_factory_make("capsfilter", "gdino-ocaps");
    GstCaps *icaps = gst_caps_from_string("video/x-raw,format=I420");
    g_object_set(ocaps, "caps", icaps, NULL);
    gst_caps_unref(icaps);
    jenc = gst_element_factory_make("jpegenc", "gdino-jenc");
    sink = gst_element_factory_make("multifilesink", "nvvideo-renderer");
    g_object_set(G_OBJECT(sink), "location", gdino_out, NULL);
  } else if (gdino_sink && *gdino_sink) {
    sink = gst_element_factory_make(gdino_sink, "nvvideo-renderer");
  } else if (prop.integrated) {
    sink = gst_element_factory_make("nv3dsink", "nvvideo-renderer");
  } else {
#ifdef __aarch64__
    sink = gst_element_factory_make("nv3dsink", "nvvideo-renderer");
#else
    sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");
#endif
  }
'''

# ---- edit 5: add the encoder tail when saving to JPEGs ---------------------
LINK_ANCHOR = '''  gst_bin_add_many(GST_BIN(pipeline), queue1, preprocess, queue2, pgie, queue3, tiler,
                   queue4, nvvidconv, queue5, nvosd, queue6, sink, NULL);
  /* we link the elements together
  * nvstreammux -> pgie -> nvtiler -> nvvidconv -> nvosd -> video-renderer */
  if (!gst_element_link_many(streammux, queue1, preprocess, queue2, pgie, queue3, tiler,
                             queue4, nvvidconv, queue5, nvosd, queue6, sink, NULL))
  {
    g_printerr("Elements could not be linked. Exiting.\\n");
    return -1;
  }
'''
LINK_NEW = '''  if (gdino_out) {
    gst_bin_add_many(GST_BIN(pipeline), queue1, preprocess, queue2, pgie, queue3, tiler,
                     queue4, nvvidconv, queue5, nvosd, queue6, oconv, ocaps, jenc, sink, NULL);
    if (!gst_element_link_many(streammux, queue1, preprocess, queue2, pgie, queue3, tiler,
                               queue4, nvvidconv, queue5, nvosd, queue6, oconv, ocaps, jenc, sink, NULL)) {
      g_printerr("Elements could not be linked. Exiting.\\n");
      return -1;
    }
  } else {
    gst_bin_add_many(GST_BIN(pipeline), queue1, preprocess, queue2, pgie, queue3, tiler,
                     queue4, nvvidconv, queue5, nvosd, queue6, sink, NULL);
    if (!gst_element_link_many(streammux, queue1, preprocess, queue2, pgie, queue3, tiler,
                               queue4, nvvidconv, queue5, nvosd, queue6, sink, NULL)) {
      g_printerr("Elements could not be linked. Exiting.\\n");
      return -1;
    }
  }
'''

# ---- edit 6: render late frames instead of dropping them -------------------
# The transformer adds latency, so frames arrive past their clock deadline; with
# sync=true the sink drops them and stutters. sync=false renders them instead.
SYNC_ANCHOR = '  g_object_set(G_OBJECT(sink), "qos", 0, NULL);\n'
SYNC_NEW = '  g_object_set(G_OBJECT(sink), "qos", 0, "sync", FALSE, NULL);\n'

# ---- edit 7: install the Ctrl+C / SIGTERM handler on the main loop ---------
SIGNAL_ANCHOR = '  loop = g_main_loop_new(NULL, FALSE);\n'
SIGNAL_NEW = (SIGNAL_ANCHOR +
              '  g_unix_signal_add(SIGINT, gdino_quit, loop);\n'
              '  g_unix_signal_add(SIGTERM, gdino_quit, loop);\n')

EDITS = [
    ("include", INC_ANCHOR, INC_NEW),
    ("helper", HELPER_ANCHOR, HELPER_NEW),
    ("probe-call", CALL_ANCHOR, CALL_NEW),
    ("sink", SINK_ANCHOR, SINK_NEW),
    ("link", LINK_ANCHOR, LINK_NEW),
    ("sync", SYNC_ANCHOR, SYNC_NEW),
    ("signal", SIGNAL_ANCHOR, SIGNAL_NEW),
]


def apply_edits(text):
    for name, anchor, new in EDITS:
        n = text.count(anchor)
        if n != 1:
            sys.exit(f"ERROR: anchor '{name}' matched {n} times (expected 1). "
                     f"The DeepStream sample may differ from the 9.0 version "
                     f"this integrator targets.")
        text = text.replace(anchor, new, 1)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", default=DEFAULT_SAMPLE,
                    help="installed deepstream-preprocess-test directory")
    ap.add_argument("--build-dir", default=os.path.join(REPO, "app", "build_app"),
                    help="where to assemble + build the patched app")
    ap.add_argument("--cuda-ver", default="13.1")
    ap.add_argument("--out", default=os.path.join(REPO, "build", "gdino-preprocess-test"),
                    help="final binary path")
    args = ap.parse_args()

    src_cpp = os.path.join(args.sample_dir, CPP)
    src_mk = os.path.join(args.sample_dir, MK)
    if not os.path.isfile(src_cpp) or not os.path.isfile(src_mk):
        sys.exit(f"ERROR: sample not found at {args.sample_dir} "
                 f"(is DeepStream installed / are you in the container?)")

    # assemble a private copy of the sample (we never store NVIDIA source in the repo)
    if os.path.isdir(args.build_dir):
        shutil.rmtree(args.build_dir)
    os.makedirs(args.build_dir)
    shutil.copy(src_mk, os.path.join(args.build_dir, MK))

    with open(src_cpp) as f:
        text = f.read()
    text = apply_edits(text)
    with open(os.path.join(args.build_dir, CPP), "w") as f:
        f.write(text)
    print(f"patched {CPP}: {len(EDITS)} edits applied")

    # The sample Makefile's include paths are relative to its install dir and
    # break in a copied dir, so append absolute ones plus our headers and lib.
    ds_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(args.sample_dir)))))  # .../deepstream
    ds_inc = os.path.join(ds_root, "sources", "includes")
    pre_inc = os.path.join(ds_root, "sources", "gst-plugins",
                           "gst-nvdspreprocess", "include")
    with open(os.path.join(args.build_dir, MK), "a") as f:
        f.write(f"\n# --- GDINO integration ---\n")
        f.write(f"CFLAGS += -I{REPO}/src -I{ds_inc} -I{pre_inc}\n")
        f.write(f"LIBS += -L{REPO}/build -lgdino_common -Wl,-rpath,{REPO}/build\n")
    print(f"patched Makefile: GDINO + DeepStream absolute includes + libgdino_common")

    subprocess.run(["make", f"CUDA_VER={args.cuda_ver}"],
                   cwd=args.build_dir, check=True)

    built = os.path.join(args.build_dir, "deepstream-preprocess-test")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    shutil.copy(built, args.out)
    print(f"\nOK -> {args.out}")


if __name__ == "__main__":
    main()
