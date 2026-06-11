/*
 * gdino_app.cpp — Grounding-DINO DeepStream 9 application.
 *
 * A single self-contained GStreamer application that builds the full pipeline.
 *
 * Pipeline:
 *   uridecodebin → nvstreammux → nvdspreprocess → nvinfer →
 *   nvmultistreamtiler → nvvideoconvert → nvdsosd →
 *   { nvvideoconvert → nvv4l2h264enc → mp4mux → filesink   (GDINO_OUT set)
 *   | <custom element>                                      (GDINO_SINK set)
 *   | nveglglessink / nv3dsink                             (live display)  }
 *
 * Flags (output/runtime knobs):
 *   --out FILE      write an annotated H.264 MP4 to FILE
 *   --sink ELEM     use a custom GStreamer sink element (e.g. fakesink, nveglglessink)
 *                   (with neither --out nor --sink, a display sink is used)
 *   --out-w N       output (tiler) width  (default 1280)
 *   --out-h N       output (tiler) height (default 720)
 *   --bitrate BPS   H.264 encoder bitrate in bits/s for --out
 *   --log           log every detection to stdout
 * Detection-tuning knobs live in the bbox parser (libnvds_gdino_parser), which is an
 * nvinfer plugin with no argv, so it reads env: GDINO_THR (0.30), GDINO_NMS_IOU (0.50),
 * GDINO_MAX_AREA (0.92). run.sh exposes these as flags and sets the env for you.
 *
 * Usage: gdino-app [flags] <preprocess-config> <infer-config> <uri> [<uri> ...]
 */

#include <gst/gst.h>
#include <glib.h>
#include <glib-unix.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

#include "gstnvdsmeta.h"
#include "nvdspreprocess_meta.h"
#include "gdino_prompt_store.h"

#define MAX_LABEL_SIZE   128
#define PERF_INTERVAL_SEC 5          /* DeepStream-style **PERF: FPS print interval */
#define MAX_SRC          16

#define MUXER_W          1920
#define MUXER_H          1080
#define MUXER_TIMEOUT_US 40000
#define TILER_W          1280
#define TILER_H          720

/* set by the --log flag; read in the label-stamp probe */
static gboolean g_log_dets = FALSE;

/* per-source frame counters for the **PERF: FPS printer (atomic; read+reset each tick) */
static gint   g_fps_count[MAX_SRC] = {0};
static guint  g_nsrc        = 1;
static gint64 g_perf_last_us = 0;

/* DeepStream-style periodic perf print: **PERF: FPS<src> <fps> ... over the last interval */
static gboolean perf_print_cb(gpointer)
{
    gint64 now = g_get_monotonic_time();
    double dt  = (now - g_perf_last_us) / 1e6;
    g_perf_last_us = now;
    GString *s = g_string_new("**PERF: ");
    double total = 0.0;
    for (guint i = 0; i < g_nsrc; i++) {
        gint c = g_atomic_int_and(&g_fps_count[i], 0);   /* read + reset */
        double f = dt > 0.0 ? c / dt : 0.0;
        total += f;
        g_string_append_printf(s, "FPS%u %.2f  ", i, f);
    }
    if (g_nsrc > 1) g_string_append_printf(s, "(total %.2f)", total);
    g_print("%s\n", s->str);
    g_string_free(s, TRUE);
    return G_SOURCE_CONTINUE;
}

/* ── label-stamp + FPS probe ─────────────────────────────────────────────── */

static GstPadProbeReturn
pgie_src_probe(GstPad *, GstPadProbeInfo *info, gpointer)
{
    auto *batch_meta = gst_buffer_get_nvds_batch_meta((GstBuffer *)info->data);

    for (auto *lf = batch_meta->frame_meta_list; lf; lf = lf->next) {
        auto *fm   = (NvDsFrameMeta *)lf->data;
        auto  snap = gdino::PromptStore::instance().current();
        guint nr   = 0;

        for (auto *lo = fm->obj_meta_list; lo; lo = lo->next) {
            auto *om = (NvDsObjectMeta *)lo->data;
            ++nr;

            const char *lbl = "object";
            if (snap && om->class_id >= 0 &&
                om->class_id < (int)snap->text.phrases.size())
                lbl = snap->text.phrases[om->class_id].c_str();

            g_strlcpy(om->obj_label, lbl, MAX_LABEL_SIZE);
            if (om->text_params.display_text)
                g_free(om->text_params.display_text);
            om->text_params.display_text =
                g_strdup_printf("%s %.2f", lbl, om->confidence);
            om->text_params.x_offset = (int)om->rect_params.left;
            om->text_params.y_offset = (int)(om->rect_params.top > 12
                                             ? om->rect_params.top - 12 : 0);
            om->text_params.font_params.font_name  = (char *)"Serif";
            om->text_params.font_params.font_size  = 11;
            om->text_params.font_params.font_color = {1, 1, 1, 1};
            om->text_params.set_bg_clr             = 1;
            om->text_params.text_bg_clr            = {0, 0, 0, 0.6};
            om->rect_params.border_width           = 3;
            om->rect_params.border_color           = {0, 1, 0, 1};

            if (g_log_dets)
                g_print("[gdino] src=%d frm=%d  %s %.2f\n",
                        fm->source_id, fm->frame_num, lbl, om->confidence);
        }

        g_print("src=%d frame=%d objs=%u\n", fm->source_id, fm->frame_num, nr);
        guint si = fm->source_id < MAX_SRC ? fm->source_id : 0;
        g_atomic_int_inc(&g_fps_count[si]);
    }
    return GST_PAD_PROBE_OK;
}

/* ── bus callback ────────────────────────────────────────────────────────── */

static gboolean
bus_call(GstBus *, GstMessage *msg, gpointer data)
{
    auto *loop = (GMainLoop *)data;
    switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
        g_print("End of stream\n");
        g_main_loop_quit(loop);
        break;
    case GST_MESSAGE_ERROR: {
        gchar *dbg = nullptr; GError *err = nullptr;
        gst_message_parse_error(msg, &err, &dbg);
        g_printerr("ERROR from %s: %s\n", GST_OBJECT_NAME(msg->src), err->message);
        if (dbg) g_printerr("  detail: %s\n", dbg);
        g_free(dbg); g_error_free(err);
        g_main_loop_quit(loop);
        break;
    }
    case GST_MESSAGE_WARNING: {
        gchar *dbg = nullptr; GError *err = nullptr;
        gst_message_parse_warning(msg, &err, &dbg);
        g_printerr("WARNING from %s: %s\n", GST_OBJECT_NAME(msg->src), err->message);
        g_free(dbg); g_error_free(err);
        break;
    }
    default: break;
    }
    return TRUE;
}

static gboolean quit_cb(gpointer loop)
{
    g_print("\n[gdino] signal — stopping...\n");
    g_main_loop_quit((GMainLoop *)loop);
    return G_SOURCE_REMOVE;
}

/* ── source bin (uridecodebin wrapped in a bin with a ghost src pad) ──────── */

static void
cb_newpad(GstElement *, GstPad *src_pad, gpointer data)
{
    auto *caps = gst_pad_get_current_caps(src_pad);
    auto *str  = gst_caps_get_structure(caps, 0);
    auto *feat = gst_caps_get_features(caps, 0);

    if (!strncmp(gst_structure_get_name(str), "video", 5) &&
        gst_caps_features_contains(feat, "memory:NVMM")) {
        auto *ghost = gst_element_get_static_pad((GstElement *)data, "src");
        gst_ghost_pad_set_target(GST_GHOST_PAD(ghost), src_pad);
        gst_object_unref(ghost);
    }
    gst_caps_unref(caps);
}

static void
child_added(GstChildProxy *proxy, GObject *obj, gchar *name, gpointer ud)
{
    if (g_strrstr(name, "decodebin"))
        g_signal_connect(obj, "child-added", G_CALLBACK(child_added), ud);
}

static GstElement *
make_source_bin(guint idx, const char *uri)
{
    gchar bname[20]; g_snprintf(bname, sizeof(bname), "source-bin-%02u", idx);
    auto *bin = gst_bin_new(bname);
    auto *dec = gst_element_factory_make("uridecodebin", "uri-decode-bin");
    if (!bin || !dec) { g_printerr("Cannot create source bin\n"); return nullptr; }

    g_object_set(dec, "uri", uri, nullptr);
    g_signal_connect(dec, "pad-added",   G_CALLBACK(cb_newpad),   bin);
    g_signal_connect(dec, "child-added", G_CALLBACK(child_added), nullptr);
    gst_bin_add(GST_BIN(bin), dec);

    if (!gst_element_add_pad(bin, gst_ghost_pad_new_no_target("src", GST_PAD_SRC))) {
        g_printerr("Cannot add ghost pad to source bin\n");
        return nullptr;
    }
    return bin;
}

/* ── element factory helper ──────────────────────────────────────────────── */

static GstElement *
make_elem(const char *factory, const char *name)
{
    auto *e = gst_element_factory_make(factory, name);
    if (!e) g_printerr("ERROR: cannot create '%s' (as '%s')\n", factory, name);
    return e;
}

/* ── main ────────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
    /* Flags + positionals. Positionals: <preprocess-cfg> <infer-cfg> <uri> [<uri>...] */
    const char *out_file = nullptr, *sink_name = nullptr;
    int out_w = 0, out_h = 0, bitrate = 0;
    std::vector<char *> pos;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (a == "--out")     out_file  = next();
        else if (a == "--sink")    sink_name = next();
        else if (a == "--out-w")   out_w     = atoi(next());
        else if (a == "--out-h")   out_h     = atoi(next());
        else if (a == "--bitrate") bitrate   = atoi(next());
        else if (a == "--log")     g_log_dets = TRUE;
        else                       pos.push_back(argv[i]);
    }
    if (pos.size() < 3) {
        g_printerr("Usage: %s [--out FILE|--sink ELEM] [--out-w N] [--out-h N] "
                   "[--bitrate BPS] [--log] <preprocess-cfg> <infer-cfg> <uri> [<uri>...]\n",
                   argv[0]);
        return 1;
    }

    gchar *pre_cfg = realpath(pos[0], nullptr);
    gchar *inf_cfg = realpath(pos[1], nullptr);
    int    nsrc    = (int)pos.size() - 2;

    /* detect integrated GPU to choose the right display sink */
    int dev = -1; cudaGetDevice(&dev);
    struct cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);

    gst_init(nullptr, nullptr);
    auto *loop = g_main_loop_new(nullptr, FALSE);

    /* ── create pipeline elements ── */
    auto *pipeline   = gst_pipeline_new("gdino-pipeline");
    auto *streammux  = make_elem("nvstreammux",        "stream-muxer");
    auto *preprocess = make_elem("nvdspreprocess",     "preprocess");
    auto *pgie       = make_elem("nvinfer",            "pgie");
    auto *tiler      = make_elem("nvmultistreamtiler", "tiler");
    auto *nvvidconv  = make_elem("nvvideoconvert",     "converter");
    auto *nvosd      = make_elem("nvdsosd",            "osd");
    auto *q1 = make_elem("queue","q1"), *q2 = make_elem("queue","q2"),
         *q3 = make_elem("queue","q3"), *q4 = make_elem("queue","q4"),
         *q5 = make_elem("queue","q5"), *q6 = make_elem("queue","q6");

    /* ── sink branch (selected by flags) ── */
    const char *gdino_out  = out_file;
    const char *gdino_sink = sink_name;
    GstElement *oconv = nullptr, *ocaps = nullptr,
               *h264enc = nullptr, *h264parse = nullptr,
               *muxer = nullptr, *sink = nullptr;

    if (gdino_out) {
        oconv     = make_elem("nvvideoconvert", "out-conv");
        ocaps     = make_elem("capsfilter",     "out-caps");
        h264enc   = make_elem("nvv4l2h264enc",  "h264enc");
        h264parse = make_elem("h264parse",      "h264parse");
        muxer     = make_elem("mp4mux",         "muxer");
        sink      = make_elem("filesink",        "sink");
        auto *caps = gst_caps_from_string("video/x-raw(memory:NVMM),format=NV12");
        g_object_set(ocaps, "caps", caps, nullptr);
        gst_caps_unref(caps);
        if (bitrate > 0) g_object_set(h264enc, "bitrate", (guint)bitrate, nullptr);
        g_object_set(sink, "location", gdino_out, nullptr);
    } else if (gdino_sink && *gdino_sink) {
        sink = make_elem(gdino_sink, "sink");
    } else {
        sink = make_elem(prop.integrated ? "nv3dsink" : "nveglglessink", "sink");
    }

    /* ── null-check all elements ── */
    if (!pipeline || !streammux || !preprocess || !pgie || !tiler ||
        !nvvidconv || !nvosd || !q1 || !q2 || !q3 || !q4 || !q5 || !q6 || !sink ||
        (gdino_out && (!oconv || !ocaps || !h264enc || !h264parse || !muxer))) {
        g_printerr("Failed to create one or more pipeline elements — aborting.\n");
        return 1;
    }

    /* ── configure elements ── */
    g_object_set(streammux,
                 "batch-size",          nsrc,
                 "width",               MUXER_W,
                 "height",              MUXER_H,
                 "batched-push-timeout",MUXER_TIMEOUT_US,
                 nullptr);
    g_object_set(preprocess, "config-file", pre_cfg, nullptr);
    g_object_set(pgie,
                 "config-file-path",  inf_cfg,
                 "input-tensor-meta", TRUE,
                 nullptr);

    guint tiler_w = out_w > 0 ? (guint)out_w : TILER_W;
    guint tiler_h = out_h > 0 ? (guint)out_h : TILER_H;
    guint rows = (guint)sqrt((double)nsrc);
    g_object_set(tiler,
                 "rows",    rows,
                 "columns", (guint)ceil((double)nsrc / rows),
                 "width",   tiler_w,
                 "height",  tiler_h,
                 nullptr);
    g_object_set(nvosd, "process-mode", 1, "display-text", 1, nullptr);
    g_object_set(sink,  "qos", FALSE, "sync", FALSE, nullptr);

    /* ── wire up sources → streammux ── */
    gst_bin_add(GST_BIN(pipeline), streammux);
    for (int i = 0; i < nsrc; i++) {
        auto *src = make_source_bin(i, pos[i + 2]);
        if (!src) return 1;
        gst_bin_add(GST_BIN(pipeline), src);

        gchar pad_name[16]; g_snprintf(pad_name, sizeof(pad_name), "sink_%d", i);
        auto *sinkpad = gst_element_request_pad_simple(streammux, pad_name);
        auto *srcpad  = gst_element_get_static_pad(src, "src");
        if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
            g_printerr("Failed to link source %d to streammux\n", i);
            return 1;
        }
        gst_object_unref(srcpad);
        gst_object_unref(sinkpad);
    }

    /* ── helper: link a pair and abort on failure ── */
    auto link2 = [](GstElement *a, GstElement *b) -> bool {
        if (!gst_element_link(a, b)) {
            g_printerr("ERROR: cannot link '%s' → '%s'\n",
                       GST_ELEMENT_NAME(a), GST_ELEMENT_NAME(b));
            return false;
        }
        return true;
    };

    /* ── add remaining elements + link the chain ── */
    if (gdino_out) {
        gst_bin_add_many(GST_BIN(pipeline),
                         q1, preprocess, q2, pgie, q3, tiler,
                         q4, nvvidconv, q5, nvosd, q6,
                         oconv, ocaps, h264enc, h264parse, muxer, sink, nullptr);

        /* link the main chain up to h264parse */
        if (!link2(streammux, q1)       || !link2(q1, preprocess)  ||
            !link2(preprocess, q2)      || !link2(q2, pgie)         ||
            !link2(pgie, q3)            || !link2(q3, tiler)        ||
            !link2(tiler, q4)           || !link2(q4, nvvidconv)    ||
            !link2(nvvidconv, q5)       || !link2(q5, nvosd)        ||
            !link2(nvosd, q6)           || !link2(q6, oconv)        ||
            !link2(oconv, ocaps)        || !link2(ocaps, h264enc)   ||
            !link2(h264enc, h264parse))
            return 1;

        /* mp4mux has a REQUEST video sink pad — request it explicitly.
           h264parse ensures stream-format=avc,alignment=au which mp4mux requires. */
        GstPad *parse_src = gst_element_get_static_pad(h264parse, "src");
        GstPad *mux_sink  = gst_element_request_pad_simple(muxer, "video_%u");
        if (!parse_src || !mux_sink) {
            g_printerr("ERROR: cannot get pads for h264parse→mp4mux\n"); return 1;
        }
        GstPadLinkReturn lr = gst_pad_link(parse_src, mux_sink);
        if (lr != GST_PAD_LINK_OK) {
            g_printerr("ERROR: h264parse→mp4mux pad link failed (%d)\n", lr); return 1;
        }
        gst_object_unref(parse_src);
        gst_object_unref(mux_sink);

        if (!link2(muxer, sink)) return 1;

    } else {
        gst_bin_add_many(GST_BIN(pipeline),
                         q1, preprocess, q2, pgie, q3, tiler,
                         q4, nvvidconv, q5, nvosd, q6, sink, nullptr);
        if (!link2(streammux, q1)    || !link2(q1, preprocess) ||
            !link2(preprocess, q2)   || !link2(q2, pgie)        ||
            !link2(pgie, q3)         || !link2(q3, tiler)       ||
            !link2(tiler, q4)        || !link2(q4, nvvidconv)   ||
            !link2(nvvidconv, q5)    || !link2(q5, nvosd)       ||
            !link2(nvosd, q6)        || !link2(q6, sink))
            return 1;
    }

    /* ── add label-stamp + frame-count probe on pgie src pad ── */
    auto *pgie_src = gst_element_get_static_pad(pgie, "src");
    gst_pad_add_probe(pgie_src, GST_PAD_PROBE_TYPE_BUFFER,
                      pgie_src_probe, nullptr, nullptr);
    gst_object_unref(pgie_src);

    /* ── DeepStream-style **PERF: FPS printer (every PERF_INTERVAL_SEC) ── */
    g_nsrc = (guint)nsrc < MAX_SRC ? (guint)nsrc : MAX_SRC;
    g_perf_last_us = g_get_monotonic_time();
    g_timeout_add_seconds(PERF_INTERVAL_SEC, perf_print_cb, nullptr);

    /* ── bus watch + signal handlers ── */
    auto *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    g_unix_signal_add(SIGINT,  quit_cb, loop);
    g_unix_signal_add(SIGTERM, quit_cb, loop);

    /* ── run ── */
    g_print("Now playing...\n");
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_main_loop_run(loop);

    g_print("Stopping pipeline...\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    g_main_loop_unref(loop);
    g_free(pre_cfg);
    g_free(inf_cfg);
    return 0;
}
