/*
 * gdino_app.cpp — Grounding-DINO DeepStream 9 application.
 *
 * This is our own app, written from scratch. It no longer patches NVIDIA's
 * deepstream-preprocess-test sample — the full pipeline is here.
 *
 * Pipeline:
 *   uridecodebin → nvstreammux → nvdspreprocess → nvinfer →
 *   nvmultistreamtiler → nvvideoconvert → nvdsosd →
 *   { nvvideoconvert → nvv4l2h264enc → mp4mux → filesink   (GDINO_OUT set)
 *   | <custom element>                                      (GDINO_SINK set)
 *   | nveglglessink / nv3dsink                             (live display)  }
 *
 * Env vars:
 *   GDINO_OUT   path for the output MP4 (e.g. /work/out/gdino_out.mp4)
 *   GDINO_SINK  name of a GStreamer sink element (e.g. nveglglessink)
 *   GDINO_THR   detection threshold (default 0.25)
 *   GDINO_LOG   set to anything to log every detection
 *
 * Usage: gdino-app <preprocess-config> <infer-config> <uri> [<uri> ...]
 */

#include <gst/gst.h>
#include <glib.h>
#include <glib-unix.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstring>

#include "gstnvdsmeta.h"
#include "nvdspreprocess_meta.h"
#include "gdino_prompt_store.h"

#define MAX_LABEL_SIZE   128
#define FPS_INTERVAL     300

#define MUXER_W          1920
#define MUXER_H          1080
#define MUXER_TIMEOUT_US 40000
#define TILER_W          1280
#define TILER_H          720

/* ── label-stamp + FPS probe ─────────────────────────────────────────────── */

static GstPadProbeReturn
pgie_src_probe(GstPad *, GstPadProbeInfo *info, gpointer)
{
    auto *batch_meta = gst_buffer_get_nvds_batch_meta((GstBuffer *)info->data);

    static struct timeval fps_t0 = {};
    static guint          fps_n  = 0;
    static gboolean       fps_init = FALSE;
    if (!fps_init) { gettimeofday(&fps_t0, NULL); fps_init = TRUE; }

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

            if (g_getenv("GDINO_LOG"))
                g_print("[gdino] src=%d frm=%d  %s %.2f\n",
                        fm->source_id, fm->frame_num, lbl, om->confidence);
        }

        g_print("src=%d frame=%d objs=%u\n", fm->source_id, fm->frame_num, nr);

        if (++fps_n % FPS_INTERVAL == 0) {
            struct timeval t1; gettimeofday(&t1, NULL);
            double s = (t1.tv_sec  - fps_t0.tv_sec) +
                       (t1.tv_usec - fps_t0.tv_usec) * 1e-6;
            g_print("[gdino] FPS: %.1f\n", (double)FPS_INTERVAL / s);
            fps_t0 = t1;
        }
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
    if (argc < 4) {
        g_printerr("Usage: %s <preprocess-cfg> <infer-cfg> <uri> [<uri>...]\n",
                   argv[0]);
        return 1;
    }

    gchar *pre_cfg = realpath(argv[1], nullptr);
    gchar *inf_cfg = realpath(argv[2], nullptr);
    int    nsrc    = argc - 3;

    /* detect integrated GPU to choose the right display sink */
    int dev = -1; cudaGetDevice(&dev);
    struct cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);

    gst_init(&argc, &argv);
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

    /* ── sink branch (selected by env) ── */
    const char *gdino_out  = g_getenv("GDINO_OUT");
    const char *gdino_sink = g_getenv("GDINO_SINK");
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

    guint rows = (guint)sqrt((double)nsrc);
    g_object_set(tiler,
                 "rows",    rows,
                 "columns", (guint)ceil((double)nsrc / rows),
                 "width",   TILER_W,
                 "height",  TILER_H,
                 nullptr);
    g_object_set(nvosd, "process-mode", 1, "display-text", 1, nullptr);
    g_object_set(sink,  "qos", FALSE, "sync", FALSE, nullptr);

    /* ── wire up sources → streammux ── */
    gst_bin_add(GST_BIN(pipeline), streammux);
    for (int i = 0; i < nsrc; i++) {
        auto *src = make_source_bin(i, argv[i + 3]);
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

    /* ── add label-stamp + FPS probe on pgie src pad ── */
    auto *pgie_src = gst_element_get_static_pad(pgie, "src");
    gst_pad_add_probe(pgie_src, GST_PAD_PROBE_TYPE_BUFFER,
                      pgie_src_probe, nullptr, nullptr);
    gst_object_unref(pgie_src);

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
