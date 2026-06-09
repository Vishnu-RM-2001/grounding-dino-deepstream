# Builds the three plugin libraries:
#   libgdino_common.so          tokenizer + text recipe + PromptStore + decoder
#   libnvds_gdino_preprocess.so nvdspreprocess custom lib
#   libnvds_gdino_parser.so     nvinfer bbox parser
# Usually invoked via scripts/01_build_libs.sh (which provides nvcc).
DS   ?= /opt/nvidia/deepstream/deepstream
CUDA ?= /usr/local/cuda
BUILD = build

CXX       ?= g++
CXXFLAGS   = -std=c++17 -O2 -fPIC -Wall -Wno-unused-parameter
INC        = -Isrc
# EXTRA_INC: point to nvtx3 headers if not on the default CUDA include path
# (DS9 samples image: EXTRA_INC=-I/usr/local/cuda-13.1/targets/x86_64-linux/include)
EXTRA_INC ?=
DS_INC     = -I$(DS)/sources/includes \
             -I$(DS)/sources/gst-plugins/gst-nvdspreprocess/include \
             -I$(CUDA)/include $(EXTRA_INC)
RPATH      = -Wl,-rpath,'$$ORIGIN'

# libgdino_common.so: the model logic shared by BOTH plugin libs + the app probe
COMMON_SRC = src/bert_tokenizer.cpp \
             src/gdino_text.cpp \
             src/gdino_prompt_store.cpp \
             src/gdino_decode.cpp
GST_CFLAGS = $(shell pkg-config --cflags gstreamer-1.0 glib-2.0)

.PHONY: all clean
all: $(BUILD)/libgdino_common.so $(BUILD)/libnvds_gdino_preprocess.so $(BUILD)/libnvds_gdino_parser.so

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/libgdino_common.so: $(COMMON_SRC) | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) -shared $^ -o $@ -lpthread

$(BUILD)/normalize.o: src/normalize.cu | $(BUILD)
	$(CUDA)/bin/nvcc -ccbin $(CXX) -Xcompiler -fPIC -O2 -c $< -o $@

$(BUILD)/libnvds_gdino_preprocess.so: src/gdino_preprocess.cpp $(BUILD)/normalize.o $(BUILD)/libgdino_common.so | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) $(DS_INC) $(GST_CFLAGS) -shared \
	  src/gdino_preprocess.cpp $(BUILD)/normalize.o -o $@ \
	  -L$(BUILD) -lgdino_common -L$(CUDA)/lib64 -lcudart $(RPATH)

$(BUILD)/libnvds_gdino_parser.so: src/nvdsparse_gdino.cpp $(BUILD)/libgdino_common.so | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) $(DS_INC) -shared \
	  src/nvdsparse_gdino.cpp -o $@ -L$(BUILD) -lgdino_common $(RPATH)

$(BUILD)/gdino-app: app/gdino_app.cpp $(BUILD)/libgdino_common.so | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INC) $(DS_INC) $(GST_CFLAGS) \
	  app/gdino_app.cpp -o $@ \
	  -L$(BUILD) -lgdino_common \
	  -L$(DS)/lib -lnvdsgst_meta \
	  $(shell pkg-config --libs gstreamer-1.0 glib-2.0) \
	  -L$(CUDA)/lib64 -lcudart $(RPATH)

clean:
	rm -rf $(BUILD)
