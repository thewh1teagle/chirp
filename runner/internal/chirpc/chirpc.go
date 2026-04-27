package chirpc

/*
#cgo CFLAGS: -I${SRCDIR}/../../third_party/chirp-c/include -I${SRCDIR}/../../../chirp-c/src
#cgo LDFLAGS: -L${SRCDIR}/../../third_party/chirp-c/lib -L${SRCDIR}/../../../chirp-c/build -lchirp-runtime-lib
#cgo LDFLAGS: -L${SRCDIR}/../../../chirp-c/build/_deps/llama_cpp-build/ggml/src -lggml -lggml-cpu -lggml-base
#cgo LDFLAGS: -L${SRCDIR}/../../../chirp-c/build/_deps/llama_cpp-build/ggml/src/ggml-blas
#cgo LDFLAGS: -L${SRCDIR}/../../../chirp-c/build/_deps/llama_cpp-build/ggml/src/ggml-metal
#cgo LDFLAGS: -L${SRCDIR}/../../../chirp-c/build/_deps/llama_cpp-build/ggml/src/ggml-vulkan
#cgo LDFLAGS: -L${SRCDIR}/../../../chirp-c/build/_deps/kissfft-build -lkissfft-float
#cgo LDFLAGS: -L${SRCDIR}/../../../chirp-c/build/_deps/soxr-build/src -lsoxr
#cgo linux LDFLAGS: -lggml-vulkan -lvulkan -Wl,-rpath,${SRCDIR}/../../third_party/chirp-c/lib -Wl,-rpath,${SRCDIR}/../../../chirp-c/build/_deps/llama_cpp-build/ggml/src
#cgo darwin LDFLAGS: -lggml-metal -lggml-blas -framework Accelerate -framework Metal -framework Foundation -framework MetalKit -framework CoreGraphics -Wl,-rpath,${SRCDIR}/../../third_party/chirp-c/lib -Wl,-rpath,${SRCDIR}/../../../chirp-c/build/_deps/llama_cpp-build/ggml/src
#cgo windows LDFLAGS: -lggml-vulkan -lvulkan-1
#cgo LDFLAGS: -lstdc++ -lm
#include "qwen3_tts.h"
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

type Params struct {
	ModelPath   string
	CodecPath   string
	MaxTokens   int
	Temperature float32
	TopK        int
}

type Context struct {
	ptr *C.qwen3_tts_context
}

func New(params Params) (*Context, error) {
	if params.ModelPath == "" {
		return nil, errors.New("model path is required")
	}
	if params.CodecPath == "" {
		return nil, errors.New("codec path is required")
	}

	cModel := C.CString(params.ModelPath)
	cCodec := C.CString(params.CodecPath)
	defer C.free(unsafe.Pointer(cModel))
	defer C.free(unsafe.Pointer(cCodec))

	cParams := C.qwen3_tts_default_params()
	cParams.model_path = cModel
	cParams.codec_path = cCodec
	if params.MaxTokens > 0 {
		cParams.max_tokens = C.int32_t(params.MaxTokens)
	}
	if params.Temperature >= 0 {
		cParams.temperature = C.float(params.Temperature)
	}
	if params.TopK > 0 {
		cParams.top_k = C.int32_t(params.TopK)
	}

	ptr := C.qwen3_tts_init(&cParams)
	if ptr == nil {
		return nil, errors.New("failed to initialize qwen3-tts context")
	}
	ctx := &Context{ptr: ptr}
	if msg := ctx.Error(); msg != "" {
		ctx.Close()
		return nil, errors.New(msg)
	}
	return ctx, nil
}

func (c *Context) Close() {
	if c == nil || c.ptr == nil {
		return
	}
	C.qwen3_tts_free(c.ptr)
	c.ptr = nil
}

func (c *Context) Error() string {
	if c == nil || c.ptr == nil {
		return "qwen3-tts context is nil"
	}
	return C.GoString(C.qwen3_tts_get_error(c.ptr))
}

func (c *Context) SynthesizeToFile(text, refPath, outputPath string) error {
	if c == nil || c.ptr == nil {
		return errors.New("qwen3-tts context is nil")
	}
	if text == "" {
		return errors.New("text is required")
	}
	if outputPath == "" {
		return errors.New("output path is required")
	}

	cText := C.CString(text)
	cOutput := C.CString(outputPath)
	defer C.free(unsafe.Pointer(cText))
	defer C.free(unsafe.Pointer(cOutput))

	var cRef *C.char
	if refPath != "" {
		cRef = C.CString(refPath)
		defer C.free(unsafe.Pointer(cRef))
	}

	if C.qwen3_tts_synthesize_to_file(c.ptr, cText, cRef, cOutput) == 0 {
		return fmt.Errorf("synthesis failed: %s", c.Error())
	}
	return nil
}
