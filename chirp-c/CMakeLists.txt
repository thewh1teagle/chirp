cmake_minimum_required(VERSION 3.20)
project(chirp_runtime LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG b8941
)
FetchContent_GetProperties(llama_cpp)
if(NOT llama_cpp_POPULATED)
    FetchContent_Populate(llama_cpp)
endif()
add_subdirectory(${llama_cpp_SOURCE_DIR}/ggml ${llama_cpp_BINARY_DIR}/ggml)

FetchContent_Declare(
    dr_libs
    URL https://raw.githubusercontent.com/mackron/dr_libs/master/dr_wav.h
    DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_MakeAvailable(dr_libs)
FetchContent_GetProperties(dr_libs SOURCE_DIR DR_LIBS_SOURCE_DIR)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(KISSFFT_STATIC ON CACHE BOOL "" FORCE)
set(KISSFFT_TEST OFF CACHE BOOL "" FORCE)
set(KISSFFT_TOOLS OFF CACHE BOOL "" FORCE)
set(KISSFFT_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(KISSFFT_DATATYPE float CACHE STRING "" FORCE)
FetchContent_Declare(
    kissfft
    GIT_REPOSITORY https://github.com/mborgerding/kissfft.git
    GIT_TAG 6e9e673e420c4bf47d4a60c57c578f93e4ec192f
)
FetchContent_MakeAvailable(kissfft)

set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WITH_OPENMP OFF CACHE BOOL "" FORCE)
set(WITH_LSR_BINDINGS OFF CACHE BOOL "" FORCE)
set(WITH_AVFFT OFF CACHE BOOL "" FORCE)
set(WITH_CR32S OFF CACHE BOOL "" FORCE)
set(WITH_CR64S OFF CACHE BOOL "" FORCE)
set(WITH_PFFFT OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    soxr
    GIT_REPOSITORY https://github.com/chirlu/soxr.git
    GIT_TAG 945b592b70470e29f917f4de89b4281fbbd540c0
)
FetchContent_GetProperties(soxr)
if(NOT soxr_POPULATED)
    FetchContent_Populate(soxr)
endif()
list(PREPEND CMAKE_MODULE_PATH ${soxr_SOURCE_DIR}/cmake/Modules)
set(CMAKE_POLICY_DEFAULT_CMP0075 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0127 NEW)
add_subdirectory(${soxr_SOURCE_DIR} ${soxr_BINARY_DIR})

set(CHIRP_RUNTIME_SOURCES
    src/qwen3_tts.cpp
    src/text/qwen_tokenizer.cpp
    src/speaker/speaker_encoder.cpp
    src/ar/qwen3_tts_lifecycle.cpp
    src/ar/qwen3_tts_model.cpp
    src/ar/qwen3_tts_embeddings.cpp
    src/ar/qwen3_tts_prefill_graph.cpp
    src/ar/qwen3_tts_talker_graph.cpp
    src/ar/qwen3_tts_code_predictor_graph.cpp
    src/ar/qwen3_tts_forward.cpp
    src/ar/qwen3_tts_code_predictor.cpp
    src/ar/qwen3_tts_generate.cpp
    src/ar/gguf_loader.cpp
    src/ar/coreml_code_predictor_stub.cpp
    src/decoder/codec_decoder_lifecycle.cpp
    src/decoder/codec_decoder_model.cpp
    src/decoder/codec_decoder_blocks.cpp
    src/decoder/codec_decoder_graph.cpp
    src/decoder/codec_decoder_free.cpp
    src/decoder/codec_wav.cpp
    src/encoder/codec_tensor_reader.cpp
)

add_library(chirp-runtime-lib STATIC ${CHIRP_RUNTIME_SOURCES})

target_link_libraries(chirp-runtime-lib PUBLIC ggml kissfft soxr ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(chirp-runtime-lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ar
    ${CMAKE_CURRENT_SOURCE_DIR}/src/decoder
    ${CMAKE_CURRENT_SOURCE_DIR}/src/encoder
    ${CMAKE_CURRENT_SOURCE_DIR}/src/speaker
    ${CMAKE_CURRENT_SOURCE_DIR}/src/text
    ${DR_LIBS_SOURCE_DIR}
    ${soxr_SOURCE_DIR}/src
)

add_executable(chirp-runtime src/main.cpp)
target_link_libraries(chirp-runtime PRIVATE chirp-runtime-lib)
