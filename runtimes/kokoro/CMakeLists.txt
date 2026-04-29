cmake_minimum_required(VERSION 3.20)
project(chirp_kokoro_c LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

set(PYKE_ORT_VERSION "1.24.4" CACHE STRING "Pyke ORT static binary version")
set(PYKE_ORT_HASH "c75cd91afe59f8205ee3bb81100b88f0af9134b8e05bedb45ae8fa8eb37fef1a")

if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(PYKE_ORT_TARGET "aarch64-apple-darwin")
    set(PYKE_ORT_URL "https://cdn.pyke.io/0/pyke:ort-rs/ms@${PYKE_ORT_VERSION}/${PYKE_ORT_TARGET}.tar.lzma2")
else()
    message(FATAL_ERROR "This POC currently fetches Pyke static ORT for macOS arm64 only")
endif()

FetchContent_Declare(
    onnxruntime_headers
    URL "https://github.com/microsoft/onnxruntime/releases/download/v${PYKE_ORT_VERSION}/onnxruntime-osx-arm64-${PYKE_ORT_VERSION}.tgz"
)
FetchContent_MakeAvailable(onnxruntime_headers)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(COMPILE_INTONATIONS ON CACHE BOOL "" FORCE)
set(USE_LIBPCAUDIO OFF CACHE BOOL "" FORCE)
set(USE_SPEECHPLAYER OFF CACHE BOOL "" FORCE)
set(USE_ASYNC OFF CACHE BOOL "" FORCE)
set(USE_MBROLA OFF CACHE BOOL "" FORCE)
set(USE_LIBSONIC OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    espeak_ng
    GIT_REPOSITORY https://github.com/espeak-ng/espeak-ng.git
    GIT_TAG 1.52.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(espeak_ng)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_Declare(
    miniz
    GIT_REPOSITORY https://github.com/richgel999/miniz.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(INSTALL_PROJECT OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(miniz)

FetchContent_Declare(
    libnpy
    GIT_REPOSITORY https://github.com/llohse/libnpy.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(libnpy)

set(PYKE_ORT_DIR "${CMAKE_BINARY_DIR}/pyke-ort/${PYKE_ORT_TARGET}/${PYKE_ORT_HASH}")
set(PYKE_ORT_LIB "${PYKE_ORT_DIR}/libonnxruntime.a")

if(NOT EXISTS "${PYKE_ORT_LIB}")
    file(MAKE_DIRECTORY "${PYKE_ORT_DIR}")
    execute_process(
        COMMAND
            uv run python -c
            "import io, lzma, pathlib, sys, tarfile, urllib.request; url=sys.argv[1]; out=pathlib.Path(sys.argv[2]); data=urllib.request.urlopen(url, timeout=180).read(); dec=lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=[{'id': lzma.FILTER_LZMA2, 'dict_size': 1 << 26}]); tar_data=dec.decompress(data); tarfile.open(fileobj=io.BytesIO(tar_data), mode='r:').extractall(out)"
            "${PYKE_ORT_URL}"
            "${PYKE_ORT_DIR}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE PYKE_ORT_RESULT
    )
    if(NOT PYKE_ORT_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to fetch/extract Pyke static ONNX Runtime")
    endif()
endif()

if(NOT EXISTS "${PYKE_ORT_LIB}")
    message(FATAL_ERROR "Expected static ONNX Runtime library at ${PYKE_ORT_LIB}")
endif()

add_library(onnxruntime_static STATIC IMPORTED GLOBAL)
set_target_properties(onnxruntime_static PROPERTIES
    IMPORTED_LOCATION "${PYKE_ORT_LIB}"
)

add_library(chirp-kokoro-lib STATIC
    src/chirp_kokoro.cpp
    src/internal/chunk.cpp
    src/internal/kokoro_model.cpp
    src/internal/misaki.cpp
    src/internal/phonemize.cpp
    src/internal/voice_loader.cpp
)
target_include_directories(chirp-kokoro-lib PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/internal"
    "${CMAKE_CURRENT_SOURCE_DIR}/examples"
    "${onnxruntime_headers_SOURCE_DIR}/include"
    "${espeak_ng_SOURCE_DIR}/src/ucd-tools/src/include"
    "${espeak_ng_SOURCE_DIR}/src/include"
    "${espeak_ng_BINARY_DIR}/include"
    "${libnpy_SOURCE_DIR}/include"
)
target_link_libraries(chirp-kokoro-lib PRIVATE
    onnxruntime_static
    miniz
    "${espeak_ng_BINARY_DIR}/src/libespeak-ng/libespeak-ng.a"
    "${espeak_ng_BINARY_DIR}/src/ucd-tools/libucd.a"
    m
    c++
    "-framework Foundation"
)
add_dependencies(chirp-kokoro-lib data)

add_executable(kokoro-basic examples/basic.cpp)
target_link_libraries(kokoro-basic PRIVATE chirp-kokoro-lib)

add_executable(chirp-kokoro-worker worker/main.cpp)
target_link_libraries(chirp-kokoro-worker PRIVATE chirp-kokoro-lib)
