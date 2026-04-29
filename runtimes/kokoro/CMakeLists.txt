cmake_minimum_required(VERSION 3.20)
project(chirp_kokoro_c LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

set(PYKE_ORT_VERSION "1.24.4" CACHE STRING "Pyke ORT static binary version")

if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(PYKE_ORT_TARGET "aarch64-apple-darwin")
    set(PYKE_ORT_HASH "c75cd91afe59f8205ee3bb81100b88f0af9134b8e05bedb45ae8fa8eb37fef1a")
    set(ORT_HEADERS_ARCHIVE "onnxruntime-osx-arm64-${PYKE_ORT_VERSION}.tgz")
elseif(UNIX AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    set(PYKE_ORT_TARGET "x86_64-unknown-linux-gnu")
    set(PYKE_ORT_HASH "b9581353b714f22f5581e98c323ce36da01824f51e3258ef369277bb01c67b68")
    set(ORT_HEADERS_ARCHIVE "onnxruntime-linux-x64-${PYKE_ORT_VERSION}.tgz")
elseif(UNIX AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(PYKE_ORT_TARGET "aarch64-unknown-linux-gnu")
    set(PYKE_ORT_HASH "fa2de17abc12c9887de7e34fa389be2bd2a2c334b66994f476ffc2f94d004c20")
    set(ORT_HEADERS_ARCHIVE "onnxruntime-linux-aarch64-${PYKE_ORT_VERSION}.tgz")
elseif(WIN32 AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    set(PYKE_ORT_TARGET "x86_64-pc-windows-msvc")
    set(PYKE_ORT_HASH "469166ffcd6afb041d03d2d9a77f39d2a2c44c7800ae42e5fb688ef138ffbb6d")
    set(ORT_HEADERS_ARCHIVE "onnxruntime-win-x64-${PYKE_ORT_VERSION}.zip")
else()
    message(FATAL_ERROR "Unsupported Kokoro static ORT platform: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(PYKE_ORT_URL "https://cdn.pyke.io/0/pyke:ort-rs/ms@${PYKE_ORT_VERSION}/${PYKE_ORT_TARGET}.tar.lzma2")

FetchContent_Declare(
    onnxruntime_headers
    URL "https://github.com/microsoft/onnxruntime/releases/download/v${PYKE_ORT_VERSION}/${ORT_HEADERS_ARCHIVE}"
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
if(WIN32)
    set(PYKE_ORT_LIB "${PYKE_ORT_DIR}/onnxruntime.lib")
else()
    set(PYKE_ORT_LIB "${PYKE_ORT_DIR}/libonnxruntime.a")
endif()

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
    "$<TARGET_FILE:espeak-ng>"
    "$<TARGET_FILE:ucd>"
)
if(APPLE)
    target_link_libraries(chirp-kokoro-lib PRIVATE m c++ "-framework Foundation")
elseif(UNIX)
    target_link_libraries(chirp-kokoro-lib PRIVATE m stdc++ pthread dl)
elseif(WIN32)
    target_link_libraries(chirp-kokoro-lib PRIVATE dxguid DXCORE DXGI D3D12 DirectML ws2_32 bcrypt)
endif()
add_dependencies(chirp-kokoro-lib data espeak-ng ucd)

add_executable(kokoro-basic examples/basic.cpp)
target_link_libraries(kokoro-basic PRIVATE chirp-kokoro-lib)

add_executable(chirp-kokoro-worker worker/main.cpp)
target_link_libraries(chirp-kokoro-worker PRIVATE chirp-kokoro-lib)
