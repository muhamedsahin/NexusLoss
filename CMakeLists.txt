cmake_minimum_required(VERSION 3.20)

# Ayni yapida ikinci kez add_subdirectory edilirse hedefler yeniden tanimlanmaz.
if(TARGET nexusloss)
    return()
endif()

project(nexusloss VERSION 0.1.0 LANGUAGES CXX)

# C++20 kullan
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Kütüphane: basliklar ve AVX2 tek-gecis kayip cekirdegi.
add_library(nexusloss STATIC src/fused_simd.cpp)
add_library(nexusloss::nexusloss ALIAS nexusloss)

target_include_directories(nexusloss PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(nexusloss PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(nexusloss PRIVATE /W4 /EHsc /utf-8)
    target_compile_options(nexusloss INTERFACE /W4 /utf-8)
    set_source_files_properties(src/fused_simd.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
    target_compile_options(nexusloss PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(nexusloss INTERFACE -Wall -Wextra -Wpedantic)
    set_source_files_properties(src/fused_simd.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()

# Aç/kapa seçenekleri
option(NEXUSLOSS_BUILD_TESTS "Testleri derle" ON)
option(NEXUSLOSS_BUILD_EXAMPLES "Ornekleri derle" ON)

# Örnekler
if(NEXUSLOSS_BUILD_EXAMPLES)
    add_executable(basic_example examples/main.cpp)
    target_link_libraries(basic_example PRIVATE nexusloss)
endif()

# Testler GoogleTest otomatik indirilir
if(NEXUSLOSS_BUILD_TESTS)
    enable_testing()
    include(FetchContent)
    FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(nexusloss_tests
            tests/test_smoke.cpp
            tests/test_classification.cpp
            tests/test_regression.cpp
            tests/test_segmentation.cpp
            tests/test_detection.cpp
            tests/test_distillation.cpp
            tests/test_generative.cpp
            tests/test_metric.cpp
            tests/test_point_cloud.cpp
            tests/test_ranking.cpp
            tests/test_reinforcement.cpp
            tests/test_self_supervised.cpp
            tests/test_sequence.cpp
            tests/test_survival.cpp
    )
    target_link_libraries(nexusloss_tests PRIVATE nexusloss GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(nexusloss_tests)
endif()