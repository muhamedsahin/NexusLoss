cmake_minimum_required(VERSION 3.20)
project(nexusloss VERSION 0.1.0 LANGUAGES CXX)

# C++20 kullan
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Kütüphanenin kendisi (header-only)
# INTERFACE değerlenecek cpp dosyası yok demek yani header only bir kütüphane anlamına geliyor
add_library(nexusloss INTERFACE)
add_library(nexusloss::nexusloss ALIAS nexusloss)

target_include_directories(nexusloss INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(nexusloss INTERFACE cxx_std_20)

if(MSVC)
    target_compile_options(nexusloss INTERFACE /W4 /utf-8)
else()
    target_compile_options(nexusloss INTERFACE -Wall -Wextra -Wpedantic)
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