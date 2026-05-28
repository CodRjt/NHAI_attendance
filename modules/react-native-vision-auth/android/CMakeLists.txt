cmake_minimum_required(VERSION 3.4.1)
project(VisionAuth)

set (CMAKE_VERBOSE_MAKEFILE ON)

add_library(
    react-native-vision-auth    STATIC
    ../cpp/VisionAuthImpl.cpp
)

set_target_properties(
    react-native-vision-auth PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

target_include_directories(
    react-native-vision-auth    PUBLIC
    ../cpp
)

target_link_libraries(
    react-native-vision-auth    jsi
    reactnative
    react_codegen_VisionAuthSpec
)
