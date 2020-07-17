#
# Copyright 2020 NXP
# SPDX-License-Identifier: MIT
#

cmake_minimum_required (VERSION 3.0.2) # same as armnn
project(pyarmnn)

set(additional_cmake_files)
list(APPEND additional_cmake_files
    cmake/GlobalConfig.cmake)

foreach(cmake_file ${additional_cmake_files})
    include(${cmake_file})
endforeach()

if (BUILD_PYTHON_WHL_PACKAGE OR BUILD_PYTHON_SRC_PACKAGE)
    add_subdirectory(python/pyarmnn)
else()
    message(STATUS "Nothing to build :)")
endif()
