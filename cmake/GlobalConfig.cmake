#
# Copyright 2020 NXP
# SPDX-License-Identifier: MIT
#

option(BUILD_PYTHON_WHL "Build Python wheel package" OFF)
option(BUILD_PYTHON_SRC "Build Python source package" OFF)
option(ARMNN_LIB "Path to ArmNN libraries" OFF)
option(ARMNN_INCLUDE "Path to ArmNN headers" OFF)

# Set to release configuration by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Setting and checking the python environment to be able to build whl/src packages
if(BUILD_PYTHON_WHL OR BUILD_PYTHON_SRC)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    
    if(BUILD_PYTHON_WHL)
        message(STATUS "PyArmNN wheel package is enabled")
    else()
        message(STATUS "PyArmNN wheel package is disabled")
    endif()
    if(BUILD_PYTHON_SRC)
        message(STATUS "PyArmNN source package is enabled")
    else()
        message(STATUS "PyArmNN source package is disabled")
    endif()
    if(NOT ${Python3_FOUND})
        message(FATAL_ERROR "Python 3.x not found")
    endif()
    message(STATUS "Python ${Python3_VERSION} found at ${Python3_EXECUTABLE}")
    if(NOT ${Python3_Development_FOUND})
        message(FATAL_ERROR "Python development package not found")
    endif() 
    if (DEFINED SWIG_DIR)
        if(EXISTS "${SWIG_DIR}/bin/swig")
            set(SWIG_EXECUTABLE "${SWIG_DIR}/bin/swig")
            execute_process(COMMAND ${SWIG_EXECUTABLE} -version
                OUTPUT_VARIABLE SWIG_version_output
                ERROR_VARIABLE SWIG_version_output
                RESULT_VARIABLE SWIG_version_result)
            if(SWIG_version_result)
                message(WARNING "Command \"${SWIG_EXECUTABLE} -version\" failed with output:\n${SWIG_version_output}")
                find_package(SWIG)
            else()
                string(REGEX REPLACE ".*SWIG Version[^0-9.]*\([0-9.]+\).*" "\\1"
                SWIG_version_output "${SWIG_version_output}")
                set(SWIG_VERSION ${SWIG_version_output} CACHE STRING "Swig version" FORCE)
                set(SWIG_FOUND True)
            endif()
        else()
            find_package(SWIG)
        endif()
    else()
        find_package(SWIG)
    endif()
    if (SWIG_EXECUTABLE)
        message(STATUS "SWIG ${SWIG_VERSION} found at ${SWIG_EXECUTABLE}")
        string(REPLACE "." ";" VERSION_LIST ${SWIG_VERSION})
        list(GET VERSION_LIST 0 SWIG_VERSION_MAJOR)
        list(GET VERSION_LIST 1 SWIG_VERSION_MINOR)
        list(GET VERSION_LIST 2 SWIG_VERSION_PATCH)
        if (${SWIG_VERSION_MAJOR} LESS 4)
            message(FATAL_ERROR "SWIG version 4.x required")
        endif()
    else()
        message(FATAL_ERROR "SWIG not found")
    endif()
    # if all goes well PYTHON_DEVENV_ENABLED can be checked in other cmakes
    set(PYTHON_DEVENV_ENABLED ON)
    
    # ARMNN_INCLUDE and ARMNN_LIB must be manually set
    if (NOT ARMNN_INCLUDE)
        if ($ENV{ARMNN_INCLUDE})
            set(ARMNN_INCLUDE $ENV{ARMNN_INCLUDE})
        else()
            message(FATAL_ERROR "ARMNN_INCLUDE not set")
        endif()
    endif()
    
    if (NOT ARMNN_LIB)
        if ($ENV{ARMNN_LIB})
            set(ARMNN_LIB $ENV{ARMNN_LIB})
        else()
            message(FATAL_ERROR "ARMNN_LIB not set")
        endif()
    endif()
    message(STATUS "ArmNN headers found at ${ARMNN_INCLUDE}")
    message(STATUS "ArmNN libraries found at ${ARMNN_LIB}")
endif()
