#
# Copyright 2020 NXP
# SPDX-License-Identifier: MIT
#

if(PYTHON_DEVENV_ENABLED)
    set(SETUP_PY_IN         "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
    set(SETUP_PY            "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
    set(SWIG_GENERATE_IN    "${CMAKE_CURRENT_SOURCE_DIR}/swig_generate.py.in")
    set(SWIG_GENERATE       "${CMAKE_CURRENT_BINARY_DIR}/swig_generate.py")
    set(DEPS                "${CMAKE_CURRENT_SOURCE_DIR}/src/pyarmnn/__init__.py"
                            "${CMAKE_CURRENT_SOURCE_DIR}/src/pyarmnn/_version.py")
    set(OUT_WRAPPERS        "${CMAKE_CURRENT_BINARY_DIR}/pyarmnn.wrappers.timestamp")

    configure_file(${SETUP_PY_IN} ${SETUP_PY})    
    configure_file(${SWIG_GENERATE_IN} ${SWIG_GENERATE})

    add_custom_command(OUTPUT ${OUT_WRAPPERS}
                       COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/README.md ${CMAKE_CURRENT_BINARY_DIR}
                       COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/licences.txt ${CMAKE_CURRENT_BINARY_DIR}
                       COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR}/src
                       COMMAND ${Python3_EXECUTABLE} ${SETUP_PY} clean --all
                       COMMAND ${Python3_EXECUTABLE} ${SWIG_GENERATE}
                       COMMAND ${Python3_EXECUTABLE} ${SETUP_PY} build_ext
                       COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src/pyarmnn/_generated ${CMAKE_CURRENT_BINARY_DIR}/src/pyarmnn/_generated
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUT_WRAPPERS}
                       DEPENDS ${DEPS})

    if(BUILD_PYTHON_SRC_PACKAGE)
        set(OUT_SRC "${CMAKE_CURRENT_BINARY_DIR}/pyarmnn.src.timestamp")
        add_custom_command(OUTPUT  ${OUT_SRC}
                           COMMAND ${Python3_EXECUTABLE} ${SETUP_PY} sdist
                           COMMAND ${CMAKE_COMMAND} -E touch ${OUT_SRC}
                           DEPENDS ${DEPS} ${OUT_WRAPPERS})
    endif()
    if(BUILD_PYTHON_WHL_PACKAGE)
        set(OUT_WHL "${CMAKE_CURRENT_BINARY_DIR}/pyarmnn.whl.timestamp")
        add_custom_command(OUTPUT  ${OUT_WHL}
                           COMMAND ${Python3_EXECUTABLE} ${SETUP_PY} bdist_wheel
                           COMMAND ${CMAKE_COMMAND} -E touch ${OUT_WHL}
                           DEPENDS ${DEPS} ${OUT_WRAPPERS})
    endif()
    add_custom_target(pyarmnn ALL DEPENDS ${OUT_WRAPPERS} ${OUT_SRC} ${OUT_WHL})
endif()
