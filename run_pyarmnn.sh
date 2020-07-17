#!/bin/bash
# Copyright 2020 NXP
# SPDX-License-Identifier: MIT

PYARMNN_REPO=`pwd`
HELP_STR="This is a simple script to build, install or test pyarmnn.\n\nPlease export or set the following variables for this script to be fully functional:\n \
ARMNN_LIB       - Path to ArmNN libraries\n \
ARMNN_INCLUDE   - Path to ArmNN include folder\n \
PROTOBUF_LIB    - Path to Protocol Buffers libraries\n"
USAGE_STR="Usage: ./`basename "$0"` [build/test/install/gen_html_doc/download_res/all]"

# VARIABLE SETTINGS
# ARMNN_LIB=<armnn_lib_path>
# ARMNN_INCLUDE=<armnn_include_path>
# PROTOBUF_LIB=<protobuf_lib_path>
###

run_build()
{
    if [ -z $ARMNN_LIB ] || [ -z $ARMNN_INCLUDE ]
    then
        echo "Error: ARMNN_LIB or ARMNN_INCLUDE not set. Unable to build."
        exit 0
    fi
    cd $PYARMNN_REPO
    rm -rf build
    mkdir build
    cd build
    cmake .. -DBUILD_PYTHON_WHL_PACKAGE=1 -DBUILD_PYTHON_SRC_PACKAGE=0 -DARMNN_INCLUDE=$ARMNN_INCLUDE -DARMNN_LIB=$ARMNN_LIB
    make -j6
}

run_test()
{
    if [ -z $ARMNN_LIB ]
    then        
        echo "Warning: ARMNN_LIB not set. Your application might not link all the dependencies."
    else
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ARMNN_LIB
    fi
    if [ -z $PROTOBUF_LIB ]
    then        
        echo "Warning: PROTOBUF_LIB not set. Your application might not link all the dependencies."
    else
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PROTOBUF_LIB
    fi
    if [ -z $1 ]
    then
        python3 -m pytest $PYARMNN_REPO/python/pyarmnn/test/ -v
    else
        python3 -m pytest $PYARMNN_REPO/python/pyarmnn/test/$1.py -v
    fi
}

run_genhtmldoc()
{
    if [ -z $ARMNN_LIB ]
    then        
        echo "Warning: ARMNN_LIB not set. Your application might not link all the dependencies."
    else
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ARMNN_LIB
    fi
    if [ -z $PROTOBUF_LIB ]
    then        
        echo "Warning: PROTOBUF_LIB not set. Your application might not link all the dependencies."
    else
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PROTOBUF_LIB
    fi    
    cd $PYARMNN_REPO/python/pyarmnn
    rm -rf docs
    python3 $PYARMNN_REPO/python/pyarmnn/scripts/generate_docs.py -o $PYARMNN_REPO/python/pyarmnn/docs --html pyarmnn
}

run_dlres()
{
    python3 $PYARMNN_REPO/python/pyarmnn/scripts/download_resources.py
}

run_install()
{
    pip3 uninstall pyarmnn
    pip3 install cd $PYARMNN_REPO/build/python/pyarmnn/dist/pyarmnn-*.whl
}

unset LD_LIBRARY_PATH
if [ "$1" = "build" ]
then
    run_build
elif [ "$1" = "test" ]
then
    run_test $2
elif [ "$1" = "install" ]
then
    run_install
elif [ "$1" = "gen_html_doc" ]
then
    run_genhtmldoc
elif [ "$1" = "download_res" ]
then
    run_dlres    
elif [ "$1" = "all" ]
then
    run_build
    run_install
    run_genhtmldoc
    run_dlres
    run_test $2
elif [ -z $1 ]
then
    echo -e "No parameters passed.\n$USAGE_STR"
elif [ "$1" = "help" ]
then
    echo -e "$HELP_STR\n\n$USAGE_STR"
else
    echo -e "Invalid usage.\n\n$USAGE_STR"
fi
