//
// Copyright © 2017 Arm Ltd. All rights reserved.
// Copyright 2020 NXP
// SPDX-License-Identifier: MIT
//
%module pyarmnn_version

%include "std_string.i"

%{
#define SWIG_FILE_WITH_INIT
#include "armnn/Version.hpp"
%}

%{
    std::string GetVersion()
    {
        return ARMNN_VERSION;
    };

    std::string GetMajorVersion()
    {
        return "19";
    };

    std::string GetMinorVersion()
    {
        return "08";
    };
%}
%feature("docstring",
"
    Returns Arm NN library full version: MAJOR + MINOR + INCREMENTAL.

    Returns:
        str: Full version of Arm NN installed.

") GetVersion;
std::string GetVersion();

%feature("docstring",
"
    Returns Arm NN library major version. The year of the release.

    Returns:
        str: Major version of Arm NN installed.

") GetMajorVersion;
std::string GetMajorVersion();

%feature("docstring",
"
    Returns Arm NN library minor version. Month of the year of the release.

    Returns:
        str: Minor version of Arm NN installed.

") GetMinorVersion;
std::string GetMinorVersion();
