# Copyright © 2020 Arm Ltd. All rights reserved.
# Copyright 2020 NXP
# SPDX-License-Identifier: MIT
import os
import importlib


def test_rel_version():
    import pyarmnn._version as v
    importlib.reload(v)
    assert "dev" not in v.__version__
    del v


def test_dev_version():
    import pyarmnn._version as v
    os.environ["PYARMNN_DEV_VER"] = "1"

    importlib.reload(v)

    assert "19.8.1.dev1" == v.__version__

    del os.environ["PYARMNN_DEV_VER"]
    del v