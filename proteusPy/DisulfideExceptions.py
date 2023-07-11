# Copyright (C) 2022, Eric G. Suchanek, PhD. (suchanek@mac.com)
#
# This file is part of the ProteusPy distribution and governed by your
# choice of the "ProteusPy License Agreement" or the "BSD 3-Clause License".
# Please see the LICENSE file that should have been included as part of this
# package.

import os
import warnings

from proteusPy.ProteusPyWarning import ProteusPyWarning

"""Some Disulfide-specific exceptions."""

# General error
class DisulfideException(Exception):
    """Define class DisulfideException."""

    pass

class DisulfideConstructionException(Exception):
    """Define class DisulfideConstructionException."""

    pass


class DisulfideConstructionWarning(ProteusPyWarning):
    """Define class DisulfideConstructionWarning."""

    pass

class DisulfideParseWarning(ProteusPyWarning):
    """Define class DisulfideConstructionWarning."""

    pass

# The SMCRA structure could not be written to file
class DisulfideIOException(Exception):
    """Define class DisulfideIOException."""

    pass
