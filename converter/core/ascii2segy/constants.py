# -*- coding: utf-8 -*-
from segpy import datatypes
from segpy import encoding
import numpy as np

ENDIAN = '>'
DEFAULT_SEGY_ENCODING = encoding.EBCDIC
SAMPLE_RATE = 32.767
SEG_Y_TYPE = datatypes.DATA_SAMPLE_FORMAT_TO_SEG_Y_TYPE[1]
SEISMIC_DIMENSIONS = 3
NUMPY_DTYPES = {
    'ibm': np.dtype('f4'),
    'int32': np.dtype('i4'),
    'int16': np.dtype('i2'),
    'float32': np.dtype('f4'),
    'int8': np.dtype('i1')
}
DTYPE = NUMPY_DTYPES[SEG_Y_TYPE].type
OUTFILE = 'file.segy'
USE_EXTENDED_HEADER = False
