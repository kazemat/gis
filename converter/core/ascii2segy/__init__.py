# -*- coding: utf-8 -*-
import logging
import re
import sys
import warnings
from typing import List

import numpy as np
import pandas as pd
import pylab
import segpy
from segpy import binary_reel_header
from segpy import toolkit
from segpy import trace_header

from converter.core.ascii2segy.grid import Grid
from converter.core.ascii2segy.constants import *


np.set_printoptions(suppress=True)

# aliases
ar = np.array

logger = logging.getLogger(__name__)

# check if the python running this is 64 bit
python_is_64 = sys.maxsize > 2 ** 32
if not python_is_64:
    warnings.warn("Warning you are not running in 64bit python, you will likely run into memory errors")


# ==============================================================================
# Read ascii data
# ==============================================================================


def get_frame(items: List[List]) -> pd.DataFrame:
    data = pd.DataFrame(items, columns=['x', 'y', 't', 'v', 'inline', 'xline'])
    data = data.sort_index(by=['inline', 'xline', 't'])

    zeros_in_max_xline = int(np.ceil(np.log10(data.xline.max())))
    data['itrace'] = data.inline.multiply(10 ** zeros_in_max_xline) + data.xline

    return data


def angles(line):
    """get angles between points in line"""
    x = line[:, 0]
    y = line[:, 1]
    ang = np.arctan2(np.diff(x), np.diff(y))
    ang2 = np.unwrap(ang)
    return np.array(ang2)


# ==============================================================================
# Format file header into segy header
# ==============================================================================
# format_header='.'.join(header).replace('  ',' ').replace('\n','\\')

def minify(s):
    if isinstance(s, list):
        s = '\n'.join([h.strip() for h in s])
    s = re.sub(r'\s+', ' ', s)
    return s


def checks(data):
    # group by trace, for times get the difference, and drop NotANumbers
    time_diffs = data.groupby(data.itrace).t.diff().dropna()
    vel_diffs = data.groupby(data.itrace).v.diff().dropna()
    trace_lengths = data.itrace.groupby(data.itrace).count()

    logger.debug("Data summary")
    logger.debug(data.describe())
    logger.debug(data.info())
    logger.debug("Time intervals")
    logger.debug(time_diffs.describe())
    logger.debug("Vel diffs")
    logger.debug(vel_diffs.describe())
    logger.debug("Trace lengths")
    logger.debug(trace_lengths.describe())

    np.testing.assert_((time_diffs >= 0).all(),
                       msg='Not all times increasing per trace: {}'.format(time_diffs[time_diffs < 0]))

    # I assertt it should be sorted
    np.testing.assert_(np.alltrue(data.itrace.diff().dropna() >= 0), msg="Data is not sorted")

    # not NaN
    assert (data.notnull().all().all())

    # Inlines are not decimals
    np.testing.assert_array_equal(data.inline, data.inline.astype(int), err_msg="Inlines are not integers")
    # Crosslines are not decimals
    np.testing.assert_array_equal(data.xline, data.xline.astype(int), err_msg="Crosslines are not integers")

    logger.debug("Input data OK")


# ==============================================================================
# Write SEGY
# ==============================================================================


def get_header(header: List):
    if any(ar([len(s) for s in header]) > 80):  # if they are all shorted than 80
        header = minify(header)  # else wrapthem and remove extra whitespac
        split_header = []
        inds = range(0, len(header), 80)
        for i in range(len(inds) - 1):
            header_line = header[inds[i]:inds[i + 1]]
            split_header.append(header_line)
    else:
        split_header = header
    joined_header = split_header
    textual_reel_header = joined_header + (40 - len(joined_header)) * [' ' * 80]
    # remove extra
    textual_reel_header = [s[:80] + (80 - len(s)) * ' ' for s in textual_reel_header]
    textual_reel_header = textual_reel_header[:40]

    return textual_reel_header


def write_segy_file(data: pd.DataFrame, outfile: str):
    metadata = data.groupby(data.itrace).first()  # one row for each trace

    # lets work out the whole inline xline grid
    inlines_r = pd.Series(data.inline.unique())

    xlines_r = pd.Series(data.xline.unique())

    # choose interval as most common inline step, should I use min but ignore zero and nan?
    inline_int = inlines_r.diff().mode().values
    xline_int = xlines_r.diff().mode().values
    inlines = np.arange(inlines_r.min(), inlines_r.max(), inline_int)
    xlines = np.arange(xlines_r.min(), xlines_r.max(), xline_int)

    # find inline and xline with most data
    xline_lens = metadata.xline.groupby(metadata.xline).count()
    longest_xline_n = xline_lens.argmax()
    longest_xline = metadata[metadata.xline == longest_xline_n]
    inline_lens = metadata.inline.groupby(metadata.inline).count()
    longest_inline_n = inline_lens.argmax()
    longest_inline = metadata[metadata.inline == longest_inline_n]

    if SEISMIC_DIMENSIONS == 3:
        # work out bins spacing
        ibins = pylab.distances_along_curve(longest_xline[['x', 'y']]) / longest_xline.inline.diff().iloc[1:]
        ibin = ibins.mean()
        xbins = pylab.distances_along_curve(longest_inline[['x', 'y']]) / longest_inline.xline.diff().iloc[1:]
        xbin = xbins.mean()

        inline_angs = angles(longest_inline[['x', 'y']].values)
        inline_rot = inline_angs.mean()

        pi = metadata.iloc[0]  # reference point, might not be origin as we don't have that yet
        pj = metadata.iloc[100]  # use this for a test
        gd = Grid(pi, inline_rot, ibin, xbin)
        x, y = gd.ix2xy(pj.inline, pj.xline)

        # check the predicted coords are right withing a meter
        np.testing.assert_allclose(pj.x, x, atol=1)
        np.testing.assert_allclose(pj.y, y, atol=1)

    logger.info("Writing segy {}".format(OUTFILE))
    tmin = data.t.min()
    # times to interpolate onto
    itimes = np.arange(tmin, data.t.max() + SAMPLE_RATE, SAMPLE_RATE)

    template_trace = segpy.trace_header.TraceHeaderRev1()
    template_binary_reel_header = segpy.binary_reel_header.BinaryReelHeader()

    header = []
    textual_reel_header = get_header(header)
    metadata = data.groupby(data.itrace).first()

    # Format text header
    if USE_EXTENDED_HEADER:
        logger.debug("Formating extended textual header")
        extended_textual_header = toolkit.format_extended_textual_header('', DEFAULT_SEGY_ENCODING)
    else:
        extended_textual_header = toolkit.format_extended_textual_header(''.join(header), DEFAULT_SEGY_ENCODING)

    # Format binary header
    binary_reel_header = template_binary_reel_header
    # samples length microseconds
    binary_reel_header.sample_interval = int(SAMPLE_RATE * 1000)

    # number of samples
    binary_reel_header.num_samples = len(itimes)
    # len(data) # also must be # not sure how to work this out,
    #  maybe I need to scan the file first or after insert it
    binary_reel_header.data_traces_per_ensemble = 0
    binary_reel_header.auxiliary_traces_per_ensemble = 0
    # http://oivdoc91.vsg3d.com/APIS/RefManCpp/struct_so_v_r_segy_file_header.html#a612dab3b4d9c671ba6554a8fb4a88057
    binary_reel_header.trace_sorting = 4

    # 0 or 1 TODO move to 1
    binary_reel_header.format_revision_num = 256

    # see binary header def.py file as it changes by revision. 1 is always ibm float
    binary_reel_header.data_sample_format = DTYPE
    if USE_EXTENDED_HEADER:
        # see binary header def.py file as it changes by revision. 1 is always ibm float
        binary_reel_header.num_extended_textual_headers = len(extended_textual_header)

        # Pre-Format trace headerf
        trace_header_format = toolkit.make_header_packer(segpy.trace_header.TraceHeaderRev1, ENDIAN)

    if SEISMIC_DIMENSIONS == 3:
        # either iterate over the grid for 3d
        xxlines, iinlines = np.meshgrid(xlines, inlines)
        trace_iter = np.vstack([iinlines.flat, xxlines.flat]).T
    else:
        # or for 2d just iterate over cdp an sp
        trace_iter = np.vstack([inlines_r, xlines_r]).T

    i = 0
    tracen = 1

    with open(outfile, 'wb') as fo:
        # Write headers
        toolkit.write_textual_reel_header(fo, textual_reel_header, DEFAULT_SEGY_ENCODING)
        toolkit.write_binary_reel_header(fo, binary_reel_header, ENDIAN)
        if USE_EXTENDED_HEADER:
            toolkit.write_extended_textual_headers(fo, extended_textual_header, DEFAULT_SEGY_ENCODING)

        for inline, xline in trace_iter:
            i += 1
            if ((metadata.inline == inline) * (metadata.xline == xline)):
                trace = data[(data.inline == inline) * (data.xline == xline)]
                metatrace = metadata[(metadata.inline == inline) * (metadata.xline == xline)]
                x = metatrace.x
                y = metatrace.y
                times = trace.t.values
                vels = trace.v.values
            elif SEISMIC_DIMENSIONS == 3:
                x, y = gd.ix2xy(inline, xline)
                times = itimes
                vels = np.zeros(itimes.shape)
            else:
                logger.warning("inline/xline or cdp/sp not found", inline, xline)
                continue

            cdp = inline
            sp = xline

            if i % 1000 == 0:
                logger.debug("Writing trace: "
                             "{: 8.0f}/{}, {: 6.2f}%".format(i, len(trace_iter),
                                                             (1.0 * i / len(trace_iter)) * 100))

            if len(vels) == 0:
                logger.error("Error no vels on trace", i)
                continue

            # interpolate data
            samples = np.interp(itimes, times, vels)

            # ensure datatype is ok
            samples = np.require(samples, dtype='d')

            # Format trace header
            trace_header = template_trace
            trace_header.line_sequence_num = 1000 + tracen
            trace_header.field_record_num = tracen
            trace_header.trace_num = tracen
            if SEISMIC_DIMENSIONS == 3:
                trace_header.file_sequence_num = inline
                trace_header.ensemble_num = xline
                trace_header.inline_number = inline
                trace_header.crossline_number = xline
            else:
                trace_header.file_sequence_num = 1000 + tracen
                trace_header.ensemble_num = cdp
                trace_header.shotpoint_number = sp
            trace_header.num_samples = len(samples)  # number of samples
            trace_header.sample_interval = SAMPLE_RATE  # sample interval
            trace_header.cdp_x = x
            trace_header.cdp_y = y
            trace_header.source_x = x
            trace_header.source_y = y

            # write trace header and data
            toolkit.write_trace_header(fo, trace_header, trace_header_format, pos=None)
            toolkit.write_trace_samples(fo, samples=samples, ctype=SEG_Y_TYPE, endian=ENDIAN, pos=None)

            tracen += 1
        logger.debug(
            "Writing trace: {: 8.0f}/{}, {: 6.2f}%".format(i,
                                                           len(trace_iter),
                                                           (1.0 * i / len(trace_iter)) * 100))
