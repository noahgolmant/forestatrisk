#!/usr/bin/python
# -*- coding: utf-8 -*-

# ==============================================================================
# author          :Ghislain Vieilledent
# email           :ghislain.vieilledent@cirad.fr, ghislainv@gmail.com
# web             :https://ecology.ghislainv.fr
# python_version  :>=2.7
# license         :GPLv3
# ==============================================================================

# Import
from __future__ import division, print_function  # Python 3 compatibility
from glob import glob  # To explore files in a folder
import os  # Operating system interfaces
import sys  # To read and write files

# Third party imports
import numpy as np  # For arrays
from osgeo import gdal  # GIS libraries
import pandas as pd  # To export result as a pandas DF
import xarray as xr
import typing as t

# Local imports
from ..misc import makeblock, progress_bar, makeblock_xr


# sample()
def sample(dset: xr.Dataset,
           forest_band: str,
           feature_bands: t.List[str],
           nsamp=10000,
           adapt=True,
           seed=1234,
           csize=10,
           blk_rows=0
) -> pd.DataFrame:

    """Sample points and extract raster values.

    This function (i) randomly draws spatial points in deforested and
    forested areas and (ii) extract environmental variable values for
    each spatial point.

    :param dset: xarray dataset containing features and deforestation bands

    :param nsamp: Number of random spatial points.

    :param adapt: Boolean. Adapt ``nsamp`` to forest area: 1000 for 1 Mha of
        forest, with min=10000 and max=50000. Default to ``True``.

    :param seed: Seed for random number generator.

    :param csize: Spatial cell size in km.

    :param feature_bands: bands in dset to use as features

    :param forest_bands: Name of the forest raster data varaible
       (1=forest, 0=deforested)

    :param output_file: Path to file to save sample points.

    :param blk_rows: If > 0, number of lines per block.

    :return: A Pandas DataFrame, each row being one observation.

    """

    # Set random seed
    np.random.seed(seed)

    # =============================================
    # Sampling pixels
    # =============================================

    print("Sample 2x {} pixels (deforested vs. forest)".format(nsamp))

    # Read defor raster
    forestB = dset[forest_band]

    # Make blocks
    blockinfo = makeblock_xr(dset, blk_rows=blk_rows)
    nblock = blockinfo[0]
    nblock_x = blockinfo[1]
    x = blockinfo[3]
    y = blockinfo[4]
    nx = blockinfo[5]
    ny = blockinfo[6]
    print("Divide region in {} blocks".format(nblock))

    # Number of defor/forest pixels by block
    print("Compute number of deforested and forest pixels per block")
    ndc = 0
    ndc_block = np.zeros(nblock, dtype=int)
    nfc = 0
    nfc_block = np.zeros(nblock, dtype=int)

    # Loop on blocks of data
    for b in range(nblock):
        # Progress bar
        progress_bar(nblock, b + 1)
        # Position in 1D-arrays
        px = b % nblock_x
        py = b // nblock_x
        # Read the data into a numpy array
        forest = forestB.isel(x=slice(x[px], x[px] + nx[px]), y=slice(y[py], y[py] + ny[py]))
        forest = forest.values
        # Identify pixels (x/y coordinates) which are deforested
        deforpix = np.nonzero(forest == 0)
        ndc_block[b] = len(deforpix[0])  # Number of defor pixels
        ndc += len(deforpix[0])
        # Identify pixels (x/y coordinates) which are forest
        forpix = np.nonzero(forest == 1)
        nfc_block[b] = len(forpix[0])  # Number of forest pixels
        nfc += len(forpix[0])

    # Adapt nsamp to forest area
    if adapt is True:
        gt = dset.rio.transform()
        pix_area = gt[1] * (-gt[5])
        farea = pix_area * (nfc + ndc) / 10000  # farea in ha
        nsamp_prop = 1000 * farea / 1e6  # 1000 per 1Mha
        if nsamp_prop >= 50000:
            nsamp = 50000
        elif nsamp_prop <= 10000:
            nsamp = 10000
        else:
            nsamp = int(np.rint(nsamp_prop))
    else:
        nsamp = nsamp

    # Proba of drawing a block
    print("Draw blocks at random")
    proba_block_d = ndc_block / ndc
    proba_block_f = nfc_block / nfc
    # Draw block number nsamp times
    block_draw_d = np.random.choice(list(range(nblock)), size=nsamp,
                                    replace=True, p=proba_block_d)
    block_draw_f = np.random.choice(list(range(nblock)), size=nsamp,
                                    replace=True, p=proba_block_f)
    # Number of times the block is drawn
    nblock_draw_d = np.zeros(nblock, dtype=int)
    nblock_draw_f = np.zeros(nblock, dtype=int)
    for s in range(nsamp):
        nblock_draw_d[block_draw_d[s]] += 1
        nblock_draw_f[block_draw_f[s]] += 1

    # Draw defor/forest pixels in blocks
    print("Draw pixels at random in blocks")
    # Object to store coordinates of selected pixels
    deforselect = np.empty(shape=(0, 2), dtype=int)
    forselect = np.empty(shape=(0, 2), dtype=int)
    # Loop on blocks of data
    for b in range(nblock):
        # Progress bar
        progress_bar(nblock, b + 1)
        # nbdraw
        nbdraw_d = nblock_draw_d[b]
        nbdraw_f = nblock_draw_f[b]
        # Position in 1D-arrays
        px = b % nblock_x
        py = b // nblock_x
        # Read the data
        forest = forestB.isel(x=slice(x[px], x[px] + nx[px]), y=slice(y[py], y[py] + ny[py]))
        forest = forest.values
        # Identify pixels (x/y coordinates) which are deforested
        # !! Values returned in row-major, C-style order (y/x) !!
        deforpix = np.nonzero(forest == 0)
        deforpix = np.transpose((x[px] + deforpix[1],
                                 y[py] + deforpix[0]))
        ndc_block = len(deforpix)
        # Identify pixels (x/y coordinates) which are forested
        forpix = np.nonzero(forest == 1)
        forpix = np.transpose((x[px] + forpix[1],
                               y[py] + forpix[0]))
        nfc_block = len(forpix)
        # Sample deforested pixels
        if nbdraw_d > 0:
            if nbdraw_d < ndc_block:
                i = np.random.choice(ndc_block, size=nbdraw_d,
                                     replace=False)
                deforselect = np.concatenate((deforselect, deforpix[i]),
                                             axis=0)
            else:
                # nbdraw = ndc_block
                deforselect = np.concatenate((deforselect, deforpix),
                                             axis=0)
        # Sample forest pixels
        if nbdraw_f > 0:
            if nbdraw_f < nfc_block:
                i = np.random.choice(nfc_block, size=nbdraw_f,
                                     replace=False)
                forselect = np.concatenate((forselect, forpix[i]),
                                           axis=0)
            else:
                # nbdraw = ndc_block
                forselect = np.concatenate((forselect, forpix),
                                           axis=0)

    # =============================================
    # Compute center of pixel coordinates
    # =============================================

    print("Compute center of pixel coordinates")

    # Landscape variables from forest raster
    gt = dset.rio.transform()
    ncol_r = dset.rio.width
    nrow_r = dset.rio.height
    Xmin = gt[0]
    Xmax = gt[0] + gt[1] * ncol_r
    Ymin = gt[3] + gt[5] * nrow_r
    Ymax = gt[3]

    # Concatenate selected pixels
    select = np.concatenate((deforselect, forselect), axis=0)

    # Offsets and coordinates #
    xOffset = select[:, 0]
    yOffset = select[:, 1]
    pts_x = (xOffset + 0.5) * gt[1] + gt[0]  # +0.5 for center of pixels
    pts_y = (yOffset + 0.5) * gt[5] + gt[3]  # +0.5 for center of pixels

    # ================================================
    # Compute cell number for spatial autocorrelation
    # ================================================

    # Cell number from region
    print("Compute number of {} x {} km spatial cells".format(csize, csize))
    csize = csize * 1000  # Transform km in m
    ncol = int(np.ceil((Xmax - Xmin) / csize))
    nrow = int(np.ceil((Ymax - Ymin) / csize))
    ncell = ncol * nrow
    print("... {} cells ({} x {})".format(ncell, nrow, ncol))
    # bigI and bigJ are the coordinates of the cells and start at zero
    print("Identify cell number from XY coordinates")
    bigJ = ((pts_x - Xmin) / csize).astype(int)
    bigI = ((Ymax - pts_y) / csize).astype(int)
    cell = bigI * ncol + bigJ  # Cell number starts at zero

    # =============================================
    # Extract values from rasters
    # =============================================

    nobs = len(xOffset)
    feature_dset = dset[feature_bands]
    # Get values for all the specified coordinates
    x_coords = xr.DataArray(xOffset, dims='z')
    y_coords = xr.DataArray(yOffset, dims='z')

    feature_data = feature_dset.sel(x=x_coords, y=y_coords, method='nearest')
    # Convert to numpy array
    val = feature_data.values
    # Reshape to 2D array
    val = val.reshape((nobs, len(feature_bands)))
    # To float
    val = val.astype(np.float32)

    # Add XY coordinates and cell number
    pts_x.shape = (nobs, 1)
    pts_y.shape = (nobs, 1)
    cell.shape = (nobs, 1)
    val = np.concatenate((val, pts_x, pts_y, cell), axis=1)

    # Convert to pandas DataFrame and return the result
    colname = feature_bands + ["X", "Y", "cell"]
    val_DF = pd.DataFrame(val, columns=colname)
    return val_DF

# End
