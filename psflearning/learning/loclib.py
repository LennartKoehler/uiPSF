from __future__ import annotations

"""
Copyright (c) 2022      Ries Lab, EMBL, Heidelberg, Germany
All rights reserved

@author: Sheng Liu
"""
#%%
import ctypes
import numpy.ctypeslib as ctl
import numpy as np
import h5py as h5
import logging
from .utilities import psf2cspline_np
import matplotlib.pyplot as plt
from ..progress import ProgressReporter
import os
import sys
from psflearning import io
from dataclasses import dataclass
from typing import Any, Optional, Union


def _cuda_available() -> bool:
    """Check CUDA availability via the CUDA driver library, without TensorFlow."""
    try:
        if sys.platform.startswith('linux'):
            ctypes.CDLL('libcuda.so.1')
        elif sys.platform.startswith('win'):
            ctypes.CDLL('nvcuda.dll')
        else:
            return False
        return True
    except OSError:
        return False


@dataclass
class Positions:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    zast: Optional[np.ndarray] = None

    def to_dict(self) -> dict[str, np.ndarray]:
        d: dict[str, np.ndarray] = {"x": self.x, "y": self.y, "z": self.z}
        if self.zast is not None:
            d["zast"] = self.zast
        return d



@dataclass
class LocalizationResult:
    """Structured result from MLE localization.

    Created by :func:`localizationlib.loc_ast` and :func:`localize`.
    """

    parameters: np.ndarray
    crlb: np.ndarray
    log_likelihood: np.ndarray
    spline_coefficients: np.ndarray
    mse_z_ratio: np.ndarray
    positions: Positions

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "mle_parameters": self.parameters,
            "cramer_rao_bounds": self.crlb,
            "log_likelihoods": self.log_likelihood,
            "spline_coefficients": self.spline_coefficients,
            "localized_positions": self.positions.to_dict(),
        }
        return d
#%%
class localizationlib:
    def __init__(self, usecuda: bool = False) -> None:
        """Initialize the localization library with CPU or GPU fitting routines."""
        thispath = os.path.dirname(os.path.abspath(__file__))
        pkgpath = os.path.dirname(os.path.dirname(thispath))
        cfg = io.param.load(pkgpath+'/config/path/config_path.yaml')
        if sys.platform.startswith('win'):
            dllpath_cpu_ast = pkgpath+cfg.Paths.spline.win.cpu.ast
            dllpath_gpu_ast = pkgpath+cfg.Paths.spline.win.cuda.ast

            if _cuda_available():
                lib_gpu_ast = ctypes.CDLL(dllpath_gpu_ast)
            else:
                usecuda = False
        elif sys.platform.startswith('darwin'):
            usecuda = False
            dllpath_cpu_ast = pkgpath+cfg.Paths.spline.mac.cpu.ast
        elif sys.platform.startswith('linux'):

            dllpath_cpu_ast = pkgpath+cfg.Paths.spline.linux.cpu.ast
            dllpath_gpu_ast = pkgpath+cfg.Paths.spline.linux.cuda.ast
            if _cuda_available():
                lib_gpu_ast = ctypes.CDLL(dllpath_gpu_ast)
            else:
                usecuda = False

        try:
            lib_cpu_ast = ctypes.CDLL(dllpath_cpu_ast)
        except OSError:
            logging.warning('MLE CPU fitting is not available')

        if usecuda:
            self._mleFit = lib_gpu_ast.GPUmleFit_LM
        else:
            self._mleFit = lib_cpu_ast.CPUmleFit_LM

        self._mleFit.argtypes = [
            ctl.ndpointer(np.float32), # data
            ctypes.c_int32,          # fittype
            ctypes.c_int32,          # iterations
            ctl.ndpointer(np.float32), # spline_coeff
            ctl.ndpointer(np.float32), # varim
            ctypes.c_float,             # init_z
            ctl.ndpointer(np.int32),   # datasize
            ctl.ndpointer(np.int32),    # spline_size
            ctl.ndpointer(np.float32), # P
            ctl.ndpointer(np.float32), # CRLB
            ctl.ndpointer(np.float32) # LL
        ]


    def loc_ast(self, rois: np.ndarray, I_model: np.ndarray, pixelsize_z: float, initz: np.ndarray | None = None, reporter: ProgressReporter | None = None) -> LocalizationResult:
        """Perform single-channel astigmatic localization fitting."""
        rsz = rois.shape[-1]
        Nbead = rois.shape[0]
        if len(rois.shape)>3:
            Nz = rois.shape[-3]
        else:
            Nz = 1
        Nfit = Nbead*Nz
        Nparam = 5
        offset = np.min(I_model)
        Imd = I_model-offset
        normf = np.median(np.sum(Imd,axis = (-1,-2)))
        Imd = Imd/normf

        reporter.begin_stage('4/6: calculating spline coefficients', total=1)
        coeff = psf2cspline_np(Imd)
        reporter.update(1)
        reporter.close()

        coeff = coeff.astype(np.float32)
        data = rois.reshape((Nfit,rsz,rsz))
        bxsz = np.min((rsz,20))
        data = data[:,rsz//2-bxsz//2:rsz//2+bxsz//2,rsz//2-bxsz//2:rsz//2+bxsz//2].astype(np.float32)
        data = np.maximum(data,0.0)

        ccz = coeff.shape[-3]//2
        if initz is None:
            Nzm = Imd.shape[0]
            initz = np.linspace(-Nzm*pixelsize_z/2,Nzm*pixelsize_z/2,np.int32(Nzm*pixelsize_z/0.5))*0.8/pixelsize_z+ccz
            #else:
            #    initz = np.array([-1,0,1])*0.5/pixelsize_z+ccz
        else:
            initz = np.array(initz)*0.5/pixelsize_z+ccz
        zstart = initz.astype(np.float32)

        datasize = np.array(np.flip(data.shape)).astype(np.int32)
        splinesize = np.array(np.flip(coeff.shape)).astype(np.int32)
        varim = np.array((0)).astype(np.float32)
        Pk = np.zeros((Nparam+1,Nfit)).astype(np.float32)
        CRLBk = np.zeros((Nparam,Nfit)).astype(np.float32)
        LLk = np.zeros((Nfit)).astype(np.float32)
        fittype = np.int32(5)
        iterations = np.int32(100)
        P = np.zeros((Nparam+1,Nfit)).astype(np.float32)
        CRLB = np.zeros((Nparam,Nfit)).astype(np.float32)
        LL = np.zeros((Nfit)).astype(np.float32)-1e10

        reporter.begin_stage('5/6: localization', total=len(zstart))

        for z0 in zstart:

            self._mleFit(data,fittype,iterations,coeff,varim,z0,datasize,splinesize,Pk,CRLBk,LLk)
            mask = (LLk-LL)>1e-4
            LL[mask] = LLk[mask]
            P[:,mask] = Pk[:,mask]
            CRLB[:,mask] = CRLBk[:,mask]

            reporter.update(1)

        reporter.close()


        zf = P[4].reshape((Nbead,Nz))
        xf = P[1].reshape((Nbead,Nz))
        yf = P[0].reshape((Nbead,Nz))

        zg = np.linspace(0,Nz-1,Nz)
        if Nz>1:
            zf = zf-np.median(zf-zg,axis=1,keepdims=True)
            zdiff = zf-zg
            xf = xf-np.median(xf,axis=1,keepdims=True)
            yf = yf-np.median(yf,axis=1,keepdims=True)
            if Nz>4:
                zind = range(2,Nz-2,1)
            else:
                zind = range(0,Nz,1)

            zdiff = zdiff-np.mean(zdiff[:,zind],axis=1,keepdims=True)
            msez = np.mean(np.square((np.median(zf-zg,axis=0)-(zf-zg))[:,zind]),axis=1)
        else:
            zdiff = zf
            msez = np.array([1.0])


        if Nbead == 1:
            msezRatio = np.array([1.0])
        else:
            msezRatio =msez/(np.median(msez)+1e-6)
        loc_dict = Positions(x=xf, y=yf, z=zf)

        return LocalizationResult(
            parameters=P,
            crlb=CRLB,
            log_likelihood=LL,
            spline_coefficients=coeff,
            mse_z_ratio=msezRatio,
            positions=loc_dict,
        )
