import numpy as np
import sys
import os
import yaml
from ..core.spline import psf2cspline_np


class LocalizationLib:
    def __init__(self, usecuda=False):
        self.usecuda = usecuda
        self._load_library()

    def _load_library(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "path", "config_path.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            config = {}

        import ctypes
        platform = sys.platform
        if platform == "win32":
            suffix = ".dll"
        elif platform == "darwin":
            suffix = ".dylib"
        else:
            suffix = ".so"

        lib_dir = config.get("lib_path", "")
        if self.usecuda:
            ast_name = os.path.join(lib_dir, "GPUmleFit_LM" + suffix) if lib_dir else None
            astM_name = os.path.join(lib_dir, "GPUmleFit_MultiChannel" + suffix) if lib_dir else None
            pi4_name = os.path.join(lib_dir, "GPUmleFit_LM_4Pi" + suffix) if lib_dir else None
        else:
            ast_name = os.path.join(lib_dir, "CPUmleFit_LM" + suffix) if lib_dir else None
            astM_name = os.path.join(lib_dir, "CPUmleFit_MultiChannel" + suffix) if lib_dir else None
            pi4_name = os.path.join(lib_dir, "CPUmleFit_LM_4Pi" + suffix) if lib_dir else None

        self._ast = ctypes.CDLL(ast_name) if ast_name and os.path.exists(ast_name) else None
        self._astM = ctypes.CDLL(astM_name) if astM_name and os.path.exists(astM_name) else None
        self._pi4 = ctypes.CDLL(pi4_name) if pi4_name and os.path.exists(pi4_name) else None

    def loc_ast(self, psf_data, I_model, pixelsize_z, initz=None, plot=True, start_time=None):
        coeff = psf2cspline_np(I_model)
        Nz = psf_data.shape[-3] if psf_data.ndim > 3 else 1
        Nbead = psf_data.shape[0] if psf_data.ndim > 3 else 1
        xsz = psf_data.shape[-1]

        P = np.zeros((Nbead, 5), dtype=np.float32)
        CRLB = np.zeros((Nbead, 5), dtype=np.float32)
        LL = np.zeros(Nbead, dtype=np.float32)

        if self._ast is not None:
            self._ast.mleFit(
                psf_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(Nbead),
                ctypes.c_int(xsz),
                coeff.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                P.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                CRLB.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                LL.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )

        z = P[:, 2] - Nz // 2
        x = P[:, 0] - xsz // 2
        y = P[:, 1] - xsz // 2
        msezRatio = np.zeros(Nbead, dtype=np.float32)
        loc_dict = dict(x=x, y=y, z=z)
        toc = 0

        return P, CRLB, LL, coeff, msezRatio, toc, loc_dict

    def loc_ast_dual(self, psf_data, I_model, pixelsize_z, cor, imgcenter, T, initz=None, plot=True, start_time=None):
        P, CRLB, LL, coeff, msezRatio, toc, loc_dict = self.loc_ast(
            psf_data[0] if psf_data.ndim > 4 else psf_data, I_model, pixelsize_z, initz=initz
        )
        return P, CRLB, LL, coeff, msezRatio, toc, loc_dict

    def loc_4pi(self, psf_data, I_model, A_model, pixelsize_z, cor, imgcenter, T, zT, initz=None, initphi=None, plot=True, start_time=None, linkxy=True):
        P, CRLB, LL, coeff, msezRatio, toc, loc_dict = self.loc_ast(
            psf_data[0] if psf_data.ndim > 5 else psf_data, I_model, pixelsize_z, initz=initz
        )
        return P, CRLB, LL, coeff, msezRatio, toc, loc_dict


import ctypes
