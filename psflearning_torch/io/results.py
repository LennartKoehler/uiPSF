import json
import numpy as np
import h5py as h5
from omegaconf import OmegaConf
from ..core.spline import psf2cspline_np


def save_results(filename, param, res_dict, locres_dict, rois_dict):
    with h5.File(filename, "w") as f:
        f.attrs["params"] = json.dumps(OmegaConf.to_container(param))
        g3 = f.create_group("rois")
        g1 = f.create_group("res")
        g2 = f.create_group("locres")

        for k, v in locres_dict.items():
            if isinstance(v, dict):
                gi = g2.create_group(k)
                for ki, vi in v.items():
                    gi[ki] = vi
            else:
                g2[k] = v
        for k, v in res_dict.items():
            if isinstance(v, dict):
                gi = g1.create_group(k)
                for ki, vi in v.items():
                    gi[ki] = vi
            else:
                g1[k] = v
        for k, v in rois_dict.items():
            g3[k] = v


def load_results(path):
    import hdfdict
    from dotted_dict import DottedDict

    with h5.File(path, "r") as f:
        res = DottedDict(hdfdict.load(f, lazy=False))
        params = OmegaConf.create(json.loads(f.attrs["params"]))
    return res, params


def generate_cspline(res_dict, psfobj, keyname="I_model"):
    param = psfobj.param if hasattr(psfobj, "param") else None
    channeltype = psfobj.channeltype if hasattr(psfobj, "channeltype") else "single"
    coeff = []

    if channeltype == "single":
        if keyname in res_dict:
            I_model = res_dict[keyname]
            offset = np.min(I_model)
            Imd = I_model - offset
            normf = np.median(np.sum(Imd, axis=(-1, -2)))
            Imd = Imd / normf
            coeff = psf2cspline_np(Imd).astype(np.float32)
    elif channeltype == "multi":
        if keyname in res_dict.get("channel0", {}):
            Nchannel = len(psfobj.sub_psfs)
            I_model = []
            for i in range(Nchannel):
                I_model.append(res_dict["channel" + str(i)][keyname])
            I_model = np.stack(I_model)
            offset = np.min(I_model)
            Imd = I_model - offset
            normf = np.max(np.median(np.sum(Imd, axis=(-1, -2)), axis=-1))
            Imd = Imd / normf
            Iall = []
            for i in range(Nchannel):
                c = psf2cspline_np(Imd[i])
                Iall.append(c)
            coeff = np.stack(Iall).astype(np.float32)
    elif channeltype == "4pi":
        if keyname in res_dict.get("channel0", {}):
            Nchannel = len(psfobj.sub_psfs)
            I_model = []
            A_model = []
            for i in range(Nchannel):
                I_model.append(res_dict["channel" + str(i)][keyname])
                if keyname == "I_model":
                    A_model.append(res_dict["channel" + str(i)]["A_model"])
                else:
                    A_model.append(res_dict["channel" + str(i)]["A_model_reverse"])
            I_model = np.stack(I_model)
            A_model = np.stack(A_model)
            offset = np.min(I_model - 2 * np.abs(A_model))
            Imd = I_model - offset
            normf = np.max(np.median(np.sum(Imd[:, 1:-1], axis=(-1, -2)), axis=-1)) * 2.0
            Imd = Imd / normf
            Amd = A_model / normf
            IABall = []
            for i in range(Nchannel):
                Ii = Imd[i]
                Ai = 2 * np.real(Amd[i])
                Bi = -2 * np.imag(Amd[i])
                IAB = [psf2cspline_np(Ai), psf2cspline_np(Bi), psf2cspline_np(Ii)]
                IAB = np.stack(IAB)
                IABall.append(IAB)
            coeff = np.stack(IABall).astype(np.float32)

    return coeff
