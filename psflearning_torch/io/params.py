from omegaconf import OmegaConf


def load(path: str):
    return OmegaConf.load(path)


def combine(basefile, psftype=None, channeltype=None, sysfile=None):
    base = OmegaConf.load(f"config/{basefile}.yaml")

    if psftype is not None:
        psf_cfg = OmegaConf.load(f"config/psftype/{psftype}.yaml")
        base = redefine(base, psf_cfg)

    if channeltype is not None:
        ch_cfg = OmegaConf.load(f"config/channeltype/{channeltype}.yaml")
        base = redefine(base, ch_cfg)

    if sysfile is not None:
        sys_cfg = OmegaConf.load(f"config/systemtype/{sysfile}.yaml")
        base = redefine(base, sys_cfg)

    if psftype == "zernike" and channeltype == "4pi":
        base.PSFtype = "zernike"

    if "insitu" in (psftype or ""):
        if hasattr(base, "roi"):
            if base.roi.gauss_sigma[0] < 4 if isinstance(base.roi.gauss_sigma, list) else base.roi.gauss_sigma < 4:
                base.roi.gauss_sigma = [4, 4]
            if base.roi.max_kernel[0] < 5 if isinstance(base.roi.max_kernel, list) else base.roi.max_kernel < 5:
                base.roi.max_kernel = [5, 5]

    if "FD" in (psftype or ""):
        base.option.model.bin = 1

    return base


def redefine(baseparam, userparam):
    for key, value in userparam.items():
        if isinstance(value, dict) and key in baseparam:
            try:
                baseparam[key] = redefine(baseparam[key], value)
            except Exception:
                baseparam[key] = value
        else:
            baseparam[key] = value
    return baseparam
