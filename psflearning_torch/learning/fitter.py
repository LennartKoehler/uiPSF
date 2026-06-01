from __future__ import annotations

import time
import numpy as np
import torch
from tqdm import tqdm

from .loss import mse_real_zernike, mse_zernike_4pi


class Fitter:
    def __init__(self, data, psf, loss_func, loss_func_single=None, loss_weight=None, maxiter=200, batch_size=1600):
        self.data = data
        self.psf = psf
        self.loss_func = loss_func
        self.loss_func_single = loss_func_single
        self.loss_weight = loss_weight
        self.maxiter = maxiter
        self.batch_size = batch_size
        self.rois = None
        self.forward_images = None
        self.loc_FD = None
        self.mu = 1.0
        self.rate = 1.1

    def objective(self, variables, mu, ind=None):
        if ind is None:
            ind = [0, variables[0].shape[0]]
        self.psf.batch_indices = ind
        forward_images = self.psf.calc_forward_images(variables)
        w = self.loss_weight if self.loss_weight is not None else list(self.psf.weight)

        if self.loss_func_single:
            rois_slice = self.rois[:, ind[0]:ind[1]] if self.rois.ndim > 4 else self.rois[ind[0]:ind[1]]
            loss = self.loss_func(forward_images, rois_slice, self.loss_func_single, variables, mu, w)
        else:
            rois_slice = self.rois[ind[0]:ind[1]]
            loss = self.loss_func(forward_images, rois_slice, variables, mu, w)
        return loss

    def learn_psf(self, variables=None, start_time=None):
        if variables is None:
            variables, start_time = self.psf.calc_initials(self.data, start_time=start_time)

        _, rois, _, _ = self.data.get_image_data()
        try:
            self.rois = np.stack(rois)
        except ValueError:
            raise RuntimeError("Each channel must have same number of rois.")

        variables = self._optimize_lbfgs(variables, start_time)
        toc = self._last_toc
        ind = [0, variables[0].shape[0]]
        self.psf.batch_indices = ind
        self.forward_images = self.psf.calc_forward_images(variables)
        if isinstance(self.forward_images, torch.Tensor):
            self.forward_images = self.forward_images.detach().numpy()
        variables = self.psf.postprocess(variables)
        return variables, toc

    def _optimize_lbfgs(self, variables, start_time=None):
        Nfit = variables[0].shape[0]
        batch_size = min(self.batch_size, Nfit)
        varinfo = self.psf.variable_info
        shapes = [v.shape for v in variables]
        lengths = [np.prod(s) for s in shapes]
        dtypes = [v.dtype for v in variables]
        total_len = sum(lengths)

        pbar = tqdm(
            total=self.maxiter + 50,
            desc="3/6: learning",
            bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}s] {rate_fmt}, {postfix[0]}{postfix[2][loss]:>4.5f}, {postfix[1]}{postfix[2][time]:>4.2f}s",
            postfix=["current loss: ", "total time: ", dict(loss=0, time=start_time or 0)],
        )

        self.mu = 1.0
        self._last_toc = start_time or 0

        def flatten_vars(var_list):
            parts = []
            for v in var_list:
                t = torch.from_numpy(v.astype(np.float32)).reshape(-1)
                parts.append(t)
            return torch.cat(parts)

        def unflatten_vars(flat):
            var_list = []
            idx = 0
            for shape, length, dtype in zip(shapes, lengths, dtypes):
                var_list.append(flat[idx:idx+length].reshape(shape).detach().numpy().astype(dtype))
                idx += length
            return var_list

        x = flatten_vars(variables)
        x.requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [x], max_iter=1, line_search_fn="strong_wolfe"
        )

        step = [0]
        for outer_step in range(self.maxiter):
            ind_list = list(np.int32(np.linspace(0, Nfit, Nfit // batch_size + 2)))

            def closure():
                optimizer.zero_grad()
                var_flat = x.detach().clone()
                var_flat.requires_grad_(True)

                variables_local = unflatten_vars(var_flat)

                total_loss = torch.tensor(0.0)
                total_grad = torch.zeros_like(var_flat)

                for bi in range(len(ind_list) - 1):
                    var_batch = [None] * len(variables_local)
                    for k in range(len(variables_local)):
                        if varinfo[k]["type"] == "Nfit":
                            if varinfo[k]["id"] == 0:
                                var_batch[k] = variables_local[k][ind_list[bi]:ind_list[bi+1]]
                            elif varinfo[k]["id"] == 1:
                                var_batch[k] = variables_local[k][:, ind_list[bi]:ind_list[bi+1]]
                        else:
                            var_batch[k] = variables_local[k]

                    var_batch_t = []
                    for v in var_batch:
                        t = torch.from_numpy(v.astype(np.float32))
                        t.requires_grad_(True)
                        var_batch_t.append(t)

                    loss1 = self.objective(var_batch_t, self.mu, ind_list[bi:bi+2])
                    w1 = var_batch_t[0].shape[0] / Nfit
                    total_loss = total_loss + loss1 * w1

                    grads = torch.autograd.grad(loss1, var_batch_t, create_graph=False)
                    g_flat_parts = []
                    idx = 0
                    for k, (shape, length) in enumerate(zip(shapes, lengths)):
                        g = grads[k]
                        if g is None:
                            g = torch.zeros_like(var_batch_t[k])
                        if varinfo[k]["type"] == "Nfit":
                            if varinfo[k]["id"] == 0:
                                pad = torch.zeros(length - g.numel(), dtype=g.dtype)
                                full_g = torch.cat([g.reshape(-1), pad])
                            elif varinfo[k]["id"] == 1:
                                full_g_tensor = torch.zeros(shape, dtype=g.dtype)
                                full_g_tensor[:, ind_list[bi]:ind_list[bi+1]] = g
                                full_g = full_g_tensor.reshape(-1)
                            else:
                                full_g = g.reshape(-1)
                        else:
                            full_g = g.reshape(-1) * w1
                        g_flat_parts.append(full_g)
                        idx += length

                    batch_grad = torch.cat(g_flat_parts)
                    total_grad = total_grad + batch_grad

                x_grad = total_grad
                if x.grad is not None:
                    x.grad = x_grad
                else:
                    x.grad = x_grad

                return total_loss

            try:
                loss_val = optimizer.step(closure)
            except Exception as e:
                print(f"Optimization step failed: {e}")
                break

            self.mu *= self.rate
            self.mu = min(self.mu, 1e7)

            step[0] += 1
            current_loss = float(x.grad.norm()) if x.grad is not None else 0
            try:
                current_loss = loss_val.item() if isinstance(loss_val, torch.Tensor) else float(loss_val)
            except Exception:
                pass

            pbar.postfix[-1]["loss"] = current_loss
            pbar.postfix[-1]["time"] = (start_time or 0) + time.time() - pbar.start_t
            pbar.update(1)
            self._last_toc = pbar.postfix[-1]["time"]

        pbar.close()
        return unflatten_vars(x.detach())

    def relearn(self, initres, channeltype, threshold, start_time=None):
        metric = self.reject_metric
        mask = metric[0] > -1
        for i, val in enumerate(metric):
            mask = (val < threshold[i]) & mask
        mask = (self.minI > 0) & mask
        delete_id = np.where(~mask)
        print("outlier id:", str(delete_id[0]))

        if (delete_id[0].size > 0) and (delete_id[0].size < mask.size):
            if channeltype == "single":
                _, rois, centers, file_idxs = self.data.get_image_data()
                cor = centers[mask, :]
                fid = file_idxs[mask]
                self.data.rois = rois[mask]
                self.data.centers = cor
                self.data.file_idxs = fid
                var = initres[-1]
                var[0] = initres[-1][0][mask]
                var[1] = initres[-1][1][mask]
                var[2] = initres[-1][2][mask]
                var[-1] = initres[-1][-1][mask]
                res, toc = self.learn_psf(var, start_time=start_time)
            else:
                _, rois, centers, file_idxs = self.data.get_image_data()
                for i in range(len(self.data.channels)):
                    self.data.channels[i].rois = rois[i][mask]
                    self.data.channels[i].centers = centers[i][mask, :]
                    self.data.channels[i].file_idxs = file_idxs[i][mask]
                var = initres[-1]
                var[0] = initres[-1][0][mask]
                var[1] = initres[-1][1][:, mask]
                var[2] = initres[-1][2][:, mask]
                var[-2] = initres[-1][-2][:, mask]
                if channeltype == "4pi":
                    var[3] = initres[-1][3][:, mask]
                res, toc = self.learn_psf(var, start_time=start_time)
        else:
            res = initres
            toc = start_time
        return res, toc

    def localize(self, res, channeltype, usecuda=True, initz=None, plot=True, start_time=None):
        from .localization import LocalizationLib

        intensity = np.abs(np.squeeze(res[2], axis=(-1, -2)))
        if res[2].dtype == "complex64":
            intensityR = intensity
        else:
            intensityR = np.real(np.squeeze(res[2], axis=(-1, -2)))
        I_model = res[3]
        psf_data = self.rois
        pz = self.data.pixelsize_z

        dll = LocalizationLib(usecuda=usecuda)
        if channeltype == "single":
            locres = dll.loc_ast(psf_data, I_model, pz, initz=initz, plot=plot, start_time=start_time)
            mydiff = self.forward_images[:, 1:-1] - psf_data[:, 1:-1]
            mse1 = np.mean(np.square(mydiff), axis=(-3, -2, -1)) / np.mean(psf_data, axis=(-3, -2, -1))
        elif channeltype == "multi":
            _, _, centers, _ = self.data.get_image_data()
            cor = np.stack(centers)[..., -2:]
            imgcenter = self.psf.imgcenter
            T = res[-2]
            locres = dll.loc_ast_dual(psf_data, I_model, pz, cor, imgcenter, T, initz=initz, plot=plot, start_time=start_time)
            mydiff = self.forward_images[:, :, 1:-1] - psf_data[:, :, 1:-1]
            mse1 = np.mean(
                np.mean(np.square(mydiff), axis=(-3, -2, -1)) / np.mean(psf_data, axis=(-3, -2, -1)),
                axis=0,
            )
        elif channeltype == "4pi":
            _, _, centers, _ = self.data.get_image_data()
            A_model = res[4]
            cor = np.stack(centers)
            imgcenter = self.psf.imgcenter
            T = np.squeeze(res[-2])
            zT = np.array([self.psf.sub_psfs[0].zT])
            locres = dll.loc_4pi(
                psf_data, I_model, A_model, pz, cor, imgcenter, T, zT, initz=initz, plot=plot, start_time=start_time
            )
            mydiff = self.forward_images[:, :, :, 1:-1] - psf_data[:, :, :, 1:-1]
            mse1 = np.mean(
                np.mean(np.square(mydiff), axis=(-4, -3, -2, -1)) / np.mean(psf_data, axis=(-4, -3, -2, -1)),
                axis=0,
            )
        else:
            raise TypeError("supported channeltype is: single, multi, 4pi")

        if channeltype == "single":
            if len(intensity.shape) < 2:
                avgI = intensity
                minI = intensityR
            else:
                avgI = np.median(intensity, axis=1)
                minI = np.min(intensityR, axis=1)
        else:
            if len(intensity.shape) < 3:
                avgI = intensity[0]
                minI = intensityR[0]
            else:
                avgI = np.median(intensity[0], axis=1)
                minI = np.min(intensityR[0], axis=1)

        if psf_data.shape[0] == 1:
            intRatio = np.array([1.0])
            mseRatio = np.array([1.0])
        else:
            intRatio = np.square(avgI - np.median(avgI)) / np.median(avgI) / avgI
            mseRatio = mse1 / np.median(mse1)
        msezRatio = locres[4]
        metric = [msezRatio, mseRatio, intRatio]
        self.reject_metric = metric
        self.minI = minI
        return locres
