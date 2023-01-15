import os
import os.path as osp

from mmcv.runner import Hook
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

class EvalHook(Hook):
    """Evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self, dataloader, interval=1, by_epoch=False, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got '
                            f'{type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs

    def after_train_iter(self, runner):
        """After train epoch hook."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)
            imgs = iter(self.dataloader)
            output_dir = osp.join('output_imgs', runner.timestamp, f'iter_{"%05i" % runner.iter}')
            os.makedirs(output_dir, exist_ok=True)
            norm_cfg = self.dataloader.dataset.img_norm_cfg
            norm_cfg_means, norm_cfg_stds = None, None
            for idx, pair in enumerate(zip(imgs, results)):
                img, result = pair
                result = torch.tensor(result)
                output_fn = osp.join(output_dir, f'{"%05i" % idx}.png')
                input_img_torch = img['img'][0][0].permute((1, 2, 0))
                
                # denormalize
                if None in {norm_cfg_means, norm_cfg_stds}:    
                    norm_cfg_means = torch.tensor(norm_cfg['mean']).view(1, 1, 3).repeat((*input_img_torch.shape[:2], 1))
                    norm_cfg_stds = torch.tensor(norm_cfg['std']).view(1, 1, 3).repeat((*input_img_torch.shape[:2], 1))
                
                input_img_torch = input_img_torch * norm_cfg_stds + norm_cfg_means
                input_img_numpy = input_img_torch.cpu().numpy()
                result_downsampled = F.interpolate(result.float().unsqueeze(0).unsqueeze(0),
                                                   size=input_img_torch.shape[:2], mode='nearest').squeeze()
                runner.model.module.show_result(input_img_numpy, result_downsampled.unsqueeze(0),
                                                palette=self.dataloader.dataset.PALETTE,
                                                out_file=output_fn)

    def after_train_epoch(self, runner):
        """After train epoch hook."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        """Call evaluate function of dataset."""
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True


class DistEvalHook(EvalHook):
    """Distributed evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 by_epoch=False,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(
                'dataloader must be a pytorch DataLoader, but got {}'.format(
                    type(dataloader)))
        self.dataloader = dataloader
        self.interval = interval
        self.gpu_collect = gpu_collect
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs

    def after_train_iter(self, runner):
        """After train epoch hook."""
        if not self.every_n_iters(runner, self.interval): # self.by_epoch or ...
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
            
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)
            imgs = iter(self.dataloader)
            output_dir = osp.join('output_imgs', runner.timestamp, f'iter_{"%05i" % runner.iter}')
            os.makedirs(output_dir, exist_ok=True)
            norm_cfg = self.dataloader.dataset.img_norm_cfg
            norm_cfg_means, norm_cfg_stds = None, None
            for idx, pair in enumerate(zip(imgs, results)):
                img, result = pair
                result = torch.tensor(result)
                output_fn = osp.join(output_dir, f'{"%05i" % idx}.png')
                input_img_torch = img['img'][0][0].permute((1, 2, 0))
                
                # denormalize
                if None in {norm_cfg_means, norm_cfg_stds}:    
                    norm_cfg_means = torch.tensor(norm_cfg['mean']).view(1, 1, 3).repeat((*input_img_torch.shape[:2], 1))
                    norm_cfg_stds = torch.tensor(norm_cfg['std']).view(1, 1, 3).repeat((*input_img_torch.shape[:2], 1))
                
                input_img_torch = input_img_torch * norm_cfg_stds + norm_cfg_means
                input_img_numpy = input_img_torch.cpu().numpy()
                result_downsampled = F.interpolate(result.float().unsqueeze(0).unsqueeze(0),
                                                   size=input_img_torch.shape[:2], mode='nearest').squeeze()
                runner.model.module.show_result(input_img_numpy, result_downsampled.unsqueeze(0),
                                                palette=self.dataloader.dataset.PALETTE,
                                                out_file=output_fn)

    def after_train_epoch(self, runner):
        """After train epoch hook."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)
