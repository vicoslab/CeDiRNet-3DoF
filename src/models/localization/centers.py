import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import GaussianLayer

class GradientMultiplyLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask_bw):
        ctx.save_for_backward(mask_bw)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        print('called backward on GradientMultiplyLayer ')
        mask_bw, = ctx.saved_tensors
        return grad_output.mul(mask_bw), None

class Conv1dMultiscaleLocalization(nn.Module):

    def __init__(self, mask_thr=0.05, local_max_thr=0.5, local_max_thr_use_abs=True, exclude_border_px=5,
                 use_conv_sum=False, use_conv_abs=False, learnable=False,
                 allow_input_backprop=True, backprop_only_positive=True,
                 apply_input_smoothing_for_local_max=0, use_findcontours_for_local_max=False, return_time=False,
                 local_max_min_dist=5):
        super(Conv1dMultiscaleLocalization, self).__init__()

        self.mask_thr = mask_thr
        self.local_max_thr = local_max_thr
        self.local_max_thr_use_abs = local_max_thr_use_abs
        self.local_max_min_dist = local_max_min_dist
        self.allow_input_backprop = allow_input_backprop
        self.backprop_only_positive = backprop_only_positive

        self._init_conv_buffers(use_conv_sum, use_conv_abs)

        self.grad_backprop_multiplyer = GradientMultiplyLayer().apply
        self.grad_backprop_multiplyer_val = torch.ones(1)

        self.exclude_border_px = exclude_border_px

        self.gaussian_blur = GaussianLayer(num_channels=1, sigma=int(apply_input_smoothing_for_local_max)) if apply_input_smoothing_for_local_max else None
        self.use_findcontours_for_local_max = use_findcontours_for_local_max

        self.return_time = return_time

    def init_output(self):
        pass

    @staticmethod
    def _generate_kernel(w=5):
        x = (w - 1) // 2
        k = -np.ones((1, w))
        k[0, 0:x] = -k[0, 0:x]
        k[0, x] = 0
        return k / (w - 1)

    def _init_conv_buffers(self, use_conv_sum=False, use_conv_abs=False):
        if use_conv_sum:
            self._conv_merge_fn = lambda x: torch.sum(x, dim=1, keepdim=True)
        else:
            self._conv_merge_fn = lambda x: torch.max(x, dim=1, keepdim=True)[0]

        if use_conv_abs:
            self._conv_abs_fn = lambda x: torch.abs(x)
        else:
            self._conv_abs_fn = lambda x: x

        kernel_sizes = [3, 9, 15, 21, 31, 51, 65]
        self.dilations = [1, ]

        self.max_kernel_size = max(kernel_sizes)

        kernel_weights = [np.pad(self._generate_kernel(i)[0], (self.max_kernel_size - i) // 2) for i in kernel_sizes]
        kernel_weights = np.stack(kernel_weights)

        kernel_cos = torch.tensor(kernel_weights, dtype=torch.float32).reshape(len(kernel_sizes), 1,
                                                                               self.max_kernel_size, 1)
        kernel_sin = torch.tensor(kernel_weights, dtype=torch.float32).reshape(len(kernel_sizes), 1, 1,
                                                                               self.max_kernel_size)

        self.register_buffer("kernel_cos", kernel_cos)
        self.register_buffer("kernel_sin", kernel_sin)

    def _conv_response(self, C, S, R, M, mask):
        conv_resp = []

        for d in self.dilations:
            conv_resp.append(F.conv2d(C, self.kernel_cos, dilation=(d,1), padding=(self.max_kernel_size // 2*d, 0)) + \
                             F.conv2d(S, self.kernel_sin, dilation=(1,d), padding=(0, self.max_kernel_size // 2*d)))

        conv_resp = torch.cat(conv_resp,dim=1)

        conv_resp_intermediate = [V for V in conv_resp[0]]

        # use sum or max
        conv_resp = self._conv_merge_fn(self._conv_abs_fn(conv_resp))

        return conv_resp, conv_resp_intermediate

    def forward(self, C_, S_, R_, M_, mask_, ignore_region=None):

        if self.return_time:
            torch.cuda.synchronize()
            start_preproces = time.time()

        def extend_shape(X):
            if X is not None:
                while len(X.shape) < 4:
                    X = X.unsqueeze(0)
            return X

        # convert to [B x C x H x W] if not already in this shape
        inputs = list(map(extend_shape, [C_, S_, R_, M_, mask_]))

        if not self.allow_input_backprop:
            inputs = list(map(lambda X: X.detach() if X is not None else X, inputs))

        C, S, R, M, mask = inputs

        if self.return_time:
            torch.cuda.synchronize()
            start_model = time.time()

        conv_resp, conv_resp_intermediate = self._conv_response(*inputs)

        if self.return_time:
            torch.cuda.synchronize()
            start_postproc = time.time()

        conv_resp_out = None

        if not self.backprop_only_positive:
            conv_resp_out = conv_resp

        with torch.no_grad():
            # SHOULD NOT use inplace so that returned conv_resp_out values will have negative values for backprop-gradient
            conv_resp = F.relu(conv_resp.clone(), inplace=False)

            if self.backprop_only_positive:
                conv_resp_out = conv_resp


            centers = self._get_local_max_indexes(conv_resp, min_distance=self.local_max_min_dist, threshold_abs=self.local_max_thr,
                                                  exclude_border=self.exclude_border_px,
                                                  input_smooth_op=self.gaussian_blur,
                                                  use_findcontours=self.use_findcontours_for_local_max)

            if ignore_region is not None:
                ignore_region = extend_shape(ignore_region)

                b,x,y = centers[:,0].long(), centers[:,1].long(), centers[:,2].long()
                centers = centers[ignore_region[b,0,y,x] == 0, :]
                #centers = [(b, x, y, c) for (b, x, y, c) in centers if ignore_region[b, 0, y, x] == 0]

            b, x, y, c = centers[:, 0].long(), centers[:, 1].long(), centers[:, 2].long(), centers[:, 3]
            centers_mask = mask[b, 0, y, x] if mask is not None else torch.zeros_like(c)
            res = torch.stack((b, x, y, centers_mask, c),dim=1)

            #res = [(b, x, y, mask[b, 0, y, x].item() if mask is not None else 0, c) for (b, x, y, c) in centers]

            # remove batch dim if input did not have it
            if len(C_.shape) < 4:
                #res = [(x,y,m,c) for (b,x,y,m,c) in res]
                res = res[:,1:]

            if self.return_time:
                torch.cuda.synchronize()
                end = time.time()

                return res, conv_resp_out, (start_model-start_preproces, start_postproc-start_model, end-start_postproc)
            else:

                return res, conv_resp_out

    def _get_local_max_indexes(self, input_ch, min_distance, threshold_rel=None, threshold_abs=0.0, exclude_border=5,
                               input_smooth_op=None, use_findcontours=False):
        """
        Return the indeces containing all peak candidates above thresholds.
        """
        input_ch_blured = input_smooth_op(input_ch) if input_smooth_op is not None else input_ch

        size = 2 * min_distance + 1
        input_max_pool = F.max_pool2d(input_ch_blured,
                                      kernel_size=(size, size),
                                      padding=(size // 2, size // 2), stride=1)

        mask = torch.eq(input_ch_blured, input_max_pool)

        if threshold_rel is not None:
            threshold = max(threshold_abs, threshold_rel * input.max())
        else:
            threshold = threshold_abs
        mask = mask * (input_ch > threshold)

        if exclude_border > 0:
            border_mask = torch.zeros(input_ch.shape[-2:], dtype=mask.dtype, device=mask.device)
            border_mask[exclude_border:-exclude_border, exclude_border:-exclude_border] = 1

            mask *= border_mask

        if use_findcontours:
            list = []
            import cv2
            for b in range(mask.shape[0]):
                for c in range(mask.shape[1]):
                    contours,_ = cv2.findContours(mask[b,c].cpu().type(torch.uint8).numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                    contours = [(int(np.mean(pt[:,0,0]).round()),
                                 int(np.mean(pt[:,0,1]).round())) for pt in contours]
                    list = list + [(b, x, y, input_ch[b, c, y, x].item()) for x,y in contours]
            return torch.from_numpy(np.array(list))
        else:

            #return [(b.item(), x.item(), y.item(), input_ch[b, c, y, x].item()) for (b, c, y, x) in torch.nonzero(mask)]
            ids = torch.nonzero(mask)
            ret = torch.cat((ids[:,[0,3,2]], input_ch[mask].reshape(-1,1)),dim=1)
            return ret
            #return [(b.item(), x.item(), y.item(), input_ch[b, c, y, x].item()) for (b, c, y, x) in ]


class FnGuidedBackpropRelU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = grad_output.clone()

        #grad_input[grad_input < 0] = 0
        #grad_input[input < 0] = 0
        grad_input[input < 0] *= 0.1
        return grad_input

class GuidedBackpropRelU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = FnGuidedBackpropRelU().apply
    def forward(self, input):
        return self.relu(input)

class Conv2dDilatedLocalization(Conv1dMultiscaleLocalization):

    def __init__(self, mask_thr=0.05, local_max_thr=0.5, local_max_thr_use_abs=True, exclude_border_px=5,
                 use_conv_sum=True, use_conv_abs=False, learnable=False, allow_input_backprop=True,
                 backprop_only_positive=True, return_sigmoid=False, inner_ch=16, inner_kernel=5,
                 dilations=[1, 4, 8, 16, 32, 48], min_downsample=np.inf, freeze_learning=False, leaky_relu=False,
                 gradpass_relu=False, use_centerdir_radii=False, use_centerdir_magnitude=True, use_cls_mask=True,
                 **kwargs):
        self.use_centerdir_radii = use_centerdir_radii
        self.use_centerdir_magnitude = use_centerdir_magnitude
        self.use_cls_mask = use_cls_mask
        self.input_ch = 2 + sum([use_centerdir_radii, use_centerdir_magnitude, use_cls_mask])
        self.inner_ch = inner_ch
        self.inner_kernel = inner_kernel
        self.dilations = dilations
        self.min_downsample = min_downsample
        self.freeze_learning = freeze_learning
        self.relu_fn = nn.LeakyReLU if leaky_relu else nn.ReLU
        self.relu_fn = GuidedBackpropRelU if gradpass_relu else self.relu_fn
        self.batch_norm_fn = nn.SyncBatchNorm if torch.distributed.is_initialized() else nn.BatchNorm2d

        super(Conv2dDilatedLocalization, self).__init__(mask_thr, local_max_thr, local_max_thr_use_abs, exclude_border_px,
                                                        use_conv_sum, use_conv_abs, learnable, allow_input_backprop,
                                                        backprop_only_positive, **kwargs)

        self.return_sigmoid = return_sigmoid


    def init_output(self):
        init_convs = []
        for c in self.conv_dilations:
            init_convs.extend(list(c))
        if type(self.conv_start) == torch.nn.modules.Sequential:
            init_convs.extend(list(self.conv_start))
        if type(self.conv_end) == torch.nn.modules.Sequential:
            init_convs.extend(list(self.conv_end))

        init_convs.append(self.conv_merge_fn)

        for c in init_convs:
            if type(c) in [torch.nn.modules.conv.Conv2d, torch.nn.modules.conv.ConvTranspose2d]:
            #if type(c) in [torch.nn.modules.conv.Conv2d]:
                print('initialize center estimator layer with size: ', c.weight.size())

                torch.nn.init.xavier_normal_(c.weight, gain=0.05)# gain=0.05)
                if c.bias is not None:
                    torch.nn.init.zeros_(c.bias)

    def _init_conv_buffers(self, use_conv_sum=False, use_conv_abs=False):
        if use_conv_sum:
            self.conv_merge_fn = lambda x: torch.sum(x, dim=1, keepdim=True)
        else:
            self.conv_merge_fn = lambda x: torch.max(x, dim=1, keepdim=True)[0]

        if use_conv_abs:
            self.conv_abs_fn = lambda x: torch.abs(x)
        else:
            self.conv_abs_fn = lambda x: x

        use_batchnorm=True
        if use_batchnorm is False:
            self.batch_norm_fn = lambda **args: nn.Identity()

        def conv2d_block(in_ch, out_ch, kw, p, d=1, s=1):
            return [nn.Conv2d(in_ch, out_ch, kernel_size=kw, padding=p, dilation=d, stride=s, bias=not use_batchnorm),
                    self.batch_norm_fn(num_features=out_ch, track_running_stats=not self.freeze_learning),
                    self.relu_fn(inplace=True)]

        def deconv2d_block(in_ch, out_ch, kw, p, out_p, s=1):
            return [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kw, padding=p, output_padding=out_p, stride=s, bias=not use_batchnorm),
                    self.batch_norm_fn(num_features=out_ch, track_running_stats=not self.freeze_learning),
                    self.relu_fn(inplace=True)]

        start_nn = nn.Sequential(
            *conv2d_block(self.input_ch, self.inner_ch, kw=self.inner_kernel, p=self.inner_kernel // 2, s=2),
            *conv2d_block(self.inner_ch, self.inner_ch, kw=self.inner_kernel, p=self.inner_kernel // 2, s=2),
            *conv2d_block(self.inner_ch, 2 * self.inner_ch, kw=self.inner_kernel, p=self.inner_kernel // 2, s=2),
            *conv2d_block(2 * self.inner_ch, 2 * self.inner_ch, kw=self.inner_kernel, p=self.inner_kernel // 2, s=2),
        )
        seq_nn = nn.ModuleList([
            nn.Sequential(
                *conv2d_block(2 * self.inner_ch, 2 * self.inner_ch, kw=self.inner_kernel, p=self.inner_kernel // 2 * d, d=d)
            )
            for d in self.dilations
        ])
        end_nn = nn.Sequential(
            *deconv2d_block(len(self.dilations) * 2 * self.inner_ch, 2 * self.inner_ch, kw=5, p=5 // 2, out_p=3 // 2, s=2),
            *deconv2d_block(2 * self.inner_ch, 2 * self.inner_ch, kw=5, p=5 // 2, out_p=3 // 2, s=2),
            *deconv2d_block(2 * self.inner_ch, self.inner_ch, kw=5, p=5 // 2, out_p=3 // 2, s=2),
            *deconv2d_block(self.inner_ch, self.inner_ch, kw=5, p=5 // 2, out_p=3 // 2, s=2),
        )

        self.conv_start = start_nn
        self.conv_dilations = seq_nn
        self.conv_end = end_nn
        self.conv_merge_fn = nn.Conv2d(self.inner_ch, 1, kernel_size=3,  padding=3//2)

        self.upsample = {2**i:nn.Upsample(scale_factor=2**i,mode='bilinear') for i in range(6)}


    def _conv_response(self, C, S, R, M, mask):

        input = [C,S]
        if self.use_centerdir_radii:
            input += [R]
        if self.use_centerdir_magnitude:
            input += [M]
        if self.use_cls_mask:
            input += [mask]

        input = torch.cat(input,dim=1)

        input_min_size = min(input.shape[-2:])
        MIN_DOWNSAMPLE = min(self.min_downsample, input_min_size)

        conv_resp = None

        for i in range(int((np.log(input_min_size) - np.log(MIN_DOWNSAMPLE))/np.log(2))+1):
            n = 2**i
            input_i = input[:,:,::n,::n]
            c1 = self.conv_start(input_i)
            conv_resp_i = [conv_op(c1) for conv_op in self.conv_dilations]

            conv_resp_i = torch.cat(conv_resp_i, dim=1)
            conv_resp_i = self.conv_end(conv_resp_i)

            # upsample image if not in original size
            if n > 1:
                conv_resp_i = self.upsample[n](conv_resp_i)

            # sum
            conv_resp = conv_resp_i if conv_resp is None else (conv_resp + conv_resp_i)

        # use sum or max
        conv_resp = self.conv_merge_fn(self.conv_abs_fn(conv_resp))

        if self.return_sigmoid:
            conv_resp = torch.sigmoid(conv_resp)

        conv_resp_intermediate = []
        return conv_resp, conv_resp_intermediate


