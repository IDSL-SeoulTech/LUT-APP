import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json

ACT_FUNCS = {
    'gelu': nn.GELU(),
    'silu': nn.SiLU(),
    'exp': torch.exp,
    'reci': lambda x: 1 / x,
    'rsqrt': lambda x: 1 / torch.sqrt(x)
}

def decompose_float_rsqrt(value):
    abs_value = torch.abs(value)
    exponent = torch.floor(torch.log2(abs_value + (abs_value == 0).float()))
    mantissa = abs_value / (2 ** exponent)
    is_even = (exponent % 2 == 0).float().to(value.device)
    is_odd = torch.where(is_even == 1, torch.tensor(0.0, dtype=is_even.dtype).to(value.device), torch.tensor(1.0, dtype=is_even.dtype).to(value.device))
    exp_rsqrt_even = 2 ** (-0.5 * exponent)
    exp_rsqrt_odd = 2 ** (-0.5 * (exponent - 1))
    mantissa_even = mantissa
    mantissa_odd = 2 * mantissa
    mantissa = is_even * mantissa_even + is_odd * mantissa_odd
    exponent = is_even * exp_rsqrt_even + is_odd * exp_rsqrt_odd
    return mantissa, exponent

def decompose_float_reci(value):
    abs_value = torch.abs(value)
    exponent = torch.floor(torch.log2(abs_value + (abs_value == 0).float()))
    mantissa = abs_value / (2 ** exponent)
    exp_rsqrt = 2 ** (-1 * exponent)
    return mantissa, exp_rsqrt

class gade_lut_pwl(nn.Module):
    def __init__(self, pwl_type, pwl_dir, dff=True) -> None:
        """
        :param pwl_type: the required non-linear function to approximate
        :param pwl_dir: pretrained dir
        """
        super(gade_lut_pwl, self).__init__()
        self.dff = dff
        with open(pwl_dir, 'r') as f:
            params = json.load(f)
        self.pwl_params = params[pwl_type]

        self.breakpoints = torch.tensor(self.pwl_params['breakpoints'])
        self.breakpoint_fip = torch.round(self.breakpoints * 2**(4)).clamp(-128, 127)
        self.slope_dff = torch.tensor(self.pwl_params['slope_dff'])
        self.intercept_dff = torch.tensor(self.pwl_params['intercept_dff'])

        if self.dff == True:
            self.slope_scale = torch.floor((1 + torch.log2(self.slope_dff.abs())).clamp(0, 7))
            self.intercept_scale = torch.floor((1 + torch.log2(self.intercept_dff.abs())).clamp(0, 7))
        else:
            self.slope_scale = 2
            self.intercept_scale = 2

        self.slope_value = torch.round(self.slope_dff * 2**(7 - self.slope_scale)).clamp(-128, 127)
        self.intercept_value = torch.round(self.intercept_dff * 2**(7 - self.intercept_scale)).clamp(-128, 127)

    def forward(self, x):
        if self.dff == True:
            x_scale = torch.floor((1 + torch.log2(x.abs())).clamp(0, 7))
        else:
            x_scale = 4
        x_value = torch.round(x * 2**(7 - x_scale)).clamp(-128, 127)

        x_scale_comp = x_scale.clamp(0, 3) # makes comparator happy
        # In DFF comparator, x_value should be interpreted as Q0.7
        x_fip = torch.floor((x_value / 2**7) * 2**(x_scale_comp + 4)).clamp(-128, 127)

        satur_cond = x_scale_comp > 3
        pos_satur = (torch.sign(x_value) == 1) & satur_cond
        neg_satur = (torch.sign(x_value) == -1) & satur_cond

        indices = torch.bucketize(x_fip, self.breakpoint_fip, right=True)
        indices[pos_satur] = len(self.slope_value) - 1
        indices[neg_satur] = 0

        slope_scale_idxd = self.slope_scale[indices]
        slope_value_idxd = self.slope_value[indices]

        intercept_scale_idxd = self.intercept_scale[indices]
        intercept_value_idxd = self.intercept_value[indices]

        mult_scale = x_scale + slope_scale_idxd
        mult_value = x_value * slope_value_idxd

        scale_sub = (7 + intercept_scale_idxd) - mult_scale

        add_value = mult_value + (intercept_value_idxd * 2**scale_sub)
        add_scale = mult_scale

        out = add_value / 2**(14 - mult_scale)
        return out
        
def config_parser():
    parser = argparse.ArgumentParser(description='LUT-APP GADE Operator Validation')
    parser.add_argument("--act_func", type=str, default='exp', help="Activation function name")
    parser.add_argument("--distance", type=float, default=0.01, help="Distnace between samples")
    parser.add_argument("--input_range", nargs='+', type=float, default=[-9.0, 0.0], help="Input range")
    parser.add_argument("--dynamic", action='store_true', default=True, help="Activate Dynamic Fixed-point Format(DFF)")
    parser.add_argument("--param_path", type=str, default='./gade_pwl_param/exp/entry_8/exp_7_dff_True_best_0.001806344.json', help="Activation function name")
    return parser

if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()

    # Sample Generation
    lower_bound = args.input_range[0]
    upper_bound = args.input_range[1]
    step = args.distance
    steps = (upper_bound - lower_bound) / step
    num_samples = int(steps) + 1
    fp32_test_input = torch.linspace(lower_bound, upper_bound, num_samples)

    # FP32 Torch Ground Truth Generation
    fp32_func = ACT_FUNCS[args.act_func]
    fp32_groundtruth = fp32_func(fp32_test_input)

    # GADE Output Generation
    gade_module = gade_lut_pwl(pwl_type=args.act_func, pwl_dir=args.param_path, dff=args.dynamic)

    if args.act_func == 'reci':
        norm_input, exp_scale = decompose_float_reci(fp32_test_input)
        fp32_approx = gade_module(norm_input) * exp_scale
    elif args.act_func == 'rsqrt':
        norm_input, exp_scale = decompose_float_rsqrt(fp32_test_input)
        fp32_approx = gade_module(norm_input) * exp_scale
    else:
        fp32_approx = gade_module(fp32_test_input)
        if args.act_func == 'exp':
            fp32_shift_mask = fp32_test_input <= -5.5625
            fp32_approx[fp32_shift_mask] /= 2**5

    # Measure MSE & MAE
    mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()
    mse = mse_loss(fp32_groundtruth, fp32_approx)
    mae = mae_loss(fp32_groundtruth, fp32_approx)
    print(f"[FUNC: {args.act_func}, RANGE: {args.input_range}] MSE: {mse:.8e}, MAE: {mae:.8e}")