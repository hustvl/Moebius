# Moebius student
from .nets.unet_lambda_prune_lite import UNet2DLambdaDWConvMixFFNConditionModel_prune_down_mid_up_block_8x8
from .nets.unet_lambda_dwconv_blocks import *  # block factories for student

# PixelHacker teacher
from .nets.unet_gla import UNet2DGLAConditionModel
