from .mobilenetv1 import MobileNetV1
from .swin_transformer import swin_transformer_tiny, swin_transformer_base, swin_transformer_small
from .resnet import ResNet50

get_model_from_name = {
    "mobilenetv1"   : MobileNetV1,
    "resnet50"      : ResNet50,
    "swin_transformer_tiny"     : swin_transformer_tiny,
    "swin_transformer_small"    : swin_transformer_small,
    "swin_transformer_base"     : swin_transformer_base
}

freeze_layers = {
    "mobilenetv1"   : 81,
    "resnet50"      : 173,
    "swin_transformer_tiny"     : 181,
    "swin_transformer_small"    : 350,
    "swin_transformer_base"     : 350
}
