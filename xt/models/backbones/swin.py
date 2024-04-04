import timm
from torch import nn


class SwinWrapper(nn.Module):
    def __init__(self, model, self_supervised=False, hidden_size=None):
        super().__init__()
        self.model = model
        self.feature_info = list(model.feature_info)
        self.self_supervised = self_supervised
        self.hidden_size = hidden_size

        if self.self_supervised:
            assert self.hidden_size is not None
            self.attn_project = nn.Linear(in_features=self.hidden_size * 8 * 8, out_features=512)
            self.attn_predict = nn.Linear(in_features=512, out_features=8 * 8 * self.hidden_size)

    def forward(self, x):
        intermediates = self.model(x)
        intermediates = list([x.permute(0, 3, 1, 2) for x in intermediates])

        if self.self_supervised:
            B, _, wsize, _ = intermediates[-1].shape
            projected_attn = self.attn_project(intermediates[-1].reshape(B, self.hidden_size * (wsize ** 2)))
            predicted_attn = self.attn_predict(projected_attn)
            predicted_attn = predicted_attn.view(B, wsize ** 2, self.hidden_size)
            return intermediates, predicted_attn
        else:
            return intermediates


class SwinXviewWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.feature_info = list(model.feature_info)

        embed_dim = model.layers_0.dim
        for i in self.feature_info:
            i["index"] = i["index"] + 1

        self.feature_info = [
            dict(num_chs=embed_dim, reduction=2, module="patch_embed", index=0)
        ] + self.feature_info
        self.upsample = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)

    def forward(self, x):
        intermediates = self.model(x)
        intermediates = [self.model.patch_embed(x)] + intermediates
        intermediates = list([x.permute(0, 3, 1, 2) for x in intermediates])
        intermediates[0] = self.upsample(intermediates[0])
        return intermediates


def swinv2_tiny_window16_256_timm(*args, **kwargs):
    input_size = (kwargs["input_size"], kwargs["input_size"])
    self_supervised = kwargs["self_supervised"]
    hidden_size = kwargs["hidden_size"]

    model = timm.create_model(
        "swinv2_tiny_window16_256.ms_in1k",
        features_only=True,
        pretrained=True,
        input_size=input_size,
    )
    print(f"Model is being trained with self_supervised={self_supervised}")
    return SwinWrapper(model, self_supervised=self_supervised, hidden_size=hidden_size)


def swinv2_small_window16_256_timm(*args, **kwargs):
    input_size = (kwargs["input_size"], kwargs["input_size"])
    model = timm.create_model(
        "swinv2_small_window16_256.ms_in1k",
        features_only=True,
        pretrained=True,
        input_size=input_size,
    )
    return SwinWrapper(model)


def swinv2_base_window16_256_timm(*args, **kwargs):
    input_size = (kwargs["input_size"], kwargs["input_size"])
    model = timm.create_model(
        "swinv2_base_window16_256.ms_in1k",
        features_only=True,
        pretrained=True,
        input_size=input_size,
    )
    return SwinWrapper(model)


def swinv2_large_window16_256_timm(*args, **kwargs):
    input_size = (kwargs["input_size"], kwargs["input_size"])
    model = timm.create_model(
        "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k",
        features_only=True,
        pretrained=True,
        input_size=input_size,
    )
    return SwinWrapper(model)


def swinv2_tiny_window16_256_timm_xview(input_size, *args, **kwargs):
    opts = {"input_size": (2, input_size, input_size)}
    model = timm.create_model(
        "swinv2_tiny_window16_256.ms_in1k",
        features_only=True,
        pretrained=True,
        in_chans=2,
        pretrained_cfg_overlay=opts,
    )
    return SwinXviewWrapper(model)


def swinv2_small_window16_256_timm_xview(input_size, *args, **kwargs):
    opts = {"input_size": (2, input_size, input_size)}
    model = timm.create_model(
        "swinv2_small_window16_256.ms_in1k",
        features_only=True,
        pretrained=True,
        in_chans=2,
        pretrained_cfg_overlay=opts,
    )
    return SwinXviewWrapper(model)


def swinv2_base_window16_256_timm_xview(input_size, *args, **kwargs):
    opts = {"input_size": (2, input_size, input_size)}
    model = timm.create_model(
        "swinv2_base_window16_256.ms_in1k",
        features_only=True,
        pretrained=True,
        in_chans=2,
        pretrained_cfg_overlay=opts,
    )
    return SwinXviewWrapper(model)


def swinv2_large_window16_256_timm_xview(input_size, *args, **kwargs):
    opts = {"input_size": (2, input_size, input_size)}
    model = timm.create_model(
        "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k",
        features_only=True,
        pretrained=True,
        in_chans=2,
        pretrained_cfg_overlay=opts,
    )
    return SwinXviewWrapper(model)
