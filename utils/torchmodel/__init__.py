from .resnet import ResNet
from .cnn import CNN1, CNN2, CNN3, CNN4
from torchsummary import summary


def create_model(model_type: str, input_channels, output_dim: int):
    model_type = model_type.lower()
    if model_type.startswith('resnet'):
        # model_names_dict = {
        #     'resnet_s': "ResNet_S",
        #     'resnet_m': "ResNet_M",
        #     'resnet_l': "ResNet_L",
        # }
        num_blocks_dict = {
            'resnet_t': (2, 2, 2),
            'resnet_s': (3, 3, 3),
            'resnet_m': (4, 4, 4),
            'resnet_l': (5, 5, 5),
        }
        return ResNet(
            in_channels=input_channels,
            num_classes=output_dim,
            num_blocks=num_blocks_dict[model_type],
            c_hidden=(16, 32, 64),
        )
    elif model_type.startswith('cnn'):
        name_class_dict = {
            'cnn1': CNN1,
            'cnn2': CNN2,
            'cnn3': CNN3,
            'cnn4': CNN4,
        }
        return name_class_dict[model_type](in_channels=input_channels, output_dims=output_dim)
    elif model_type == 'urban':
        from .urban import UrbanNet
        return UrbanNet(in_channels=input_channels, num_classes=output_dim)
    elif model_type == 'tmd':
        from .tmd import TMDNet
        return TMDNet(in_channels=input_channels, num_classes=output_dim)
    else:
        raise ValueError("Model type not implemented")
