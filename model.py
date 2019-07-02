import logging
from collections import OrderedDict

import torch
from torch import nn, optim
from torchvision import models


class FlowerRecognizor():
    def __init__(self, base_model='densenet121', use_gpu=False):
        self.log = logging.getLogger(__class__.__name__)
        self.base_model = base_model
        self.use_gpu = use_gpu
        if not use_gpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")

        self._create_model(base_model)

    def _create_model(self, base_model):
        supported_base_models = {
            'vgg13': models.vgg13,
            'vgg13_bn': models.vgg13_bn,
            'vgg16': models.vgg16,
            'vgg16_bn': models.vgg16_bn,
            'vgg19': models.vgg19,
            'vgg19_bn': models.vgg19_bn,
            'densenet121': models.densenet121,
            'densenet169': models.densenet169
        }
        input_features_dict = {
            'vgg13': 25088,
            'vgg13_bn': 25088,
            'vgg16': 25088,
            'vgg16_bn': 25088,
            'vgg19': 25088,
            'vgg19_bn': 25088,
            'densenet121': 1024,
            'densenet169': 1024
        }
        base_model_function = supported_base_models.get(base_model, None)

        if not base_model_function:
            print("Not a valid base_model. Try: {}".format(
                ','.join(supported_base_models.keys())))

        self.model = base_model_function(pretrained=True)
        input_features = input_features_dict[base_model]

        # Freeze weights of feature extractor.
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.base_model = base_model
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_features, 512)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.05)),
            ('fc3', nn.Linear(512, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

        self.model.classifier = classifier

        self.optimizer = optim.Adam(
            self.model.classifier.parameters(), lr=0.005)

    def _load_checkpoint(self, model_state_dict, optim_state_dict, class_to_idx):
        self.model.load_state_dict(model_state_dict)
        self.model.class_to_idx = class_to_idx
        self.optimizer.load_state_dict(optim_state_dict)

    @staticmethod
    def load_checkpoint(checkpoint_file, use_gpu=False):
        """
        Creates a model from an existing checkpoint files.

        Input:
        - checkpoint_file: filepath to .pth file

        Output:
        - object of FlowerRecognizor with model loaded from checkpoint
        """

        checkpoint = torch.load(checkpoint_file, map_location='cpu')

        fr = FlowerRecognizor(checkpoint["base_model"], use_gpu)

        fr._load_checkpoint(checkpoint['model_state_dict'],
                            checkpoint['optim_state_dict'],
                            checkpoint['class_to_idx'])
        return fr

    def predict(self, image_obj, topk):
        tensor_image = torch.from_numpy(image_obj).type(torch.FloatTensor)
        tensor_image = tensor_image.unsqueeze_(0)

        tensor_image.to(self.device)
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tensor_image)

            probs = torch.exp(outputs)
            top_p, top_class = probs.topk(topk, dim=1)

            top_p = top_p.numpy()[0]
            top_class = top_class.numpy()[0]
        idx_to_class = {val: key for key, val in
                        self.model.class_to_idx.items()}
        top_class = [idx_to_class[i] for i in top_class]
        return top_p, top_class
