import unittest

import torch

from torchcv.models import vgg16

expected_size = (2, 512, 7, 7)

class TestVGG16(unittest.TestCase):
    """Test of feature extractore
    """
    def test_vgg16(self):
        net = vgg16(pretrained=True)
        inputs = torch.rand(2, 3, 224, 224)
        outputs = net(inputs)
        self.assertEqual(outputs.size(), expected_size)

if __name__ == '__main__':
    unittest.main()
