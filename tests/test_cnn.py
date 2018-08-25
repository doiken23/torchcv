import unittest

import torch

from torchcv.models import vgg16, Pix2PixGenerator, Pix2PixDiscriminator

expected_size = (2, 512, 7, 7)

class TestVGG16(unittest.TestCase):
    """Test of feature extractore
    """
    def test_vgg16(self):
        net = vgg16(pretrained=True)
        inputs = torch.rand(2, 3, 224, 224)
        outputs = net(inputs)
        self.assertEqual(outputs.size(), expected_size)

class TestPix2Pix(unittest.TestCase):
    """Test of pix2pix
    """
    def test_generator(self):
        G = Pix2PixGenerator(3, 3)
        inputs = torch.rand(2, 3, 256, 256)
        outputs = G(inputs)
        self.assertEqual(inputs.size(), outputs.size())

        D = Pix2PixDiscriminator(3, 3)
        score = D(inputs, outputs)
        self.assertEqual(score.size(), (2, 1))

if __name__ == '__main__':
    unittest.main()
