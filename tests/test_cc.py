import torch
import cc_torch
import unittest


class TestCC(unittest.TestCase):
    def test_2d(self):
        img_2d = torch.tensor([
            1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0,
            1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
            1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0,
            1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0], dtype=torch.uint8).reshape(12, 8).cuda()

        expected_output = torch.tensor(
            [[1,  1,  0,  1,  1,  1,  1,  1],
             [0,  1,  1,  0,  1,  1,  1,  0],
             [1,  1,  1,  0,  1,  1,  1,  0],
             [1,  1,  0,  0,  0,  0,  0,  0],
             [0,  1,  1,  0,  1,  0,  0, 39],
             [0,  0,  0,  1,  0,  0, 39,  0],
             [0,  0,  0,  0,  0,  0,  0,  0],
             [0,  0,  0,  0,  0, 53,  0,  0],
             [0,  0,  0,  0,  0, 53,  0,  0],
             [0, 65,  0, 53, 53, 53, 53, 53],
             [0, 65,  0,  0, 53, 53, 53,  0],
             [0, 65,  0,  0, 53, 53, 53,  0]], dtype=torch.int32).cuda()

        output = cc_torch.connected_components_labeling(img_2d)
        self.assertTrue((output == expected_output).all())

    def test_3d(self):
        img_2d = torch.tensor([
            1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0,
            1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
            1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0,
            1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0], dtype=torch.uint8).reshape(12, 8)
        img_3d = img_2d[None].repeat(4, 1, 1).cuda()

        expected_output = torch.tensor(
            [[1,  1,  0,  1,  1,  1,  1,  1],
             [0,  1,  1,  0,  1,  1,  1,  0],
             [1,  1,  1,  0,  1,  1,  1,  0],
             [1,  1,  0,  0,  0,  0,  0,  0],
             [0,  1,  1,  0,  1,  0,  0, 39],
             [0,  0,  0,  1,  0,  0, 39,  0],
             [0,  0,  0,  0,  0,  0,  0,  0],
             [0,  0,  0,  0,  0, 53,  0,  0],
             [0,  0,  0,  0,  0, 53,  0,  0],
             [0, 65,  0, 53, 53, 53, 53, 53],
             [0, 65,  0,  0, 53, 53, 53,  0],
             [0, 65,  0,  0, 53, 53, 53,  0]], dtype=torch.int32)
        expected_output = expected_output[None].repeat(4, 1, 1).cuda()

        output = cc_torch.connected_components_labeling(img_3d)
        self.assertTrue((output == expected_output).all())
