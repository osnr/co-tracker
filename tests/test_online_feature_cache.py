# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import unittest

import torch

from cotracker.models.core.cotracker.cotracker3_online import CoTrackerThreeOnline


class CountingEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, frames):
        self.batch_sizes.append(frames.shape[0])
        values = frames[:, :1]
        return torch.cat([values, torch.ones_like(values)], dim=1)


class TestOnlineFeatureCache(unittest.TestCase):
    def setUp(self):
        self.model = CoTrackerThreeOnline.__new__(CoTrackerThreeOnline)
        torch.nn.Module.__init__(self.model)
        self.model.window_len = 4
        self.model.stride = 1
        self.model.latent_dim = 2
        self.model.corr_levels = 2
        self.model.fnet = CountingEncoder()
        self.model.init_video_online_processing()

    def test_only_encodes_new_frame_after_full_window(self):
        first_video = torch.arange(4.0).reshape(1, 4, 1, 1, 1).expand(-1, -1, -1, 4, 4)
        first = self.model._get_fmaps_pyramid(
            first_video, is_online=True, step=1
        )

        second_video = torch.arange(1.0, 5.0).reshape(1, 4, 1, 1, 1)
        second_video = second_video.expand(-1, -1, -1, 4, 4)
        second = self.model._get_fmaps_pyramid(
            second_video, is_online=True, step=1
        )

        self.assertEqual(self.model.fnet.batch_sizes, [4, 1])
        for first_level, second_level in zip(first, second):
            torch.testing.assert_close(second_level[:, :-1], first_level[:, 1:])

    def test_reset_discards_cached_features(self):
        video = torch.zeros(1, 4, 1, 4, 4)
        self.model._get_fmaps_pyramid(video, is_online=True, step=1)
        self.model.init_video_online_processing()
        self.model._get_fmaps_pyramid(video, is_online=True, step=1)

        self.assertEqual(self.model.fnet.batch_sizes, [4, 4])


if __name__ == "__main__":
    unittest.main()
