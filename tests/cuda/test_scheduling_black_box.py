"""Device-resident scheduled scalars with actual CUDA graph replay."""

import unittest

import torch

from deepopt import CUDAGraphCore
from deepopt.scheduling import ScheduleController, LinearWarmup


@unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware is required")
class CUDASchedulingTests(unittest.TestCase):
    def test_coefficients_update_in_place_across_graph_replay(self):
        device = torch.device("cuda", torch.cuda.current_device())
        schedules = ScheduleController({"beta": LinearWarmup(.1, 1., 3)})
        beta = schedules.tensor("beta", device=device)
        pointer = beta.data_ptr()
        graph = CUDAGraphCore(device).capture(lambda x: x * beta)
        x = torch.ones(3, device=device)
        for update in range(1, 4):
            schedules.prepare_update(update)
            torch.testing.assert_close(graph(x), x * schedules["beta"])
            self.assertEqual(beta.data_ptr(), pointer)
            schedules.complete_update(update)
        graph.close()


if __name__ == "__main__":
    unittest.main()
