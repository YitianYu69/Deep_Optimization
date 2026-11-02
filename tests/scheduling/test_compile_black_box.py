"""Count actual Dynamo compilations as coefficient tensor values change."""

import unittest

import torch

from deepopt.scheduling import ScheduleController, LinearWarmup


class ScheduleCompilationTests(unittest.TestCase):
    def test_stable_scalar_does_not_recompile_and_matches_reference(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        graphs = []
        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward
        function = torch.compile(lambda x, beta: (x * beta).square().sum(), backend=backend, fullgraph=True)
        schedules = ScheduleController({"beta": LinearWarmup(.1, 1., 20)})
        beta = schedules.tensor("beta")
        for update in range(1, 21):
            schedules.prepare_update(update)
            x = torch.ones(3, requires_grad=True)
            actual = function(x, beta)
            expected = (x * schedules["beta"]).square().sum()
            torch.testing.assert_close(actual, expected)
            actual.backward()
            torch.testing.assert_close(x.grad, torch.full_like(x, 2 * schedules["beta"] ** 2))
            schedules.complete_update(update)
        self.assertEqual(len(graphs), 1)


if __name__ == "__main__":
    unittest.main()
