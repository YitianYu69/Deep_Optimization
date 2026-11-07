"""Capture configuration shared by policy tests and real GPU tests."""


def graph_config(**options):
    return {"name": "capture-defaults", "seed": 17,
            "training": {"epochs": 1, "batch_size": 4, "learning_rate": 0.01},
            "runtime": {"device": "cpu", "cuda_graph": options}}
