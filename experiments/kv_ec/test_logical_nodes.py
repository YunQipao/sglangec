import unittest

try:
    from .logical_nodes import rank_layout, rank_manifest, verify_rank_observations
except ImportError:
    from logical_nodes import rank_layout, rank_manifest, verify_rank_observations


class LogicalNodeLayoutTest(unittest.TestCase):
    def test_requested_two_node_global_tp4_mapping(self):
        layout = rank_layout()
        self.assertEqual(layout[0].global_tp_ranks, (0, 1))
        self.assertEqual(layout[1].global_tp_ranks, (2, 3))
        self.assertEqual(layout[0].visible_devices, (0, 1))
        self.assertEqual(layout[1].visible_devices, (2, 3))
        self.assertEqual(rank_manifest(run_id="test", model_path="model")["global_tp_size"], 4)

    def test_rejects_partial_or_duplicate_gpu_allocation(self):
        with self.assertRaises(ValueError):
            rank_layout(devices=(0, 1, 2, 2))
        with self.assertRaises(ValueError):
            rank_layout(tp_size=2)

    def test_runtime_rank_vote_requires_every_tp_rank_on_expected_node(self):
        obs = [dict(node_rank=rank // 2, local_rank=rank % 2,
                    tp_rank=rank, tp_size=4) for rank in range(4)]
        self.assertEqual(verify_rank_observations(obs)["status"], "rank_map_consistent")
        obs[-1] = dict(obs[-1], node_rank=0)
        with self.assertRaisesRegex(ValueError, "rank_to_node"):
            verify_rank_observations(obs)


if __name__ == "__main__":
    unittest.main()
