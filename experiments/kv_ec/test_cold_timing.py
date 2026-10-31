import unittest

from experiments.kv_ec import cold_timing


class ColdTimingTests(unittest.TestCase):
    def make_client(self, operation_id):
        return dict(operation_id=operation_id, **cold_timing.clock_identity(),
            request_start_ns=140, first_output_token_ns=150,
            full_output_ns=170, response_complete_ns=180)

    def test_build_and_validate_shared_client_boundaries(self):
        record = cold_timing.build_timing('operation-123', 'reprefill',
            fault=100, launch=110, ready=130, client=self.make_client('operation-123'),
            validated=190)
        self.assertEqual(record['fault_boundary_to_first_output_token_ms'], 50 / 1e6)
        self.assertEqual(record['fault_boundary_to_response_complete_ms'], 80 / 1e6)
        self.assertEqual(cold_timing.validate_timing(record, 'operation-123', 'reprefill'), record)

    def test_rejects_cross_operation_and_out_of_order_events(self):
        with self.assertRaisesRegex(ValueError, 'identity'):
            cold_timing.build_timing('operation-123', 'reprefill', 100, 110, 130,
                self.make_client('other-operation'), 190)
        record = cold_timing.build_timing('operation-123', 'reprefill', 100, 110, 130,
            self.make_client('operation-123'), 190)
        record['events_ns']['response_complete'] = 160
        with self.assertRaisesRegex(ValueError, 'event_order'):
            cold_timing.validate_timing(record, 'operation-123', 'reprefill')

    def test_measurement_flags_default_off(self):
        self.assertEqual(cold_timing.measurement_config({}),
            dict(cold_comparison=False, decode_forward_timing=False))
        with self.assertRaisesRegex(ValueError, 'requires_boolean'):
            cold_timing.measurement_config({'measurement': {'decode_forward_timing': 1}})


if __name__ == '__main__':
    unittest.main()
