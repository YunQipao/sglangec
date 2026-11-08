import io
import json
import unittest
from unittest.mock import patch

from .cold_timing import stream_request


class StreamTests(unittest.TestCase):
    def response(self, counts, prefix=256, corrupt=False):
        lines = []
        for count in counts:
            ids = list(range(count))
            if corrupt and count == 17:
                ids[0] = 99
            lines.append(b'data: ' + json.dumps(dict(output_ids=ids,
                meta_info=dict(completion_tokens=count, prompt_tokens=prefix))).encode() + b'\n\n')
        return io.BytesIO(b''.join(lines) + b'data: [DONE]\n\n')

    def test_multi_token_events_and_pending_token_are_one_request(self):
        with patch('urllib.request.urlopen', return_value=self.response([1, 4, 17])) as send:
            result = stream_request('http://localhost', [3]*256, 'original-rid', 'trial-id')
        self.assertEqual(send.call_count, 1)
        payload = json.loads(send.call_args.args[0].data)
        self.assertTrue(payload['stream'])
        self.assertEqual(payload['rid'], 'original-rid')
        self.assertEqual(result['output_token_ids'], list(range(17)))
        self.assertLessEqual(result['first_output_token_ns'], result['full_output_ns'])
        self.assertLessEqual(result['full_output_ns'], result['response_complete_ns'])

    def test_long_prompt_boundary(self):
        with patch('urllib.request.urlopen', return_value=self.response([3, 17], 4096)):
            self.assertEqual(len(stream_request('http://localhost', [3]*4096, 'r', 'o')['output_token_ids']), 17)

    def test_missing_final_and_rewritten_ids_rejected(self):
        with patch('urllib.request.urlopen', return_value=self.response([1, 8])):
            with self.assertRaisesRegex(ValueError, 'missing_first_or_full'):
                stream_request('http://localhost', [3]*256, 'r', 'o')
        with patch('urllib.request.urlopen', return_value=self.response([1, 17], corrupt=True)):
            with self.assertRaisesRegex(ValueError, 'monotonic_cumulative'):
                stream_request('http://localhost', [3]*256, 'r', 'o')

    def test_wrong_prompt_count_rejected(self):
        with patch('urllib.request.urlopen', return_value=self.response([17], 512)):
            with self.assertRaisesRegex(ValueError, 'boundary_invalid'):
                stream_request('http://localhost', [3]*256, 'r', 'o')
