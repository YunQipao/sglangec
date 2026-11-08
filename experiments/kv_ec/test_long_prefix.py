import copy
import unittest

from .cold_resume_state import validate_state
from .gpu_kv_restore import RestoreRejected
from .gpu_resume_hook import token_digest, validate_decode_observation
from .test_gpu_cold_resume import ColdStateProtocolTests


class LongPrefixTests(unittest.TestCase):
    def test_all_supported_lengths_bind_state_and_pending_position(self):
        for length in (256, 512, 1024, 2048, 4096):
            with self.subTest(prefix=length):
                fixture = ColdStateProtocolTests()
                fixture.setUp()
                state, meta = copy.deepcopy(fixture.state), copy.deepcopy(fixture.meta)
                state.update(origin_input_ids=[11]*length, token_count=length,
                    materialized_kv_tokens=length, kv_allocated_len=length,
                    next_decode_position=length, next_decode_seq_len=length+1,
                    input_ids_sha256=token_digest([11]*length))
                for rank in meta['ranks']:
                    rank.update(token_count=length, token_ids_sha256=state['input_ids_sha256'])
                self.assertEqual(validate_state(state, meta, 'kv-hash'), state)
                damaged = dict(state, next_decode_position=length-1)
                with self.assertRaisesRegex(RestoreRejected, 'pending_token_or_kv_length'):
                    validate_state(damaged, meta, 'kv-hash')
                for step in (1, 3, 16):
                    prefix = list(range(1, length+1))
                    mapping = list(range(1, length+step+1))
                    validate_decode_observation(step=step, outputs=[7]*step,
                        seq_len=length+step, committed_len=length+step,
                        input_ids=[7], positions=[length+step-1], indices=mapping,
                        mapping=mapping, prefix_slots=prefix, out_slots=[mapping[-1]])

    def test_unsupported_lengths_fail_closed(self):
        fixture = ColdStateProtocolTests()
        fixture.setUp()
        for length in (0, 255, 513, 8192, True):
            with self.subTest(prefix=length), self.assertRaisesRegex(RestoreRejected, 'prefix_not_supported'):
                validate_state(dict(fixture.state, token_count=length), fixture.meta, 'kv-hash')

class BootstrapAdmissionTests(unittest.TestCase):
    def hook(self,length=1024):
        from types import SimpleNamespace as NS
        from .gpu_cold_resume_hook import GPUColdResumeHook
        h=GPUColdResumeHook.__new__(GPUColdResumeHook)
        h.cfg=dict(bootstrap_prefix=True,template_only_not_runtime_restore_source=True,
            template_prefix_tokens=256,prefix_tokens=length,input_ids=[3]*length)
        h.cold_mode='capture';h.mode='reference';h.rank=0;h.tp_size=2
        h.meta={'ranks':[dict(token_count=256,token_ids_sha256=token_digest([9]*256))]}
        h.scheduler=NS();h.req=NS(origin_input_ids=[3]*length)
        h.prefix_tokens=length;h.request_token_digest=token_digest([3]*length)
        return h
    def test_bootstrap_checks_original_template_range_without_rewriting(self):
        from unittest.mock import patch
        for length in (256,512,1024,2048,4096):
            with self.subTest(prefix=length):
                h=self.hook(length);original=copy.deepcopy(h.meta)
                with patch(__package__+'.gpu_kv_restore._metadata_identity') as check:
                    h._prefill_metadata_identity()
                    self.assertEqual(check.call_args.kwargs['token_count'],256)
                    self.assertEqual(h.meta,original)
    def test_restore_or_unsealed_bootstrap_never_bypasses_identity(self):
        from unittest.mock import patch
        for key,value in [('cold_mode','restore'),('mode','resume'),('request_token_digest','bad')]:
            h=self.hook();setattr(h,key,value)
            with self.subTest(key=key),patch(__package__+'.gpu_kv_restore._metadata_identity') as check:
                with self.assertRaisesRegex(RestoreRejected,'bootstrap_capture_identity_mismatch'):h._prefill_metadata_identity()
                check.assert_not_called()
        h=self.hook();h.cfg['template_only_not_runtime_restore_source']=False
        with self.assertRaisesRegex(RestoreRejected,'bootstrap_capture_identity_mismatch'):h._prefill_metadata_identity()
    def test_nonbootstrap_still_checks_exact_new_prefix(self):
        from unittest.mock import patch
        h=self.hook();h.cfg={}
        with patch(__package__+'.gpu_resume_hook._metadata_identity') as check:
            h._prefill_metadata_identity()
            self.assertEqual(check.call_args.kwargs['token_count'],1024)
