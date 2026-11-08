from copy import deepcopy
import json,subprocess,unittest
from pathlib import Path
from unittest.mock import patch
from . import admission
ROOT=Path(__file__).parent

class LocalIdentityTests(unittest.TestCase):
    """Mocked read-only address evidence; these tests never start services."""

    def setUp(self):
        self.config = json.loads((ROOT / "placement.json").read_text())
        self.interfaces = [dict(ifindex=2, ifname="enp161s0f1", flags=["BROADCAST", "UP", "LOWER_UP"],
                                operstate="UP", addr_info=[dict(family="inet", local="10.102.0.243", prefixlen=26)])]
        probe_patch = patch.object(admission.subprocess, "run")
        self.probe = probe_patch.start()
        self.addCleanup(probe_patch.stop)
        self.set_evidence(self.interfaces)

    def set_evidence(self, interfaces):
        self.probe.return_value = subprocess.CompletedProcess(
            ["ip", "-j", "address"], 0, stdout=json.dumps(interfaces))

    def reject(self, interfaces):
        self.set_evidence(interfaces)
        with self.assertRaisesRegex(ValueError, "local physical identity/address not verified"):
            admission.local_identity(self.config, "P", "243")

    def test_physical243_with_hostname_dns_mismatch(self):
        with patch("socket.gethostname", return_value="sdu-NF5468M6"), patch(
                "socket.getaddrinfo", return_value=[(2, 1, 6, "", ("127.0.1.1", 0))]) as dns:
            self.assertIs(admission.local_identity(self.config, "P", "243"), self.config["nodes"][2])
        dns.assert_not_called()
        self.probe.assert_called_once_with(["ip", "-j", "address"], check=True,
                                           capture_output=True, text=True, timeout=5)

    def test_dns_failure_does_not_override_local_interface(self):
        with patch("socket.getaddrinfo", side_effect=OSError("DNS unavailable")):
            self.assertEqual(admission.local_identity(self.config, "P", "243")["physical_id"], "243")

    def test_each_role_requires_its_own_address(self):
        for node in self.config["nodes"]:
            with self.subTest(role=node["role"]):
                interfaces = deepcopy(self.interfaces)
                interfaces[0]["addr_info"][0]["local"] = node["host"]
                self.set_evidence(interfaces)
                self.assertIs(admission.local_identity(self.config, node["role"], node["physical_id"]), node)

    def test_missing_or_wrong_explicit_id_and_unknown_role(self):
        with self.assertRaises(TypeError):
            admission.local_identity(self.config, "P")
        for physical_id in (None, "", "232", "233", "243 ", 243, True):
            with self.subTest(physical_id=physical_id), self.assertRaises(ValueError):
                admission.local_identity(self.config, "P", physical_id)
        for role in (None, "", "unknown"):
            with self.subTest(role=role), self.assertRaises(ValueError):
                admission.local_identity(self.config, role, "243")
        self.probe.assert_not_called()

    def test_placement_validation_cannot_be_bypassed_by_local_evidence(self):
        self.config["nodes"][2]["host"] = "192.168.2.243"
        with self.assertRaisesRegex(ValueError, "physical address mismatch"):
            admission.local_identity(self.config, "P", "243")
        self.probe.assert_not_called()

    def test_dns_alone_and_same_subnet_or_alias_are_not_local_proof(self):
        for address in ("127.0.0.1", "10.102.0.232", "10.102.0.242", "192.168.2.243", "192.168.2.245"):
            with self.subTest(address=address), patch(
                    "socket.getaddrinfo", return_value=[(2, 1, 6, "", ("10.102.0.243", 0))]):
                interfaces = deepcopy(self.interfaces)
                interfaces[0]["addr_info"][0]["local"] = address
                self.reject(interfaces)

    def test_empty_or_missing_evidence(self):
        for interfaces in ([], None, {}, [None], [{}]):
            with self.subTest(interfaces=interfaces):
                self.reject(interfaces)
        interfaces = deepcopy(self.interfaces)
        interfaces[0]["addr_info"] = []
        self.reject(interfaces)
        for key in self.interfaces[0]:
            interfaces = deepcopy(self.interfaces)
            del interfaces[0][key]
            with self.subTest(missing=key):
                self.reject(interfaces)

    def test_down_or_unusable_addresses_rejected(self):
        for key, value in (("flags", []), ("operstate", "DOWN"), ("operstate", "DORMANT")):
            interfaces = deepcopy(self.interfaces)
            interfaces[0][key] = value
            with self.subTest(key=key, value=value):
                self.reject(interfaces)
        for key in ("tentative", "dadfailed"):
            interfaces = deepcopy(self.interfaces)
            interfaces[0]["addr_info"][0][key] = True
            with self.subTest(key=key):
                self.reject(interfaces)

    def test_ambiguous_placement_addresses_rejected_even_if_down(self):
        for other_host in ("10.102.0.232", "10.102.0.233", "10.102.0.243"):
            for state in ("UP", "DOWN"):
                interfaces = deepcopy(self.interfaces)
                other = dict(interfaces[0], ifindex=3, ifname="other", operstate=state,
                             addr_info=[dict(family="inet", local=other_host, prefixlen=26)])
                with self.subTest(other_host=other_host, state=state):
                    self.reject(interfaces + [other])
        interfaces = deepcopy(self.interfaces)
        interfaces[0]["addr_info"] *= 2
        self.reject(interfaces)
        interfaces[0]["addr_info"][1] = dict(family="inet", local="10.102.0.232", prefixlen=26)
        self.reject(interfaces)

    def test_duplicate_interface_identifiers_rejected(self):
        for overrides in ({}, {"ifname": "other"}, {"ifindex": 3}):
            with self.subTest(overrides=overrides):
                self.reject(self.interfaces + [dict(self.interfaces[0], addr_info=[], **overrides)])

    def test_malformed_interface_fields_rejected(self):
        for key, value in (("ifindex", True), ("ifindex", "2"), ("ifindex", 0),
                           ("ifname", ""), ("ifname", 2), ("flags", "UP"), ("flags", [True]),
                           ("operstate", ""), ("operstate", True), ("addr_info", {})):
            interfaces = deepcopy(self.interfaces)
            interfaces[0][key] = value
            with self.subTest(key=key, value=value):
                self.reject(interfaces)

    def test_malformed_address_fields_rejected(self):
        for key, value in (("family", "inet6"), ("family", "unknown"), ("local", "10.102.0.243/26"),
                           ("local", "10.102.0.999"), ("local", None), ("prefixlen", "26"),
                           ("prefixlen", True), ("prefixlen", -1), ("prefixlen", 33),
                           ("tentative", "false"), ("dadfailed", 0)):
            interfaces = deepcopy(self.interfaces)
            interfaces[0]["addr_info"][0][key] = value
            with self.subTest(key=key, value=value):
                self.reject(interfaces)
        for key in ("family", "local", "prefixlen"):
            interfaces = deepcopy(self.interfaces)
            del interfaces[0]["addr_info"][0][key]
            with self.subTest(missing=key):
                self.reject(interfaces)
        interfaces[0]["addr_info"] = [None]
        self.reject(interfaces)

    def test_normal_loopback_ipv6_and_unrelated_addresses_are_allowed(self):
        interfaces = deepcopy(self.interfaces)
        interfaces[0]["addr_info"].append(dict(family="inet6", local="fe80::1234", prefixlen=64))
        interfaces.append(dict(ifindex=1, ifname="lo", flags=["LOOPBACK", "UP"], operstate="UNKNOWN",
                               addr_info=[dict(family="inet", local="127.0.0.1", prefixlen=8)]))
        self.set_evidence(interfaces)
        self.assertEqual(admission.local_identity(self.config, "P", "243")["host"], "10.102.0.243")

    def test_invalid_json_duplicate_keys_and_nonfinite_constants_rejected(self):
        valid = json.dumps(self.interfaces)
        for stdout in ("", "not json", valid + " trailing", valid.replace('"ifindex": 2', '"ifindex": 1, "ifindex": 2'),
                       valid.replace('"local": "10.102.0.243"', '"local": "10.102.0.232", "local": "10.102.0.243"'),
                       valid.replace('"ifindex": 2', '"extra": NaN, "ifindex": 2')):
            self.probe.return_value.stdout = stdout
            with self.subTest(stdout=stdout), self.assertRaises(ValueError):
                admission.local_identity(self.config, "P", "243")

    def test_command_failures_fail_closed(self):
        for error in (FileNotFoundError("ip missing"), PermissionError("netlink denied"),
                      subprocess.CalledProcessError(1, ["ip", "-j", "address"]),
                      subprocess.TimeoutExpired(["ip", "-j", "address"], 5)):
            self.probe.side_effect = error
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, "interface evidence unavailable/invalid"):
                admission.local_identity(self.config, "P", "243")
