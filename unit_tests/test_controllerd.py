#!/usr/bin/env python3

"""
Copyright (c) 2017, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from datetime import datetime
import unittest
import shutil
import socket
import time
import os

import requests
import pygit2

from utils import send_msg, recv_msg, GitRepoFactory, tmp_folder, \
                  ezbench_dir, timings_db_dir, unit_tests_dir
import controllerd_pb2
import controllerd

def url(url):
    return "http://127.0.0.1:8081" + url

class ControllerdTestClient:
    def __init__(self, machine_name):
        self.name = machine_name
        self.controller_name = None
        self.socket = None

    def __del__(self):
        self.close()

    def send_msg(self, msg):
        send_msg(self.socket, msg)

    def recv_msg(self, msg):
        return recv_msg(self.socket, msg)

    def connect(self, sendHello=True):
        self.close()

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.socket.connect(("127.0.0.1", controllerd.Configuration.dut_bind[1]))
        except socket.timeout:
            return False

        # Send the hello message
        if sendHello:
            hello = controllerd_pb2.ClientHello()
            hello.version = 1
            hello.machine_name = self.name
            self.send_msg(hello)

            hello = self.recv_msg(controllerd_pb2.ServerHello())
            if hello is not None:
                self.controller_name = hello.controller_name
            else:
                return False

        return True

    def gen_msg_reports(self, reports):
        msg_sig = controllerd_pb2.Signal()
        msg_avail_reports = msg_sig.reports

        msg_avail_reports.report_count = len(reports)
        for report in reports:
            msg_report = msg_avail_reports.reports.add()
            for key in report:
                setattr(msg_report, key, report[key])

        return msg_sig

    def gen_msg_tests(self, tests):
        msg_sig = controllerd_pb2.Signal()
        msg_avail_tests = msg_sig.tests

        msg_avail_tests.tests_count = len(tests)
        for test in tests:
            msg_test = msg_avail_tests.tests.add()
            for key in test:
                setattr(msg_test, key, test[key])

        return msg_sig

    def gen_msg_testsets(self, testsets):
        msg_sig = controllerd_pb2.Signal()
        msg_avail_testsets = msg_sig.testsets

        msg_avail_testsets.testsets_count = len(testsets)
        for testset in testsets:
            msg_testset = msg_avail_testsets.testsets.add()
            for key in testset:
                if key == "tests":
                    for test in testset["tests"]:
                        msg_test = msg_testset.tests.add()
                        msg_test.name = test['name']
                        msg_test.rounds = test['rounds']
                else:
                    setattr(msg_testset, key, testset[key])

        return msg_sig

    def close(self):
        try:
            self.socket.close()
        except:
            pass

class Controllerd(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

        # Generate a configuration file
        tmp_folder = ezbench_dir + "/unit_tests/tmp"
        controllerd.Configuration.name = "UnitTest"
        controllerd.Configuration.state_file = tmp_folder + "/controllerd.state"
        controllerd.Configuration.state_file_lock = tmp_folder + "/controllerd.lock"
        controllerd.Configuration.rest_bind = ("127.0.0.1", 8081)
        controllerd.Configuration.dut_bind = ("127.0.0.1", 42001)
        controllerd.Configuration.reports_base_path = tmp_folder + "/reports/"
        controllerd.Configuration.reports_base_url = tmp_folder + "/reports/"
        controllerd.Configuration.src_repo_path = tmp_folder + "/dut-repo"

        # remove any left-over tmp folder, before we re-create it
        try:
            shutil.rmtree(tmp_folder)
        except OSError as e:
            if os.path.exists(tmp_folder):
                msg = "Fatal error: cannot delete the report folder '{}': {}"
                raise ValueError(msg.format(tmp_folder, e)) from e

        # Create the temporary directory
        os.mkdir(tmp_folder)

        dut_repo = GitRepoFactory(controllerd.Configuration.src_repo_path)
        self.src_repo_commit1 = dut_repo.create_commit("Commit 1",
                                                       [("README",
                                                         "# Commit 1\n\nNothing to see!\n")])
        self.src_repo_commit2 = dut_repo.create_commit("Commit 2",
                                                       [("README",
                                                         "# Commit 2\n\nNothing to see!\n")],
                                                       [self.src_repo_commit1])
        self.src_repo_commit3 = dut_repo.create_commit("Commit 3",
                                                       [("README",
                                                         "# Commit 3\n\nNothing to see!\n")],
                                                       [self.src_repo_commit2])
        self.src_repo_commit1_tag = "commit1"
        dut_repo.create_tag(self.src_repo_commit1_tag, self.src_repo_commit1)
        self.src_repo_commit2_tag = "commit2"
        dut_repo.create_tag(self.src_repo_commit2_tag, self.src_repo_commit2)

        # Now, start the server
        self.tcpserver, self.restserver = controllerd.start_controllerd(quiet=True)

    def tearDown(self):
        self.tcpserver.close()
        self.restserver.close()

    def machine_recv_sig_count(self, machine):
        r = requests.get(url("/machines/{}".format(machine)))
        self.assertEqual(r.status_code, 200)
        state = r.json()
        self.assertIn('recv_sig_count', state)
        return state['recv_sig_count']

    def wait_until(self, timeout, predicate, msg=None, wait_between_polls=0.001):
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if not predicate():
                time.sleep(wait_between_polls)
            else:
                return

        if msg is not None:
            msg = "Timeout {}s: {}".format(timeout, msg)
        self.assertTrue(predicate(), msg=msg)

    def send_sig(self, machine, sig, timeout=0.5):
        # Get the current signal count, so we can detect if the signal has been
        # processed
        recv_sig_count = self.machine_recv_sig_count(machine.name)

        machine.send_msg(sig)

        # Wait for the sig count to increase
        self.wait_until(timeout,
                        lambda: self.machine_recv_sig_count(machine.name) >= recv_sig_count + 1,
                        "The signal was not acknowledged by the controller")

    def command_sent(self, cmd_url):
        r = requests.get(url(cmd_url))
        self.assertEqual(r.status_code, 200)
        return r.json()['last_sent'] > 0

    def command_acknowledged(self, cmd_url):
        r = requests.get(url(cmd_url))
        self.assertEqual(r.status_code, 200)
        return r.json()['acknowledged'] > 0

    def read_cmd(self, machine):
        r = requests.get(url("/machines/{}".format(machine.name)))
        self.assertEqual(r.status_code, 200)
        state = r.json()

        # Make sure the machine is online
        self.assertTrue(state['online'])

        # Look for a command that has not been acknowledged
        found = False
        for queued_cmd in state['queued_cmds']:
            self.assertIsInstance(queued_cmd['acknowledged'], float)
            if queued_cmd['acknowledged']:
                self.assertNotEqual(queued_cmd['err_code'], "NON_ACK")
                continue
            else:
                found = True
                # We found a command we need to ACK, wait for it to be sent
                # before waiting for it to detect
                self.wait_until(0.5, lambda: self.command_sent(queued_cmd['url']),
                                "The command was not sent by the controller")
                break

        if not found:
            return None

        # The REST API says we have a message, verify that the timings are OK
        # then check it out
        r = requests.get(url(queued_cmd['url']))
        state = r.json()
        self.assertEqual(r.status_code, 200)
        self.assertLessEqual(datetime.utcnow().timestamp() - state['last_sent'], 0.5)
        self.assertEqual(state['err_code'], "NON_ACK")
        self.assertNotIn("err_msg", state)

        # Read the actual command and verify that the ID matches
        cmd = machine.recv_msg(controllerd_pb2.Cmd())
        self.assertIsNot(cmd, None)
        self.assertEqual(queued_cmd['id'], cmd.id)

        # Send the acknowledgment immediately since we won't do anything with it
        sig = controllerd_pb2.Signal()
        sig.cmd_status.id = cmd.id
        sig.cmd_status.err_msg = "Debug test"
        sig.cmd_status.err_code = controllerd_pb2.CmdStatus.OK
        self.send_sig(machine, sig)

        # Wait for the controller to receive our ACK
        self.wait_until(0.5, lambda: self.command_acknowledged(queued_cmd['url']),
                            "The command did not get acknowledged by the controller")

        # Check that the acknowledgment date matches and that all the other fields
        # match what we sent
        r = requests.get(url(queued_cmd['url']))
        state = r.json()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(state['id'], cmd.id)
        self.assertIsInstance(state['description'], str)
        self.assertEqual(state['err_code'], "OK")
        self.assertEqual(state['err_msg'], sig.cmd_status.err_msg)
        self.assertLessEqual(datetime.utcnow().timestamp() - r.json()['acknowledged'], 0.5)

        return cmd

    def test_machine(self):
        # Verify that no machines are connected
        with self.subTest("No machines"):
            r = requests.get(url("/machines"))
            self.assertEqual(r.status_code, 200)
            state = r.json()
            self.assertEqual(len(state['machines']), 0)

        # Add an invalid machine and verify that no machines are connected
        with self.subTest("Client connected with no Hello message"):
            client = ControllerdTestClient("InvalidNoHelloClient")
            self.assertTrue(client.connect(sendHello=False))
            r = requests.get(url("/machines"))
            self.assertEqual(len(r.json()['machines']), 0)
            client.close()

        # Add one machine
        with self.subTest("Client connected"):
            machine1 = ControllerdTestClient("machine1")
            machine1.connect()
            self.assertEqual(machine1.controller_name, controllerd.Configuration.name)
            r = requests.get(url("/machines"))
            state = r.json()
            self.assertEqual(len(state['machines']), 1)
            self.assertTrue(state['machines']["machine1"]['online'])
            self.assertLessEqual(datetime.utcnow().timestamp() - state['machines']["machine1"]['last_seen'], 1.5)
            self.assertEqual(state['machines']["machine1"]['pending_commands'], 0)

        # Try adding another machine with the same name as the previous machine
        with self.subTest("Duplicate client connects"):
            machine1_bis = ControllerdTestClient("machine1")
            self.assertFalse(machine1_bis.connect())
            self.assertEqual(machine1_bis.controller_name, None)
            machine1_bis.close()
            machine1.close()

        # Add another machine and verify that the online state is valid for both
        with self.subTest("Two clients, one connected and one disconnected"):
            machine2 = ControllerdTestClient("machine2")
            machine2.connect()
            r = requests.get(url("/machines"))
            state = r.json()
            self.assertEqual(len(state['machines']), 2)
            self.assertFalse(state['machines']["machine1"]['online'])
            self.assertTrue(state['machines']["machine2"]['online'])
            machine2.close()

        with self.subTest("Check the default state of machine1"):
            machine1_url = url(state['machines']['machine1']['url'])
            r = requests.get(machine1_url)
            self.assertEqual(r.status_code, 200)
            state = r.json()
            self.assertEqual(state['name'], "machine1")
            self.assertIs(state['online'], False)
            self.assertGreaterEqual(state['last_seen'], 0)
            self.assertGreaterEqual(state['ping'], 0)
            self.assertEqual(len(state['reports']), 0)
            self.assertEqual(len(state['tests']), 0)
            self.assertEqual(len(state['testsets']), 0)

            # There should be one ping command in the queued commands
            self.assertEqual(len(state['queued_cmds']), 1)

        # Add reports and verify that they show up properly
        with self.subTest("Add reports"):
            reports = [{
                "name": "report1",
                "profile": "test_profile",
                "state": "RUNNING",
                "state_disk": "RUN",
                "build_time": 42.12,
                "deploy_time": 13.37
            },
            {
                "name": "report2",
                "profile": "test_profile",
                "state": "RUNNING",
                "state_disk": "RUN",
                "build_time": 42.12,
                "deploy_time": 13.37
            }]

            machine1.connect()
            self.send_sig(machine1, machine1.gen_msg_reports(reports))
            machine1.close()

        with self.subTest("Check that the reports got added properly"):
            r = requests.get(machine1_url)
            self.assertEqual(r.status_code, 200)
            state = r.json()
            self.assertEqual(len(state['reports']), 2)
            for report in state['reports']:
                self.assertEqual(state['reports'][report]["state"], "RUNNING")

                r = requests.get(url(state['reports'][report]["url"]))
                self.assertEqual(r.status_code, 200)
                state_report = r.json()

                self.assertEqual(state_report['name'], report)
                self.assertEqual(state_report['profile'], "test_profile")
                self.assertEqual(state_report['state'], "RUNNING")
                self.assertEqual(state_report['state_disk'], "RUN")
                self.assertAlmostEqual(state_report['build_time'], 42.12, places=4)
                self.assertAlmostEqual(state_report['deploy_time'], 13.37, places=4)

        # Add tests and verify that they show up properly
        with self.subTest("Add tests"):
            tests = [{
                "name": "test1",
                "exec_time_max": 42.12,
                "exec_time_median": 13.37
            },
            {
                "name": "test2",
                "exec_time_max": 42.12,
                "exec_time_median": 13.37
            },
            {
                "name": "test3",
                "exec_time_max": 42.12,
                "exec_time_median": 13.37
            }]

            machine1.connect()
            self.send_sig(machine1, machine1.gen_msg_tests(tests))
            machine1.close()

        with self.subTest("Check that the tests got added properly"):
            r = requests.get(machine1_url)
            self.assertEqual(r.status_code, 200)
            state = r.json()
            self.assertEqual(len(state['tests']), 3)
            for test in state['tests']:
                r = requests.get(url(state['tests'][test]["url"]))
                self.assertEqual(r.status_code, 200)
                state_test = r.json()

                self.assertEqual(state_test['name'], test)
                self.assertEqual(state_test['machine'], "machine1")
                self.assertAlmostEqual(state_test['exec_time_max'], 42.12, places=4)
                self.assertAlmostEqual(state_test['exec_time_median'], 13.37, places=4)

        # Add testsets and verify that they show up properly
        with self.subTest("Add testsets"):
            testsets = [{
                    "name": "testset_empty",
                    "description":  "Testset description",
                    "tests": []
                },
                {
                    "name": "testset_normal",
                    "description":  "Testset description",
                    "tests": [{
                            "name": "test1",
                            "rounds": 4
                        },
                        {
                            "name": "test3",
                            "rounds": 2
                        }]
                }]

            machine1.connect()
            self.send_sig(machine1, machine1.gen_msg_testsets(testsets))
            machine1.close()

        with self.subTest("Check that the testsets got added properly"):
            r = requests.get(machine1_url)
            self.assertEqual(r.status_code, 200)
            state = r.json()
            self.assertEqual(len(state['testsets']), 2)
            for testset in state['testsets']:
                r = requests.get(url(state['testsets'][testset]["url"]))
                self.assertEqual(r.status_code, 200)
                state_testset = r.json()

                self.assertEqual(state_testset['name'], testset)
                self.assertEqual(state_testset['description'], "Testset description")
                self.assertEqual(state_testset['machine'], "machine1")
                self.assertIsInstance(state_testset['tests'], dict)

                if state_testset['name'] == "testset_empty":
                    self.assertEqual(len(state_testset['tests']), 0)
                else:
                    self.assertEqual(len(state_testset['tests']), 2)
                    self.assertEqual(state_testset['tests']['test1'], 4)
                    self.assertEqual(state_testset['tests']['test3'], 2)

        # TODO: Test the queued_cmds

    def report_repo_fullpath(self, machine):
        return unit_tests_dir + "/tmp/reports/controllerd_test/{}".format(machine)

    def check_report_repo(self, machine, should_exist):
        path = self.report_repo_fullpath(machine)

        if should_exist:
            # Make sure the path is a valid git repo and the master branch only
            # has one commit.
            repo = pygit2.Repository(path)
            branch = repo.lookup_branch("master")
            self.assertEqual(repo[branch.target].parents, [])
        else:
            self.assertFalse(os.path.exists(path))

    def test_jobs(self):
        # Add 3 machines
        for i in range(1, 4):
            machine = ControllerdTestClient("machine{}".format(i))
            machine.connect()
            machine.close()

        job_url = url("/jobs/controllerd_test")
        with self.subTest("Delete a non-exisiting job"):
            r = requests.delete(job_url)
            self.assertEqual(r.status_code, 404)

        # Invalid minimum profile
        with self.subTest("Create job: empty"):
            r = requests.post(job_url, json={})
            self.assertEqual(r.status_code, 400)

        with self.subTest("Create job: profile missing"):
            invalid_job = { "description": "Test job" }
            r = requests.post(job_url, json=invalid_job)
            self.assertEqual(r.status_code, 400)

        with self.subTest("Create job: description missing"):
            invalid_job = { "profile": "myprofile" }
            r = requests.post(job_url, json=invalid_job)
            self.assertEqual(r.status_code, 400)

        with self.subTest("Create job: invalid profile type"):
            invalid_job = { "description": 45, "profile": "myprofile" }
            r = requests.post(job_url, json=invalid_job)
            self.assertEqual(r.status_code, 400)

        with self.subTest("Create job: invalid profile type"):
            invalid_job = { "description": "Test job", "profile": None }
            r = requests.post(job_url, json=invalid_job)
            self.assertEqual(r.status_code, 400)

        with self.subTest("Create job: invalid machine name"):
            invalid_job = { "description": "Test job", "profile": "myprofile",
                           "machines": ["machine4"] }
            r = requests.post(job_url, json=invalid_job)
            self.assertEqual(r.status_code, 400)

        with self.subTest("Create job: minimal config"):
            invalid_job = { "description": "Test job", "profile": "myprofile" }
            r = requests.post(job_url, json=invalid_job)
            self.assertEqual(r.status_code, 201)

        with self.subTest("Create job: overwrite job"):
            invalid_job = { "description": "Test job", "profile": "myprofile" }
            r = requests.post(job_url, json=invalid_job)
            self.assertEqual(r.status_code, 400)

        with self.subTest("Delete the test job"):
            r = requests.delete(job_url)
            self.assertEqual(r.status_code, 200)

        # Create the resource
        payload = { "description": "Test job", "profile": "myprofile",
                    "attributes": { "event_min_confidence": 0.995,
                                     "schedule_max_commits": 20,
                                     "perf_min_change": 0.01,
                                     "variance_max": 0.05,
                                     "variance_max_run_count": 10,
                                     "variance_min_run_count": 3,
                                     "report_priority": 42,
                                     "report_deadline_soft":  123456,
                                     "report_deadline_hard": 1234567
                                  },
                    "machines": ["machine1", "machine2"]
                  }
        with self.subTest("Create job: all attributes"):
            r = requests.post(job_url, json=payload)
            self.assertEqual(r.status_code, 201)

        with self.subTest("Try overriding the immutable attributes"):
            update = { "description": "Test job", "profile": "myprofile2",
                    "attributes": { "event_min_confidence": 2,
                                     "schedule_max_commits": 3,
                                     "perf_min_change": 4,
                                     "variance_max": 5,
                                     "variance_max_run_count": 6,
                                     "variance_min_run_count": 7,
                                     "report_priority": 8,
                                     "report_deadline_soft": 9,
                                     "report_deadline_hard": 10
                                  },
                    "machines": ["machine1", "machine2"]
                  }
            r = requests.patch(job_url, json=update)
            self.assertEqual(r.status_code, 200)

        with self.subTest("Fetching and comparing the job to the original request"):
            r = requests.get(job_url)
            self.assertEqual(r.status_code, 200)
            state = r.json()
            for key in state:
                if key == "attributes":
                    self.assertEqual(len(state['attributes']),
                                    len(payload['attributes']))
                    for attr in state['attributes']:
                        self.assertAlmostEqual(state['attributes'][attr],
                                            payload['attributes'][attr])
                elif key == "machines":
                    for machine in ["machine1", "machine2"]:
                        self.assertIn(machine, state['machines'])
                        self.assertIs(state['machines'][machine]['online'], False)
                        self.assertIsInstance(state['machines'][machine]['state'], str)
                    pass
                elif key == "id":
                    self.assertEqual(state['id'], "controllerd_test")
                else:
                    self.assertEqual(state[key],
                                     payload[key], key)

        with self.subTest("Verify that a repository has been created for machine[12]"):
            for machine, should_exist in [("machine1", True), ("machine2", True), ("machine3", False)]:
                self.check_report_repo(machine, should_exist)

        with self.subTest("Delete the report locally and make sure it gets re-created"):
            # Delete all the trees
            for machine in ["machine1", "machine2"]:
                try:
                    shutil.rmtree(self.report_repo_fullpath(machine))
                except:
                    pass
                self.check_report_repo(machine, False)

            # Submit a new job and make sure the controller re-creates the report
            r = requests.patch(job_url, json=update)
            self.assertEqual(r.status_code, 200)

            # Check that all the trees got re-created
            for machine, should_exist in [("machine1", True), ("machine2", True), ("machine3", False)]:
                self.check_report_repo(machine, should_exist)

        with self.subTest("Try overriding the mutable attributes"):
            # Update the description and machines
            update = { "description": "Test job 2", "machines": ["machine1"] }
            r = requests.patch(job_url, json=update)
            self.assertEqual(r.status_code, 200)

            # Second, check that
            r = requests.get(job_url)
            self.assertEqual(r.status_code, 200)
            state = r.json()
            self.assertEqual(state['description'], "Test job 2")
            self.assertEqual(len(state['machines']), 1)
            self.assertIn("machine1", state['machines'])

        with self.subTest("Verifying that the delete command only got sent to the machine2"):
            for machine, count in [("machine1", 0), ("machine2", 1), ("machine3", 0)]:
                r = requests.get(url("/machines/{}".format(machine)))
                self.assertEqual(r.status_code, 200)
                self.assertEqual(len(r.json()['queued_cmds']), count)

        with self.subTest("Verify that the repository got deleted only for the machine2"):
            for machine, should_exist in [("machine1", True), ("machine2", False), ("machine3", False)]:
                self.check_report_repo(machine, should_exist)

        with self.subTest("Delete the job"):
            r = requests.delete(job_url)
            self.assertEqual(r.status_code, 200)

        with self.subTest("Verifying that the delete command got sent to both machine1 and 2 but not 3"):
            for machine_name, count in [("machine1", 1), ("machine2", 1), ("machine3", 0)]:
                r = requests.get(url("/machines/{}".format(machine_name)))
                self.assertEqual(r.status_code, 200)
                self.assertEqual(len(r.json()['queued_cmds']), count)

                # Set up each machines and check we received the delete command
                machine = ControllerdTestClient(machine_name)
                machine.connect()
                for i in range(0, count):
                    cmd = self.read_cmd(machine)
                    self.assertIsNot(cmd, None)
                    self.assertTrue(cmd.HasField("delete_report"))
                    self.assertEqual(cmd.delete_report.name, "controllerd_test")
                machine.close()

    def __check_set_work_cmd(self, cmd, job, work, machine_name):
        self.assertIsNot(cmd, None)
        self.assertTrue(cmd.HasField("set_work"))

        # Check the job
        self.assertEqual(cmd.set_work.report.name, "controllerd_test")
        self.assertEqual(cmd.set_work.report.description, job["description"])
        self.assertEqual(cmd.set_work.report.profile, job["profile"])
        self.assertEqual(cmd.set_work.report.upload_url, "/controllerd_test/{}".format(machine_name))
        self.assertEqual(len(cmd.set_work.report.attributes), len(job['attributes']))
        for attr in cmd.set_work.report.attributes:
            for name in attr.DESCRIPTOR.fields_by_name:
                if attr.HasField(name):
                    self.assertAlmostEqual(getattr(attr, name), job['attributes'][name], places=4)

        # Check the work
        self.assertEqual(len(cmd.set_work.commits), len(work['commits']))
        for commit in cmd.set_work.commits:
            self.assertIn(commit.id, work['commits'])

            self.assertEqual(len(commit.tests), len(work['commits'][commit.id].get('tests', [])))
            for test in commit.tests:
                self.assertEqual(test.rounds, work['commits'][commit.id]['tests'][test.name])

            self.assertEqual(len(commit.testsets), len(work['commits'][commit.id].get('testsets', [])))
            for testset in commit.testsets:
                self.assertEqual(testset.rounds, work['commits'][commit.id]['testsets'][testset.name])

    def __check_work__(self, job_url, job, work, machines):
        r = requests.get(job_url + "/work")
        self.assertEqual(r.status_code, 200)
        state = r.json()
        self.assertIn("commits", state)
        self.assertDictEqual(state, work)

        for machine_name, has_msg in machines:
            machine = ControllerdTestClient(machine_name)

            if has_msg:
                machine.connect()
                cmd = self.read_cmd(machine)
                self.assertIsNot(cmd, None)
                machine.close()
                self.__check_set_work_cmd(cmd, job, work, machine_name)
            else:
                r = requests.get(url("/machines/{}".format(machine_name)))
                self.assertEqual(r.status_code, 200)
                self.assertEqual(len(r.json()['queued_cmds']), 0)

    def test_work(self):
        job_url = url("/jobs/controllerd_test")

        # Add 3 machines
        for i in range(1, 4):
            machine = ControllerdTestClient("machine{}".format(i))
            machine.connect()
            machine.close()

        # Create the resource
        job = { "description": "Test job", "profile": "myprofile",
                    "attributes": { "event_min_confidence": 0.995,
                                     "schedule_max_commits": 20,
                                     "perf_min_change": 0.01,
                                     "variance_max": 0.05,
                                     "variance_max_run_count": 10,
                                     "variance_min_run_count": 3,
                                     "report_priority": 42,
                                     "report_deadline_soft":  123456,
                                     "report_deadline_hard": 1234567
                                  },
                    "machines": ["machine1", "machine2"]
                  }
        with self.subTest("Create a job"):
            r = requests.post(job_url, json=job)
            self.assertEqual(r.status_code, 201)

        with self.subTest("Check that work is empty"):
            self.__check_work__(job_url, job, {"commits": {}}, [])

        with self.subTest("Queue work: No 'commits' attribute"):
            invalid_work = {"hello": "world"}
            r = requests.put(job_url + "/work", json=invalid_work)
            self.assertEqual(r.status_code, 400)

        with self.subTest("Queue work: Invalid type for 'commits'"):
            invalid_work = {"commits": "hello"}
            r = requests.put(job_url + "/work", json=invalid_work)
            self.assertEqual(r.status_code, 400)

        with self.subTest("Queue work: Invalid commit type"):
            invalid_work = {"commits": {self.src_repo_commit1: "hello"}}
            r = requests.put(job_url + "/work", json=invalid_work)
            self.assertEqual(r.status_code, 400)

        with self.subTest("Queue work: No work"):
            invalid_work = {"commits": {self.src_repo_commit1: {"hello": "world"}}}
            r = requests.put(job_url + "/work", json=invalid_work)
            self.assertEqual(r.status_code, 400)

        for name, msg_count in [('tests', 1), ('testsets', 2)]:
            with self.subTest("Queue work: {}: Invalid '{}' field type".format(name, name)):
                invalid_work = {"commits": {self.src_repo_commit1: {name: "world"}}}
                r = requests.put(job_url + "/work", json=invalid_work)
                self.assertEqual(r.status_code, 400)

            with self.subTest("Queue work: {}: Invalid round count type".format(name)):
                invalid_work = {"commits": {self.src_repo_commit1: {name: {"hello": "world"}}}}
                r = requests.put(job_url + "/work", json=invalid_work)
                self.assertEqual(r.status_code, 400)

            with self.subTest("Queue work: {}: Using HEAD for the commit".format(name)):
                invalid_work = {"commits": {"HEAD": {name: {"no-op": 3}}}}
                r = requests.put(job_url + "/work", json=invalid_work)
                self.assertEqual(r.status_code, 400)

            with self.subTest("Queue work: {}: Using a random version for the commit".format(name)):
                invalid_work = {"commits": {"0123456789": {name: {"no-op": 3}}}}
                r = requests.put(job_url + "/work", json=invalid_work)
                self.assertEqual(r.status_code, 400)

            work = {"commits": { self.src_repo_commit1: {name: {"no-op": 3, "test2": 2}},
                                 self.src_repo_commit2: {name: {"no-op": 3, "test3": 3}}}}

            with self.subTest("Queue work: {}: Valid request".format(name)):
                r = requests.put(job_url + "/work", json=work)
                self.assertEqual(r.status_code, 200)
                self.__check_work__(job_url, job, work, [("machine1", True), ("machine2", True), ("machine3", False)])

        work = {"commits": { self.src_repo_commit1:    {"tests": {"test1": 1, "test2": 2},
                                                        "testsets": {"ts1": 1, "ts2": 2}},
                             self.src_repo_commit2_tag: {"tests": {"test3": 3, "test4": 4},
                                                         "testsets": {"ts3": 3, "ts4": 4}}}}

        with self.subTest("Queue some valid work"):
            r = requests.put(job_url + "/work", json=work)
            self.assertEqual(r.status_code, 200)
            self.__check_work__(job_url, job, work, [("machine1", True), ("machine2", True), ("machine3", False)])

        with self.subTest("Patch the work"):
            patch = {"commits": { self.src_repo_commit1: {"tests": {"test1": 1, "test2": -2},
                                                          "testsets": {"ts1": -1, "ts2": 0}},
                                  self.src_repo_commit3: {"tests": {"test3": 3, "test4": 0},
                                                         "testsets": {"ts3": -3, "ts4": 4}}}}
            r = requests.patch(job_url + "/work", json=patch)
            self.assertEqual(r.status_code, 200)

            # Update work
            work = {"commits": { self.src_repo_commit1:    {"tests": {"test1": 2},
                                                            "testsets": {"ts2": 2}},
                                 self.src_repo_commit2_tag: {"tests": {"test3": 3, "test4": 4},
                                                            "testsets": {"ts3": 3, "ts4": 4}},
                                 self.src_repo_commit3:     {"tests": {"test3": 3},
                                                            "testsets": {"ts4": 4}}}}
            self.__check_work__(job_url, job, work, [("machine1", True), ("machine2", True), ("machine3", False)])


    # TODO: Add a test that verify that even after tearDown, the controller
    # still has the data available
