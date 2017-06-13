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

from collections import namedtuple
import traceback
import threading
import unittest
import shutil
import socket
import copy
import json
import time
import sys
import os

from datetime import datetime
import pygit2

from utils import send_msg, recv_msg, GitRepoFactory, tmp_folder, ezbench_dir, timings_db_dir
from ezbench.smartezbench import RunnerTest, RunningMode, SmartEzbenchAttributes, TaskEntry
from ezbench.scm import GitRepo
import controllerd_pb2
import dutd

class MockupSmartEzbench:
    def __init__(self, ezbench_dir, report_name, readonly = False,
                 hook_binary_path = None, logs_callback = None,
                 hooks_callback = None):
        self.ezbench_dir = ezbench_dir
        self.report_name = report_name
        self.readonly = readonly
        self.hook_binary_path = hook_binary_path
        self.logs_callback = logs_callback
        self.hooks_callback = hooks_callback

        self.log_folder = "{}/dut/reports/{}".format(tmp_folder, report_name)

        self.run_called = 0
        self.schedule_enhancements_called = 0
        self._first_run = True
        self._deleted = False

        self.state = dict()
        self.state["running_mode"] = RunningMode.INITIAL.value
        self.state["profile"] = None
        self.state["conf_scripts"] = []
        self.state["commit_url"] = None
        self.state["attributes"] = dict()
        self.state["commits_user"] = dict()
        self.state["commits_auto"] = dict()
        self.state["executed"] = dict()
        self.state["user_data"] = dict()

        self.run_barrier = threading.Event()
        self.run_called_event = threading.Event()
        self.schedule_enhancements_barrier = threading.Event()
        self.schedule_enhancements_called_event = threading.Event()

        # Create the report folder like smartezbench would do
        os.makedirs(self.log_folder)
        with open("{}/smartezbench.log".format(self.log_folder), "w") as f:
            f.write("blabla")

        self.__save_state()

    def __save_state(self):
        with open("{}/smartezbench.state".format(self.log_folder), "w") as f:
            f.write(json.dumps(self.state, sort_keys=True, indent=4))

    def __call_hook__(self, action, parameters = dict()):
        if self.hooks_callback is not None:
            HookCallbackState = namedtuple('HookCallbackState', ['sbench', 'action',
                                                                 'hook_parameters'])
            state = HookCallbackState(sbench=self, action=str(action),
                                        hook_parameters=parameters)
            try:
                self.hooks_callback(state)
            except:
                traceback.print_exc(file=sys.stderr)
                sys.stderr.write("\n")

    def __send_logs(self, msg):
        if self.logs_callback is not None:
            self.logs_callback(self, msg)

    def first_run(self):
        return self._first_run

    def delete(self):
        self._deleted = True
        self.__send_logs("Deleting the report")

    def running_mode(self, check_running = True):
        if check_running and self.state["running_mode"] > RunningMode.INTERMEDIATE.value:
            return RunningMode(self.state["running_mode"] - RunningMode.INTERMEDIATE.value)
        return RunningMode(self.state["running_mode"])

    def set_running_mode(self, mode):
        dsk_mode = self.running_mode(check_running=False)
        if dsk_mode.value != mode.value:
            self.state["running_mode"] = mode.value
            #self.__send_logs("Set running mode from {} to {}".format(dsk_mode.name, mode.name))
            self.__save_state()

            params = dict()
            params['ezbench_report_mode_prev'] = dsk_mode.name
            self.__call_hook__('mode_changed', params)

    def attribute(self, param):
        p = SmartEzbenchAttributes[param]
        if p == SmartEzbenchAttributes.perf_min_change:
            return self.state["attributes"].get(param, 0.005)
        elif p == SmartEzbenchAttributes.event_min_confidence:
            return self.state["attributes"].get(param, 0.99)
        elif p == SmartEzbenchAttributes.schedule_max_commits:
            return self.state["attributes"].get(param, 1)
        elif p == SmartEzbenchAttributes.variance_max:
            return self.state["attributes"].get(param, 0.025)
        elif p == SmartEzbenchAttributes.variance_max_run_count:
            return self.state["attributes"].get(param, 20)
        elif p == SmartEzbenchAttributes.variance_min_run_count:
            return self.state["attributes"].get(param, 2)
        elif p == SmartEzbenchAttributes.report_priority:
            return self.state["attributes"].get(param, 0)
        elif p == SmartEzbenchAttributes.report_deadline_soft:
            return self.state["attributes"].get(param, -1)
        elif p == SmartEzbenchAttributes.report_deadline_hard:
            return self.state["attributes"].get(param, -1)

    def set_attribute(self, param, value):
        # verify that the attribute exists
        p = SmartEzbenchAttributes[param]
        self.state["attributes"][param] = value
        self.__save_state()

    def profile(self):
        return self.state["profile"]

    def set_profile(self, profile):
        if self.state["profile"] is None:
            self.state["profile"] = profile
            self.__save_state()
            return True
        elif self.state["profile"] == profile:
            return True
        else:
            return False

    def conf_scripts(self):
        return self.state["conf_scripts"]

    def add_conf_script(self, conf_script):
        self.state["conf_scripts"].append(conf_script)
        self.__save_state()

    def remove_conf_script(self, conf_script):
        self.state["conf_scripts"].remove(conf_script)
        self.__save_state()

    def commit_url(self):
        return self.state["commit_url"]

    def set_commit_url(self, commit_url):
        self.state["commit_url"] = commit_url
        self.__save_state()

    def user_data(self, key, default=None):
        return self.state["user_data"].get(key, default)

    def set_user_data(self, key, value):
        self.state["user_data"][key] = value
        self.__save_state()

    def __task_tree_add_test__(self, task_tree, commit, test, rounds):
        if commit not in task_tree:
            task_tree[commit] = dict()
            task_tree[commit]["tests"] = dict()

        if test not in task_tree[commit]['tests']:
            task_tree[commit]['tests'][test] = dict()
            task_tree[commit]['tests'][test]['rounds'] = rounds
            total_rounds_before = 0
        else:
            total_rounds_before = task_tree[commit]['tests'][test]['rounds']
            task_tree[commit]['tests'][test]['rounds'] += rounds

        total_rounds_after = task_tree[commit]['tests'][test]['rounds']

        # if the number of rounds is equal to 0 for a test, delete it
        if task_tree[commit]['tests'][test]['rounds'] <= 0:
            del task_tree[commit]['tests'][test]
            total_rounds_after = 0

        # Delete a commit that has no test
        if len(task_tree[commit]['tests']) == 0:
            del task_tree[commit]

        return total_rounds_before, total_rounds_after

    def add_test(self, commit, test, rounds = None, user_requested=True):
        if user_requested:
            commits = self.state["commits_user"]
        else:
            commits = self.state["commits_auto"]

        ret = self.__task_tree_add_test__(commits, commit, test, rounds)
        self.__save_state()
        return ret

    def add_testset(self, commit, testset, rounds = 1, ensure=False, user_requested=True):
        if user_requested:
            commits = self.state["commits_user"]
        else:
            commits = self.state["commits_auto"]

        for test in testset:
            self.add_test(commit, test, testset[test] * rounds)

        self.__save_state()
        return 0

    def reset_work(self):
        self.state["commits_user"] = dict()
        self.state["commits_auto"] = dict()
        self.__save_state()

    def repo(self):
        return GitRepo(dutd.Configuration.repo_git_path)

    def __gen_task_list__(self, commits, user_requested=True):
        tl = []
        for commit in commits:
            for test in commits[commit]['tests']:
                wanted_rounds = commits[commit]['tests'][test]['rounds']
                tmp, found = self.__task_tree_add_test__(self.exec_state, commit, test, 0)

                # Always run what auto asks, to make the code easier to read
                if not user_requested:
                    tl.append(TaskEntry(commit, test, wanted_rounds,
                                        user_requested=user_requested))
                elif wanted_rounds > found:
                    tl.append(TaskEntry(commit, test, wanted_rounds - found,
                                        user_requested=user_requested))
        return tl

    def __reload_executed_state__(self):
        try:
            with open("{}/execution.state".format(self.log_folder), "rt") as f:
                self.exec_state = json.loads(f.read())
        except IOError as e:
            self.exec_state = dict()

    def __save_executed_state__(self):
        with open("{}/execution.state".format(self.log_folder), "wt") as f:
            f.write(json.dumps(self.exec_state, sort_keys=True, indent=4, separators=(',', ': ')))

    def run(self):
        if not self.run_barrier.wait(0.5):
            raise ValueError("run() called when it should not be")
        self.run_barrier.clear()

        self._first_run = False
        self.run_called += 1

        self.__reload_executed_state__()

        task_list = self.__gen_task_list__(self.state["commits_user"], True)
        task_list += self.__gen_task_list__(self.state["commits_auto"], False)

        if len(task_list) == 0:
            ret = False
        else:
            self.__call_hook__('start_running_tests')

            exit = False
            for task in task_list:
                if self._deleted or exit:
                    break

                task.started()
                for r in range(0, task.rounds):
                    self.__call_hook__('start_running_test', { "task": task })

                    # Get the right dictionary containing the results
                    if task.user_requested:
                        commits = self.state["commits_user"]
                    else:
                        commits = self.state["commits_auto"]

                    # Generate some fake results
                    tmp, exec_count = self.__task_tree_add_test__(self.exec_state,
                                                                task.commit, task.test, 0)
                    run_file = "{}/{}_unified_{}#{}".format(self.log_folder, task.commit,
                                                            task.test, exec_count)
                    with open(run_file, "w") as f:
                        f.write("general: float(10.0)\n")

                    # Update the execution count
                    self.__task_tree_add_test__(self.exec_state, task.commit,
                                                task.test, 1)

                    self.__save_executed_state__()
                    self.__save_state()

                    self.__call_hook__('done_running_test', { "task": task })
                    task.round_done()

                    if task.test == "test_reboot":
                        self.__call_hook__('reboot_needed', { "task": task })
                        exit = True
                        break

            self.__call_hook__('done_running_tests')
            ret = True

        # Say we are done running
        self.run_called_event.set()

        # Now wait for the right to exit
        if not self.run_barrier.wait(1):
            raise ValueError("run() never got allowed to exit")
        self.run_barrier.clear()

        return True

    def schedule_enhancements(self):
        if not self.schedule_enhancements_barrier.wait(0.5):
            raise ValueError("schedule_enhancements() called when it should not be")
        self.schedule_enhancements_barrier.clear()

        self.schedule_enhancements_called += 1

        if self.schedule_enhancements_called == 1:
            self.add_test("commit2_tag", "test1", 1, user_requested=False)
        else:
            self.set_running_mode(RunningMode.DONE)

        self.schedule_enhancements_called_event.set()
        return True

class MockupSmartEzbenchSingle:
    reports = dict()

    @classmethod
    def clear(cls):
        cls.reports = dict()

    @classmethod
    def list_reports(cls, ezbench_dir, updatedSince = 0):
        reports = []
        for report_name in cls.reports:
            if not cls.reports[report_name]._deleted:
                reports.append(report_name)
        return reports

    def __new__(cls, *args, **kwargs):
        report_name = args[1]
        if report_name in cls.reports:
            sbench = cls.reports[report_name]
            sbench._first_run = False
            return sbench
        else:
            sbench = MockupSmartEzbench(*args, **kwargs)
            cls.reports[report_name] = sbench
            return sbench

class MockupRunner:
    tests = { "test1": RunnerTest("test1", "type1", "unit1", True, 42.42),
              "test2": RunnerTest("test2", "type2", "unit2", False, 13.37),
              "test_reboot": RunnerTest("test_reboot", "type3", "unit1", False, 1234)}

    def __init__(self, ezbench_dir):
        self.ezbench_dir = ezbench_dir

    def list_tests(self):
        return self.tests.values()



class MockupTestset(dict):
    def __init__(self, name, description, tests):
        self.name = name
        self.description = description
        for test in tests:
            self[test] = tests[test]

    def parse(self, available_tests, silent=False):
        return True

    @classmethod
    def dict(cls):
        return {
            "testset1": MockupTestset("testset1", "desc1", {"test1": 1, "test2": 2, "test3": 3}),
            "testset2": MockupTestset("testset2", "desc2", {"test4": 4, "test5": 5, "test6": 6}),
            "testset3": MockupTestset("testset3", "desc3", {"test7": 7, "test8": 8, "test9": 9})
        }

    @classmethod
    def list(cls, ezbench_dir):
        return cls.dict().values()

class DutdTestServer:
    def __init__(self, controller_name):
        self.controller_name = controller_name

        self.cmd_id_next = 0
        self.dutd_socket = None
        self.dutd_logs = dict()
        self.close()

    def wait_for_client(self, handshake=True):
        self.close()

        tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcpsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        tcpsock.bind(dutd.Configuration.controller_host)
        tcpsock.listen(1)
        try:
            (self.dutd_socket, (ip, port)) = tcpsock.accept()
        except socket.timeout:
            return False
        except OSError:
            return False

        tcpsock.close()

        # Add a timeout on the socket operations to exit early
        self.dutd_socket.settimeout(.5)

        # Follow the handshake protocol
        hello = self.recv_msg(controllerd_pb2.ClientHello())
        if hello is None:
            raise ValueError("Did not receive the DUT's hello message")

        self.dutd_name = hello.machine_name
        self.protocol_version = hello.version

        if handshake:
            hello_back = controllerd_pb2.ServerHello()
            hello_back.version = 1
            hello_back.controller_name = self.controller_name
            self.send_msg(hello_back)

        return True

    def close(self):
        if self.dutd_socket is not None:
            self.dutd_socket.close()
        self.dutd_name = None
        self.protocol_version = -1


    def send_msg(self, msg):
        return send_msg(self.dutd_socket, msg)

    def recv_msg(self, msg):
        msg = recv_msg(self.dutd_socket, msg)

        # If the message is a log message, store it
        if type(msg) == controllerd_pb2.Signal:
            if msg.HasField("log"):
                if msg.log.report not in self.dutd_logs:
                    self.dutd_logs[msg.log.report] = list()
                self.dutd_logs[msg.log.report].append(msg.log.msg)

        #print(msg)
        return msg

    def send_cmd(self, cmd):
        # set a command ID
        cmd.id = self.cmd_id_next
        self.cmd_id_next += 1

        # Send the command
        self.send_msg(cmd)

        # Now wait for the ACK
        signals = []
        while True:
            sig = self.recv_msg(controllerd_pb2.Signal())
            if sig.HasField("cmd_status"):
                if sig.cmd_status.id != cmd.id:
                    raise ValueError("sig.cmd_status.id({}) != cmd.id({})".format(sig.cmd_status.id,
                                                                                  cmd.id))
                return sig.cmd_status, signals
            else:
                signals.append(sig)

        return None, []

class Dutd(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

        self.server = DutdTestServer("Test Server")

        # Generate a configuration file
        dutd.Configuration.controllerd_reconnection_period = .05
        dutd.Configuration.report_idle_loop_period = 0.001
        dutd.Configuration.report_parsing_in_separate_process = False
        dutd.Configuration.machine_name = "Test Dutd"
        dutd.Configuration.controller_host = ("127.0.0.1", 42001)
        dutd.Configuration.controller_reports_base_url = tmp_folder + "/ctrl/reports/"
        dutd.Configuration.controller_git_repo_url = tmp_folder + "/ctrl/dut-repo/"

        dutd.Configuration.credentials_user = "unused"
        dutd.Configuration.credentials_ssh_key_priv = None
        dutd.Configuration.credentials_ssh_key_pub = None
        dutd.Configuration.credentials_ssh_key_pass = None
        dutd.Configuration.credentials_password = None

        dutd.Configuration.repo_git_path = tmp_folder + "/dut/src-repo/"

        # remove any left-over tmp folder, before we re-create it
        try:
            shutil.rmtree(tmp_folder)
        except OSError as e:
            if os.path.exists(tmp_folder):
                msg = "Fatal error: cannot delete the report folder '{}': {}"
                raise ValueError(msg.format(tmp_folder, e)) from e

        # Create the temporary directories
        os.makedirs(dutd.Configuration.controller_reports_base_url, exist_ok=True)
        os.makedirs(dutd.Configuration.controller_git_repo_url, exist_ok=True)
        os.makedirs(dutd.Configuration.repo_git_path, exist_ok=True)

        # Create the source repository
        self.dut_repo = GitRepoFactory(dutd.Configuration.controller_git_repo_url, True)
        self.dut_repo_commit1 = self.dut_repo.create_commit("commit1",
                                                            [("commit", "commit1")])
        self.dut_repo_commit2 = self.dut_repo.create_commit("commit2",
                                                            [("commit", "commit2")],
                                                            [self.dut_repo_commit1])
        self.dut_repo_commit2_tag = "commit2_tag"
        self.dut_repo.create_tag(self.dut_repo_commit2_tag, self.dut_repo_commit2)

        # Create a first commit for the report_repo
        self.report_repo = GitRepoFactory(dutd.Configuration.controller_reports_base_url, True)
        self.report_repo_commit_init = self.report_repo.create_commit("initial commit",
                                                                      [("job_desc", "blabla\n")])
        #
        GitRepoFactory(dutd.Configuration.repo_git_path)

        # Create useless reports that should never be shown to the controller
        MockupSmartEzbenchSingle.clear()
        MockupSmartEzbenchSingle(ezbench_dir, "report1")
        MockupSmartEzbenchSingle(ezbench_dir, "invalid1")
        MockupSmartEzbenchSingle(ezbench_dir, "invalid2")

        self.client = dutd.ControllerClient(ezbench_dir, timings_db_dir,
                                           smartezbench_class=MockupSmartEzbenchSingle,
                                           runner_class=MockupRunner,
                                           testset_class=MockupTestset)
        self.client.start()

        run_report_thread = threading.Thread(target=self.client.run_reports)
        run_report_thread.start()

    def tearDown(self):
        self.client.stop()
        self.server.close()
        MockupSmartEzbenchSingle.clear()

    def _report_full_name(self, report_name):
        return "{}/{}".format(self.server.controller_name, report_name)

    def check_sig_reports(self, sig):
        self.assertTrue(sig.HasField("reports"), str(sig))
        self.assertEqual(len(sig.reports.reports), sig.reports.report_count)
        for report in sig.reports.reports:
            fullname = self._report_full_name(report.name)
            mocked_report = MockupSmartEzbenchSingle.reports.get(fullname, None)
            self.assertNotEqual(mocked_report, None)
            self.assertEqual(fullname, mocked_report.report_name)
            self.assertEqual(report.profile, mocked_report.profile())
            self.assertEqual(report.state, mocked_report.running_mode().name)
            self.assertEqual(report.state_disk, mocked_report.running_mode(check_running=False).name)
            self.assertEqual(report.build_time, 300)
            self.assertEqual(report.deploy_time, 120)

        # Now check how many reports we were supposed to get
        report_count = 0
        for report in MockupSmartEzbenchSingle.list_reports(ezbench_dir):
            if report.startswith(self.server.controller_name + "/"):
                sbench = MockupSmartEzbenchSingle(ezbench_dir, report)
                if not sbench._deleted:
                    report_count += 1
        self.assertEqual(len(sig.reports.reports), report_count)

    def check_set_work_sigs(self, sigs):
        # Check that we received all the signals we needed after a set_work cmd
        found_reports_sig = False
        for sig in sigs:
            if sig.HasField("reports"):
                self.check_sig_reports(sig)
                found_reports_sig = True

        self.assertTrue(found_reports_sig)

    def check_for_start_signals(self):
        tests_received = False
        testsets_received = False
        reports_received = False
        while not tests_received or not testsets_received or not reports_received:
            sig = self.server.recv_msg(controllerd_pb2.Signal())
            self.assertIsNot(sig, None)

            if sig.HasField("tests"):
                tests_received = True
                self.assertEqual(len(sig.tests.tests), sig.tests.tests_count)
                for test in sig.tests.tests:
                    mockup_test = MockupRunner.tests[test.name]
                    self.assertEqual(test.name, mockup_test.name)
                    self.assertAlmostEqual(test.exec_time_max, mockup_test.time_estimation, places=4)
                    self.assertAlmostEqual(test.exec_time_median, mockup_test.time_estimation, places=4)
            elif sig.HasField("testsets"):
                testsets_received = True
                self.assertEqual(len(sig.testsets.testsets), sig.testsets.testsets_count)
                for testset in sig.testsets.testsets:
                    mockup_testset = MockupTestset.dict()[testset.name]
                    self.assertEqual(testset.name, mockup_testset.name)
                    self.assertEqual(testset.description, mockup_testset.description)
                    for test in testset.tests:
                        self.assertIn(test.name, mockup_testset)
                        self.assertEqual(test.rounds, mockup_testset[test.name])
            elif sig.HasField("reports"):
                reports_received = True
                self.check_sig_reports(sig)

        self.assertTrue(tests_received)
        self.assertTrue(testsets_received)
        self.assertTrue(reports_received)

    def test_connection(self):
        with self.subTest("Initial connection, no message reading"):
            ret = self.server.wait_for_client(handshake=None)
            self.assertTrue(ret, "The connection to the client failed")
            self.assertEqual(self.server.dutd_name, dutd.Configuration.machine_name)
            self.server.close()

        with self.subTest("Second reconnection, full set up testing"):
            start_wait = time.monotonic()
            ret = self.server.wait_for_client()
            connect_delay = time.monotonic() - start_wait

            # Check that the connection happened following the specified timing
            self.assertTrue(ret)
            self.assertGreater(connect_delay, dutd.Configuration.controllerd_reconnection_period)
            self.assertLess(connect_delay, dutd.Configuration.controllerd_reconnection_period + 0.02)

            # Check that we automatically receive the list tests and testsets
            self.check_for_start_signals()

            # Check that the ping command works
            cmd = controllerd_pb2.Cmd()
            cmd.ping.requested = True
            ret, sigs = self.server.send_cmd(cmd)
            self.assertEqual(ret.err_code, controllerd_pb2.CmdStatus.OK)
            self.assertEqual(ret.err_msg, str())

            self.server.close()

    def _compare_cmd_with_report(self, cmd):
        report_fullname = self._report_full_name(cmd.set_work.report.name)

        # Get the report matching $report_fullname
        self.assertIn(report_fullname, MockupSmartEzbenchSingle.list_reports(ezbench_dir))
        sbench = MockupSmartEzbenchSingle(ezbench_dir, report_fullname)

        # Check the general fields
        self.assertEqual(report_fullname, self.server.controller_name + "/" + cmd.set_work.report.name)
        self.assertEqual(sbench.profile(), cmd.set_work.report.profile)
        self.assertEqual(sbench.user_data("description"), cmd.set_work.report.description)
        self.assertEqual(sbench.user_data("upload_url"), cmd.set_work.report.upload_url)
        self.assertEqual(sbench.user_data("controller_name"), self.server.controller_name)
        self.assertEqual(cmd.set_work.report.name, sbench.user_data("report_name"))

        # Make sure the report have the right attributes
        for attr in cmd.set_work.report.attributes:
            for name in attr.DESCRIPTOR.fields_by_name:
                if attr.HasField(name):
                    self.assertAlmostEqual(sbench.attribute(name), getattr(attr, name), places=4)

    def _report_git_history(self, branch="master"):
        master_head = self.report_repo.src_repo.revparse_single(branch).hex
        return [(x.hex, x.message) for x in self.report_repo.src_repo.walk(master_head)]

    def _queue_work(self, report_name, commit_id, test_name, rounds):
        cmd = controllerd_pb2.Cmd()
        cmd.set_work.report.name = report_name
        cmd.set_work.report.description = "Description for " + report_name
        cmd.set_work.report.upload_url = ""
        cmd.set_work.report.profile = "profile1"

        commit1 = cmd.set_work.commits.add()
        commit1.id = commit_id
        test = commit1.tests.add(); test.name=test_name; test.rounds=rounds

        return self.server.send_cmd(cmd)

    def _wait_for_run_to_be_done(self, sbench, expected_signals=[], check_signals=True):
        sigs = []

        # Make sure run() got executed the right amount of times already
        self.assertEqual(sbench.run_called, self.run_called)

        # Unlock run() to allow it to run
        self.assertFalse(sbench.run_barrier.is_set())
        sbench.run_barrier.set()

        # Catch the reports update
        if check_signals:
            self._wait_for_reports_update()

        # Wait for the run method to be done running
        self.assertTrue(sbench.run_called_event.wait(0.5))
        sbench.run_called_event.clear()

        # Check that we received the right amount of "report_pushed" messages
        for signal in expected_signals:
            sig = self.server.recv_msg(controllerd_pb2.Signal())
            self.assertIsNot(sig, None)
            if not sig.HasField(signal):
                self.assertTrue(sig.HasField("log"),
                                "Unexpected signal type: sig={}".format(sig))
            sigs.append(sig)

        # Catch the reports update
        if check_signals:
            self._wait_for_reports_update()

        # Update the run counter and check it matches sbench's internal state
        self.run_called += 1
        self.assertEqual(sbench.run_called, self.run_called)

        return sigs

    def _wait_for_reports_update(self):
        sig = self.server.recv_msg(controllerd_pb2.Signal())
        self.assertIsNot(sig, None)
        self.check_sig_reports(sig)

    def test_work(self):
        cmd = controllerd_pb2.Cmd()
        cmd.set_work.report.name = "report1"
        cmd.set_work.report.description = "Description for report1"
        cmd.set_work.report.upload_url = ""
        cmd.set_work.report.profile = "profile1"
        attr = cmd.set_work.report.attributes.add(); attr.event_min_confidence = 0.995
        attr = cmd.set_work.report.attributes.add(); attr.schedule_max_commits = 2
        attr = cmd.set_work.report.attributes.add(); attr.perf_min_change = 0.5
        attr = cmd.set_work.report.attributes.add(); attr.variance_max = 0.25
        attr = cmd.set_work.report.attributes.add(); attr.variance_max_run_count = 20
        attr = cmd.set_work.report.attributes.add(); attr.variance_min_run_count = 10
        attr = cmd.set_work.report.attributes.add(); attr.report_priority = 5
        attr = cmd.set_work.report.attributes.add(); attr.report_deadline_soft = 4561
        attr = cmd.set_work.report.attributes.add(); attr.report_deadline_hard = 4567

        report_fullname = self._report_full_name(cmd.set_work.report.name)

        with self.subTest("Create a report"):
            self.server.wait_for_client()
            self.check_for_start_signals()

            # Verify that the report is not already in the list of reports
            self.assertNotIn(report_fullname, MockupSmartEzbenchSingle.list_reports(ezbench_dir))

            # Test multiple times for checking the idem-potence
            for i in range(0, 3):
                ret, sigs = self.server.send_cmd(cmd)
                self.assertEqual(ret.err_code, controllerd_pb2.CmdStatus.OK)
                self.assertEqual(ret.err_msg, str())

                # Now make sure that what dutd reports is matching what we set
                self.check_set_work_sigs(sigs)

                # Verify that the command and the current state matches
                self._compare_cmd_with_report(cmd)

                # Check that we indeed got the report in the controller side
                master_hist = self._report_git_history()
                self.assertEqual(len(master_hist), 2)
                self.assertEqual(master_hist[-1][0], self.report_repo_commit_init)

                # Check that the run function has never be called
                sbench = MockupSmartEzbenchSingle(ezbench_dir, report_fullname)
                self.assertEqual(sbench.run_called, 0)

        with self.subTest("Check if we do get an answer to a ping"):
            cmd_ping = controllerd_pb2.Cmd()
            cmd_ping.ping.requested = True

            ret, sigs = self.server.send_cmd(cmd)
            self.assertEqual(ret.err_code, controllerd_pb2.CmdStatus.OK)
            self.assertEqual(ret.err_msg, str())

        with self.subTest("Update a report: Change profile"):
            cmd.set_work.report.profile = "profile2"
            ret, sigs = self.server.send_cmd(cmd)
            self.assertEqual(ret.err_code, controllerd_pb2.CmdStatus.OK)
            self.assertEqual(ret.err_msg, str())

            # Verify that nothing changed
            cmd.set_work.report.profile = "profile1"
            self._compare_cmd_with_report(cmd)

        with self.subTest("Update a report: Change upload URL"):
            cmd.set_work.report.upload_url = "new_folder"
            ret, sigs = self.server.send_cmd(cmd)
            self.assertEqual(ret.err_code, controllerd_pb2.CmdStatus.ERROR)
            self.assertEqual(ret.err_msg, "The upload_url cannot be changed")

            # Verify that nothing changed
            cmd.set_work.report.upload_url = ""
            self._compare_cmd_with_report(cmd)

        with self.subTest("Update a report: Change acceptable things"):
            cmd.set_work.report.description = "Description for report1 - updated"
            try:
                while True:
                    cmd.set_work.report.attributes.pop()
            except:
                pass
            attr = cmd.set_work.report.attributes.add(); attr.event_min_confidence = 0.994
            attr = cmd.set_work.report.attributes.add(); attr.schedule_max_commits = 3
            attr = cmd.set_work.report.attributes.add(); attr.perf_min_change = 0.6
            attr = cmd.set_work.report.attributes.add(); attr.variance_max = 0.30
            attr = cmd.set_work.report.attributes.add(); attr.variance_max_run_count = 10
            attr = cmd.set_work.report.attributes.add(); attr.variance_min_run_count = 3
            attr = cmd.set_work.report.attributes.add(); attr.report_priority = 4
            attr = cmd.set_work.report.attributes.add(); attr.report_deadline_soft = 456
            attr = cmd.set_work.report.attributes.add(); attr.report_deadline_hard = 459

            ret, sigs = self.server.send_cmd(cmd)
            self.assertEqual(ret.err_code, controllerd_pb2.CmdStatus.OK)
            self.assertEqual(ret.err_msg, str())

            self._compare_cmd_with_report(cmd)

            # Check that we indeed got the report in the controller side
            master_hist = self._report_git_history()
            self.assertEqual(len(master_hist), 3)
            self.assertEqual(master_hist[-1][0], self.report_repo_commit_init)

        with self.subTest("Queue work"):
            # Check that the run function has never be called
            sbench = MockupSmartEzbenchSingle(ezbench_dir, report_fullname)
            self.assertEqual(sbench.run_called, 0)

            commit1 = cmd.set_work.commits.add()
            commit1.id = self.dut_repo_commit1
            test = commit1.tests.add(); test.name="test1"; test.rounds=1
            test = commit1.tests.add(); test.name="test3"; test.rounds=3
            test = commit1.tests.add(); test.name="test2"; test.rounds=2

            commit2 = cmd.set_work.commits.add()
            commit2.id = self.dut_repo_commit2_tag
            testset = commit2.testsets.add(); testset.name="testset1"; testset.rounds=1
            testset = commit2.testsets.add(); testset.name="testset4"; testset.rounds=4
            testset = commit2.testsets.add(); testset.name="testset2"; testset.rounds=2

            commit3 = cmd.set_work.commits.add()
            commit3.id = "1337babedeadbeefcafe42"
            test = commit3.tests.add(); test.name="test1"; test.rounds=1

            user_work = {self.dut_repo_commit1: {'tests': {'test1': {'rounds': 1},
                                                           'test2': {'rounds': 2}}},
                        'commit2_tag': {'tests': {'test1': {'rounds': 1},
                                                  'test2': {'rounds': 2},
                                                  'test3': {'rounds': 3},
                                                  'test4': {'rounds': 8},
                                                  'test5': {'rounds': 10},
                                                  'test6': {'rounds': 12}}}
            }
            executed = copy.deepcopy(user_work)

            ret, sigs = self.server.send_cmd(cmd)
            self.assertEqual(ret.err_code, controllerd_pb2.CmdStatus.WARNING)
            self.assertNotIn("test1", ret.err_msg)
            self.assertNotIn("test2", ret.err_msg)
            self.assertIn("test3", ret.err_msg)
            self.assertNotIn("testset1", ret.err_msg)
            self.assertNotIn("testset2", ret.err_msg)
            self.assertIn("testset4", ret.err_msg)
            self.assertIn(commit3.id, ret.err_msg)

            # Verify that dutd actually fetched the code from dut-repo
            src_repo = pygit2.Repository(dutd.Configuration.repo_git_path)
            self.assertIn(self.dut_repo_commit1, src_repo)
            self.assertIn(self.dut_repo_commit2, src_repo)
            self.assertRaises(KeyError, src_repo.revparse_single, ("1337babedeadbeefcafe42"))

            # Check that the command and the programmed config matches
            commit1.tests.pop(1)
            commit2.testsets.pop(1)
            cmd.set_work.commits.pop(-1)
            self._compare_cmd_with_report(cmd)

            # Verify that the report's state as been well communicated
            self.assertEqual(len(sigs), 3, "Got the signals: {}".format(sigs))
            self.check_set_work_sigs(sigs)

            # Let's execute the worked queued and then pretend one more test needs
            # to be executed (hence the for loop).
            sbench = MockupSmartEzbenchSingle(ezbench_dir, report_fullname)
            state_auto = dict()
            self.run_called = 0
            schedule_enhancements_called = 0
            report_commits_count = 3 + 1 # Previous value + 1 because we submited more work
            test_exec_count = 39
            for i in range(0, 2):
                # Wait for the run() function to be done
                sigs = self._wait_for_run_to_be_done(sbench, test_exec_count * ['report_pushed'])

                # Check that all the report_pushed signals are for the report1
                for sig in sigs:
                    self.assertEqual(sig.report_pushed.report, "report1")

                # Check that the work queued corresponds to what we wanted and
                # that we got the expected amount of execution
                self.assertEqual(sbench.state["commits_user"], user_work)
                self.assertEqual(sbench.state["commits_auto"], state_auto)
                self.assertEqual(sbench.exec_state, executed)

                # Check that we got a new commit in the report per run
                report_commits_count += test_exec_count
                self.assertEqual(len(self._report_git_history()), report_commits_count)

                # Check that the ping command works, before returning from run()
                cmd_ping = controllerd_pb2.Cmd()
                cmd_ping.ping.requested = True
                ret, sigs = self.server.send_cmd(cmd_ping)
                self.assertEqual(ret.err_code, controllerd_pb2.CmdStatus.OK)
                self.assertEqual(ret.err_msg, str())
                self.assertEqual(len(sigs), 0, "Got the signals: {}".format(sigs))

                # Allow run() to return
                self.assertFalse(sbench.run_barrier.is_set())
                sbench.run_barrier.set()

                # Let the schedule_enhancements() function go through
                self.assertEqual(sbench.schedule_enhancements_called, schedule_enhancements_called)
                sbench.schedule_enhancements_barrier.set()
                self.assertTrue(sbench.schedule_enhancements_called_event.wait(0.5))
                self.assertFalse(sbench.run_barrier.is_set())
                sbench.schedule_enhancements_called_event.clear()
                schedule_enhancements_called += 1
                self.assertEqual(sbench.schedule_enhancements_called, schedule_enhancements_called)
                sbench.schedule_enhancements_barrier.clear()

                if i == 0:
                    # Update the auto state, since the first execution of
                    # schedule_enhancements adds one run
                    state_auto = {'commit2_tag': {'tests': {'test1': {'rounds': 1}}}}
                    executed['commit2_tag']['tests']['test1']['rounds'] += 1
                    test_exec_count = 1

                    self.assertEqual(sbench.running_mode(), RunningMode.RUN)
                else:
                    self.assertEqual(sbench.running_mode(), RunningMode.DONE)

                    # Verify that the report's state as been well communicated
                    sig = self.server.recv_msg(controllerd_pb2.Signal())
                    self.assertIsNot(sig, None, "The status of the report never got updated")
                    self.assertTrue(sig.HasField("reports"))
                    for report in sig.reports.reports:
                        if report.name == cmd.set_work.report.name:
                            self.assertEqual(report.state_disk, "DONE")

                # Check that a commit has been made after running schedule_enhancements
                sig = self.server.recv_msg(controllerd_pb2.Signal())
                self.assertIsNot(sig, None)
                self.assertTrue(sig.HasField("report_pushed"), sig)
                report_commits_count += 1

        with self.subTest("Queue work: check reboot"):
            # Check that the "reboot_needed" hook does generate a signal to the
            # controller
            sbench = MockupSmartEzbenchSingle(ezbench_dir, report_fullname)

            # Schedule the reboot test
            ret, sigs = self._queue_work("report1", self.dut_repo_commit1, "test_reboot", 1)
            self.assertEqual(ret.err_code, controllerd_pb2.CmdStatus.OK)
            self.assertEqual(len(sigs), 3, "Got the signals: {}".format(sigs))
            sigs = self._wait_for_run_to_be_done(sbench, ['report_pushed', 'reboot'])

            # Verify that we got the expected signals
            self.assertEqual(sigs[0].report_pushed.report, "report1")
            self.assertLess(datetime.utcnow().timestamp() - sigs[1].reboot.timestamp, 0.2)

            # Let the run go through
            self.assertFalse(sbench.run_barrier.is_set())
            sbench.run_barrier.set()
            sbench.schedule_enhancements_barrier.set()
            self.assertTrue(sbench.schedule_enhancements_called_event.wait(0.5))
            self.assertFalse(sbench.run_barrier.is_set())
            sbench.schedule_enhancements_called_event.clear()

            # Increase the number of commits we are supposed to have
            report_commits_count += 2

            # Verify that the report is now said to be in the DONE state
            sig = self.server.recv_msg(controllerd_pb2.Signal())
            self.assertIsNot(sig, None, "The status of the report never got updated")
            self.assertTrue(sig.HasField("reports"))
            for report in sig.reports.reports:
                if report.name == cmd.set_work.report.name:
                    self.assertEqual(report.state_disk, "DONE")

        with self.subTest("Queue work: higher priority job coming"):
            pass

        with self.subTest("Queue work: raising priority of a job"):
            pass

        with self.subTest("Report: remote report deleted before pull"):
            pass
            shutil.rmtree(dutd.Configuration.controller_reports_base_url)

            ret, sigs = self._queue_work("report1", self.dut_repo_commit1, "test1", 1)
            self.assertEqual(ret.err_code, controllerd_pb2.CmdStatus.ERROR)
            self.assertEqual(ret.err_msg, "The report cannot be created: git fetch failure")
            self.assertEqual(len(sigs), 0, "Got the signals: {}".format(sigs))

        with self.subTest("Report: non-forwadable pull and push"):
            # Make a new commit
            self.report_repo = GitRepoFactory(dutd.Configuration.controller_reports_base_url, True)
            spurious_commit1 = self.report_repo.create_commit("new_initial_commit",
                                                              [("noise", "noise1\n")])

            # Verify that there is only one branch, master
            branches = self.report_repo.src_repo.listall_branches()
            self.assertEqual(branches, ["master"])
            self.assertEqual(len(self._report_git_history()), 1)

            # Queue some work
            ret, sigs = self._queue_work("report1", self.dut_repo_commit1, "test1", 1)
            self.assertEqual(ret.err_code, controllerd_pb2.CmdStatus.OK)
            self.assertEqual(len(sigs), 4, "Got the signals: {}".format(sigs))

            sbench = MockupSmartEzbenchSingle(ezbench_dir, report_fullname)
            sigs = self._wait_for_run_to_be_done(sbench, ['report_pushed'])

            # Check that the report name is right
            self.assertEqual(sigs[0].report_pushed.report, "report1")

            # Check that the Dutd indeed pushed two branches
            branches = self.report_repo.src_repo.listall_branches()
            self.assertEqual(len(branches), 2)

            # Check that master has the right amount of commits now in master
            # and the saved masted
            self.assertEqual(len(self._report_git_history()), 3)
            self.assertEqual(len(self._report_git_history(branches[1])),
                             report_commits_count + 1)

            # Let the run go through
            self.assertFalse(sbench.run_barrier.is_set())
            sbench.run_barrier.set()
            sbench.schedule_enhancements_barrier.set()
            self.assertTrue(sbench.schedule_enhancements_called_event.wait(0.5))
            self.assertFalse(sbench.run_barrier.is_set())
            sbench.schedule_enhancements_called_event.clear()

            # Verify that the report is now said to be in the DONE state
            sig = self.server.recv_msg(controllerd_pb2.Signal())
            self.assertIsNot(sig, None, "The status of the report never got updated")
            self.assertTrue(sig.HasField("reports"))
            for report in sig.reports.reports:
                if report.name == cmd.set_work.report.name:
                    self.assertEqual(report.state_disk, "DONE")

        with self.subTest("Report: local report deleted before pull"):
            # Save the current git history, so we can compare it after the test
            ref_history = self._report_git_history()

            # Delete the local folder
            sbench = MockupSmartEzbenchSingle(ezbench_dir, report_fullname)

            # Give time to whatever libgit is doing in the background to finish
            time.sleep(.05)
            shutil.rmtree(sbench.log_folder)

            # Re-queue the same work as before and check that it went back to
            # the previous original state
            ret, sigs = self._queue_work("report1", self.dut_repo_commit1, "test1", 1)
            self.assertEqual(ret.err_code, controllerd_pb2.CmdStatus.OK)
            self.assertEqual(len(sigs), 4, "Got the signals: {}".format(sigs))
            self._wait_for_run_to_be_done(sbench, check_signals=False)

            # Make sure that the state is matching the request and not what was
            # previously found in the report upstream
            self.assertEqual(sbench.state["commits_user"],
                            {self.dut_repo_commit1: {'tests': {'test1': {'rounds': 1}}}})
            self.assertEqual(sbench.state["commits_auto"], dict())

            # Check that the head of the reference history is still found in the
            # current git history
            self.assertIn(ref_history[0][0], [c[0] for c in self._report_git_history()])

            # Let the run go through
            self.assertFalse(sbench.run_barrier.is_set())
            sbench.run_barrier.set()
            sbench.schedule_enhancements_barrier.set()
            self.assertTrue(sbench.schedule_enhancements_called_event.wait(0.5))
            self.assertFalse(sbench.run_barrier.is_set())
            sbench.schedule_enhancements_called_event.clear()

            # Verify that the report is now said to be in the DONE state
            sig = self.server.recv_msg(controllerd_pb2.Signal())
            self.assertIsNot(sig, None, "The status of the report never got updated")
            self.assertTrue(sig.HasField("reports"))
            for report in sig.reports.reports:
                if report.name == cmd.set_work.report.name:
                    self.assertEqual(report.state_disk, "DONE")

        with self.subTest("Report: remote report deleted before push"):
            # Make sure that the DUT keeps on running if the controller's
            # report gets deleted or is non-forwadable
            sbench = MockupSmartEzbenchSingle(ezbench_dir, report_fullname)

            # Check that the current state actually includes test1
            self.assertEqual(sbench.state["commits_user"],
                            {self.dut_repo_commit1: {'tests': {'test1': {'rounds': 1}}}})

            # Queue 3 runs, so we can check that all of them got executed
            ret, sigs = self._queue_work("report1", self.dut_repo_commit1, "test2", 3)
            self.assertEqual(ret.err_code, controllerd_pb2.CmdStatus.OK)
            self.assertEqual(len(sigs), 4, "Got the signals: {}".format(sigs))

            # Delete the upstream repo
            shutil.rmtree(dutd.Configuration.controller_reports_base_url)

            # Wait for the work to be done
            self._wait_for_run_to_be_done(sbench)

            # Make sure that the state is matching the request and not what was
            # previously found in the report upstream
            self.assertEqual(sbench.state["commits_user"],
                            {self.dut_repo_commit1: {'tests': {'test2': {'rounds': 3}}}})
            self.assertEqual(sbench.state["commits_auto"], dict())

            # Let the run go through
            self.assertFalse(sbench.run_barrier.is_set())
            sbench.run_barrier.set()
            sbench.schedule_enhancements_barrier.set()
            self.assertTrue(sbench.schedule_enhancements_called_event.wait(0.5))
            self.assertFalse(sbench.run_barrier.is_set())
            sbench.schedule_enhancements_called_event.clear()

            # Verify that the report is now said to be in the DONE state
            sig = self.server.recv_msg(controllerd_pb2.Signal())
            self.assertIsNot(sig, None, "The status of the report never got updated")
            self.assertTrue(sig.HasField("reports"))
            for report in sig.reports.reports:
                if report.name == cmd.set_work.report.name:
                    self.assertEqual(report.state_disk, "DONE")

        with self.subTest("Delete job: Missing job"):
            cmd = controllerd_pb2.Cmd()
            cmd.delete_report.name = "missing_report"

            # Get the current count of reports
            report_count = len(MockupSmartEzbenchSingle.list_reports(ezbench_dir))

            ret, sigs = self.server.send_cmd(cmd)
            self.assertEqual(ret.err_code, controllerd_pb2.CmdStatus.ERROR)
            self.assertEqual(ret.err_msg, "Could not find the report missing_report")
            self.assertEqual(len(sigs), 0, "Got the signals: {}".format(sigs))

            # Check that no report is deleted
            self.assertEqual(report_count, len(MockupSmartEzbenchSingle.list_reports(ezbench_dir)))

        with self.subTest("Delete job: Job not created by us"):
            cmd = controllerd_pb2.Cmd()
            cmd.delete_report.name = "invalid1"

            # Get the current count of reports
            report_count = len(MockupSmartEzbenchSingle.list_reports(ezbench_dir))

            ret, sigs = self.server.send_cmd(cmd)
            self.assertEqual(ret.err_code, controllerd_pb2.CmdStatus.ERROR)
            self.assertEqual(ret.err_msg, "Could not find the report invalid1")
            self.assertEqual(len(sigs), 0, "Got the signals: {}".format(sigs))

            # Check that no report is deleted
            self.assertEqual(report_count, len(MockupSmartEzbenchSingle.list_reports(ezbench_dir)))

        with self.subTest("Delete job: Job created by us"):
            cmd = controllerd_pb2.Cmd()
            cmd.delete_report.name = "report1"

            ret, sigs = self.server.send_cmd(cmd)
            self.assertEqual(ret.err_code, controllerd_pb2.CmdStatus.OK)
            self.assertEqual(ret.err_msg, str())

            # Make sure only the controller's job got deleted
            report1 = MockupSmartEzbenchSingle.reports["{}/report1".format(self.server.controller_name)]
            report1_out = MockupSmartEzbenchSingle.reports["report1"]
            self.assertTrue(report1._deleted)
            self.assertFalse(report1_out._deleted)
            self.assertEqual(len(sigs), 2, "Got the signals: {}".format(sigs))

            # Check that the report is not advertised anymore
            self.check_set_work_sigs(sigs)

        with self.subTest("Check all the logs"):
            # Check that there is the right information in the logs
            self.assertEqual(self.server.dutd_logs.get("report1", []),
                             ["Deleting the report"])
