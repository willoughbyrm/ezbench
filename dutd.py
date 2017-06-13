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

from collections import deque
import configparser
import statistics
import threading
import traceback
import argparse
import socket
import struct
import signal
import errno
import time
import sys
import os

from datetime import datetime

ezbench_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ezbench_dir, 'python-modules'))
from ezbench.smartezbench import *

sys.path.append(os.path.join(ezbench_dir, 'protocols'))
import controllerd_pb2

# TODO: Move timing_DB to the python modules
timings_db_dir = os.path.join(ezbench_dir, 'timing_DB')
sys.path.append(timings_db_dir)
import timing

class ControllerClient(threading.Thread):
    send_lock = threading.Lock()

    def __init__(self, ezbench_dir, timings_db_dir,
                 smartezbench_class=SmartEzbench, runner_class=Runner,
                 timingsdb_class=TimingsDB, testset_class=Testset):
        self.ezbench_dir = ezbench_dir
        self._timingsdb = timingsdb_class(timings_db_dir)

        self.smartezbench_class = smartezbench_class
        self.runner_class = runner_class
        self.testset_class = testset_class

        self.report_upload_credential = self.gen_credentials()

        self.socket = None
        self.ready_to_send = False
        self.controller_name = None

        self._sbenches_lock = threading.Lock()
        self._sbenches = dict()
        self._current_sbench = None
        self._report_exec_order = []

        self._tests_lock = threading.Lock()
        self._tests = dict()
        self.read_tests()

        self._testsets_lock = threading.Lock()
        self._testsets = dict()

        self._exit_now = False
        self._run_reports_event = threading.Event()

        self._cmd_queue = deque()
        super().__init__(target=self.serve_forever)

    def log(self, criticality, message):
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = "{time}: ({criticality}) {msg}\n".format(time=time,
                                                           criticality=criticality.name,
                                                           msg=message)
        print(log_msg, end="")

    def gen_credentials(self):
        user = Configuration.credentials_user
        if (Configuration.credentials_ssh_key_priv is not None and
           Configuration.credentials_ssh_key_pub is not None):
            priv_key_path = Configuration.credentials_ssh_key_priv
            pub_key_path = Configuration.credentials_ssh_key_pub
            if not os.path.exists(priv_key_path):
                self.log(Criticality.EE, "Report push auth: The private key '{}' does not exist".format(priv_key_path))
                sys.exit(1)
            if not os.path.exists(pub_key_path):
                self.log(Criticality.EE, "Report push auth: The public key '{}' does not exist".format(pub_key_path))
                sys.exit(1)

            self.log(Criticality.II, "Report push auth: will use the ssh key pair {} - {} with user {}".format(priv_key_path, pub_key_path, user))
            return pygit2.Keypair(user, pub_key_path, priv_key_path,
                                  Configuration.credentials_ssh_key_pass)
        elif Configuration.credentials_password is not None:
            self.log(Criticality.II, "Report push auth: will use the provided password with user {}".format(user))
            return pygit2.UserPass(user, Configuration.credentials_password)
        else:
            self.log(Criticality.II, "Report push auth: will use the ssh agent with user {}".format(user))
            return pygit2.KeypairFromAgent(user)

    def __create_smartezbench__(self, report):
        return self.smartezbench_class(self.ezbench_dir, report,
                            logs_callback=self.send_logs,
                            hooks_callback=self.sbench_hooks_callback)

    def read_tests(self):
        runner = self.runner_class(self.ezbench_dir)
        with self._tests_lock:
            for test in runner.list_tests():
                self._tests[test.name] = test

    def compute_report_exec_order(self):
        with self._sbenches_lock:
            runnable_reports = []
            for name in self._sbenches:
                mode = self._sbenches[name].running_mode(check_running=False)
                if mode == RunningMode.RUN or mode == RunningMode.PAUSE:
                    runnable_reports.append(name)

            pl = sorted(runnable_reports,
                        key=lambda sbench: self._sbenches[sbench].attribute("report_priority"))

            # If the currently-run report has the top priority, make sure that we
            # keep running it without interruption
            if (self._current_sbench is not None and
                self._current_sbench in self._report_exec_order):
                cur_prio = self._current_sbench.attribute("report_priority")
                top_prio = self._sbenches[self._report_exec_order[0]].attribute("report_priority")

                if cur_prio == top_prio:
                    pl.remove(self._current_sbench.report_name)
                    pl.insert(0, self._current_sbench.report_name)

            # We have the updated list, store it to an attribute
            self._report_exec_order = pl

            # Unblock the main thread, if we have some reports to run
            if len(self._report_exec_order) > 0:
                self._run_reports_event.set()

    def refresh_reports_state(self):
        reports = self.smartezbench_class.list_reports(self.ezbench_dir)

        sbenches = dict()
        msg_sig = controllerd_pb2.Signal()
        msg_avail_reports = msg_sig.reports
        for report in reports:
            try:
                sbenches[report] = sbench = self.__create_smartezbench__(report)

                if (self.controller_name is None or
                    sbench.user_data("controller_name", None) != self.controller_name):
                    continue

                profile = sbenches[report].profile()
                if profile is None:
                    profile = ""

                msg_report = msg_avail_reports.reports.add()
                msg_report.name = sbench.user_data("report_name")
                msg_report.profile = profile

                msg_report.state = sbenches[report].running_mode().name
                msg_report.state_disk = sbenches[report].running_mode(False).name

                samples = self._timingsdb.data("build", msg_report.profile)
                if len(samples) > 0:
                    msg_report.build_time = statistics.median(samples)
                else:
                    msg_report.build_time = 5 * 60 # default to 5 minutes

                samples = self._timingsdb.data("deploy", msg_report.profile)
                if len(samples) > 0:
                    msg_report.deploy_time = statistics.median(samples)
                else:
                    msg_report.deploy_time = 2 * 60 # default to 2 minutes

            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                sys.stderr.write("\n")
                pass

        # Work around a bug in protobuf
        msg_avail_reports.report_count = len(msg_avail_reports.reports)

        # Try sending the list of reports to the controller
        try:
            self.send_msg(msg_sig)
        except:
            pass

        return sbenches

    def update_reports_list(self):
        sbenches = self.refresh_reports_state()

        # update the list
        with self._sbenches_lock:
            self._sbenches = sbenches

        self.compute_report_exec_order()

    def update_test_list(self):
        tests = set()
        msg_sig = controllerd_pb2.Signal()
        msg_avail_tests = msg_sig.tests
        with self._tests_lock:
            msg_avail_tests.tests_count = len(self._tests)
            for test in self._tests:
                msg_test = msg_avail_tests.tests.add()
                msg_test.name = test

                samples = self._timingsdb.data("test", test)
                if len(samples) > 0:
                    msg_test.exec_time_median = statistics.median(samples)
                    msg_test.exec_time_max = max(samples)
                else:
                    msg_test.exec_time_median = self._tests[test].time_estimation
                    msg_test.exec_time_max = self._tests[test].time_estimation

                tests.add(test)

        self.send_msg(msg_sig)
        return tests


    def update_testset_list(self, available_tests):
        msg_sig = controllerd_pb2.Signal()
        msg_avail_testsets = msg_sig.testsets

        with self._testsets_lock:
            self._testsets = dict()
            testsets = self.testset_class.list(self.ezbench_dir)
            msg_avail_testsets.testsets_count = 0
            for testset in testsets:
                if not testset.parse(available_tests, silent=True):
                    continue

                # Keep a copy of the testset
                self._testsets[testset.name] = testset

                msg_avail_testsets.testsets_count += 1
                msg_testset = msg_avail_testsets.testsets.add()
                msg_testset.name = testset.name
                msg_testset.description = testset.description
                for test in testset:
                    msg_test = msg_testset.tests.add()
                    msg_test.name = test
                    msg_test.rounds = testset[test]

        self.send_msg(msg_sig)


    def send_msg(self, msg, force_send=False):
        buf = msg.SerializeToString()
        if len(buf) == 0:
            print("WARNING: Trying to send 0 bytes!")
            traceback.print_stack()
        #print("----- send msg: buf =", buf)
        pack1 = struct.pack('>I', len(buf))
        with self.send_lock:
            if self.ready_to_send or force_send:
                self.socket.sendall(pack1 + buf)

    def send_logs(self, sbench, msg):
        msg_sig = controllerd_pb2.Signal()

        # Get the report's name, exit if it is not available as we do not have
        # a valid report created by dutd
        report = sbench.user_data("report_name", None)
        if report is None:
            return

        msg_sig.log.report = report
        msg_sig.log.msg = msg
        self.send_msg(msg_sig)

    def recv_msg(self, msg):
        msg_len_buf = self.socket.recv(4, socket.MSG_WAITALL)
        if len(msg_len_buf) != 4:
            return None

        msg_len = struct.unpack('>I', msg_len_buf)[0]
        buf = self.socket.recv(msg_len, socket.MSG_WAITALL)

        if len(buf) == msg_len:
            msg.ParseFromString(buf)
            return msg
        else:
            return None

    def connect(self):
        # Reset the state to say that we are not ready to send
        with self.send_lock:
            self.ready_to_send = False
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect to the controller
        self.socket.connect(Configuration.controller_host)
        print("Handshake with {}:{} initiated".format(Configuration.controller_host[0],
                                                      Configuration.controller_host[1]))

        # Send the hello message
        hello = controllerd_pb2.ClientHello()
        hello.version = 1
        hello.machine_name = Configuration.machine_name
        self.send_msg(hello, force_send=True)

        # Read the server Hello message
        hello = self.recv_msg(controllerd_pb2.ServerHello())
        if hello is None:
            self.socket.close()
            raise IOError("The server did not send its hello message after ours")
        self.controller_name = hello.controller_name
        print("Handshake with controller '{}' succeeded - protocol version {}".format(hello.controller_name, hello.version))

        with self.send_lock:
            self.ready_to_send = True

        # Send the current state to the controller
        testlist = self.update_test_list()
        self.update_testset_list(testlist)
        self.update_reports_list()

    def serve_forever(self):
        while not self._exit_now:
            try:
                self.connect()
            except ConnectionRefusedError:
                print("The controller is not available, re-trying in {} seconds...".format(Configuration.controllerd_reconnection_period))
                time.sleep(Configuration.controllerd_reconnection_period)
                self.socket.close()
                continue
            except (IOError, BrokenPipeError):
                print("The controller got disconnected, re-trying in {} seconds...".format(Configuration.controllerd_reconnection_period))
                self.socket.close()
                time.sleep(Configuration.controllerd_reconnection_period)
                continue
            except Exception as e:
                traceback.print_exc()
                self.socket.close()
                time.sleep(1)
                continue

            try:
                while not self._exit_now:
                    cmd = self.recv_msg(controllerd_pb2.Cmd())
                    if cmd is None:
                        break

                    # Answer ping commands immediately
                    if cmd.HasField("ping"):
                        sig = controllerd_pb2.Signal()
                        sig.cmd_status.id = cmd.id
                        sig.cmd_status.err_code = controllerd_pb2.CmdStatus.OK
                        self.send_msg(sig)
                    else:
                        self._cmd_queue.append(cmd)
                        self._run_reports_event.set()

            except IOError as e:
                pass
            except Exception as e:
                traceback.print_exc()

            try:
                self.socket.shutdown(socket.SHUT_RDWR)
                self.socket.close()
            except IOError as e:
                pass
            except Exception as e:
                print(e)

            # Exit immediately if it was requested, wait before trying to
            # reconnected otherwise
            if self._exit_now:
                return
            else:
                time.sleep(Configuration.controllerd_reconnection_period)

    def __report_set_attributes__(self, sbench, attributes):
        # Go through the attributes and set them all
        for attr in attributes:
            for name in attr.DESCRIPTOR.fields_by_name:
                if attr.HasField(name):
                    sbench.set_attribute(name, getattr(attr, name))

    def __report_remote_name__(self):
        return "controllerd/{}".format(self.controller_name.replace(" ", "_"))

    def __push_reference__(self, sbench, repo, reference):
        try:
            remote = repo.remotes[self.__report_remote_name__()]
            git_cb = pygit2.RemoteCallbacks(credentials=self.report_upload_credential)
            remote.push([reference], git_cb)

            # Tell that the report got pushed
            sig = controllerd_pb2.Signal()
            sig.report_pushed.report = sbench.user_data("report_name")
            self.send_msg(sig)
        except Exception as e:
            self.log(Criticality.EE, traceback.format_exc())

    # Create git objects for all files and directories in the given directory.
    # Returns None if the directory is empty (as git cannot represent empty directories),
    # the oid of the created tree otherwise
    def __directory_to_git_tree(self, repo, dirname):
        tb = repo.TreeBuilder()
        ret = False

        for fpath in glob.glob("{}/*".format(dirname), recursive=False):
            try:
                if os.path.isdir(fpath):
                    subtree = self.__directory_to_git_tree(repo, fpath)
                    if subtree is not None:
                        tb.insert(os.path.basename(fpath),
                                  self.__directory_to_git_tree(repo, fpath),
                                  pygit2.GIT_FILEMODE_TREE)
                        ret = True
                else:
                    with open(fpath, "rb") as f:
                        data_oid = repo.create_blob(f.read())
                        path = os.path.basename(fpath)
                        tb.insert(path, data_oid, pygit2.GIT_FILEMODE_BLOB)
                        ret = True
            except IOError as e:
                print("Warning: Cannot open the file '{}'".format(fpath))
                continue
        if ret:
            return tb.write()
        else:
            return None

    def __commit_report__(self, sbench, task, push=True):
        repo = pygit2.init_repository(sbench.log_folder, False)

        # Create the new tree
        treeoid = self.__directory_to_git_tree(repo, sbench.log_folder)
        new_tree = repo[treeoid]

        # Abort early if nothing changed
        branch = repo.lookup_branch("master")
        if branch is not None:
            diff = repo[branch.target].tree.diff_to_tree(new_tree)
            if diff.patch is None or len(diff.patch) == 0:
                return str(repo[branch.target].hex)

        author = pygit2.Signature("EzBench DUT {}".format(Configuration.machine_name),
                                    "dutd@{}".format(Configuration.machine_name))

        reference = "refs/heads/master"
        try:
            ref = repo.lookup_reference(reference)
            parents = [ref.target]
        except:
            parents = []
        commit_id = repo.create_commit(reference, author, author, str(task),
                                       new_tree.hex, parents)
        commit = repo.revparse_single(str(commit_id))

        try:
            branch = repo.create_branch("master", commit, False)
        except:
            branch = repo.lookup_branch("master")
        branch.set_target(str(commit_id))

        # TODO: do not push if the connection is not ready
        if push:
            self.__push_reference__(sbench, repo, reference)

        # Now checkout the newer version
        try:
            repo.checkout("HEAD")
        except Exception as e:
            print("Could not checkout the latest commit: " + str(e))

        return str(commit_id)

    def __fetch_repo__(self, repo_path, remote_name, remote_url, fetch_ref):
        try:
            repo = pygit2.Repository(repo_path)
        except Exception as e:
            raise ValueError("Error: cannot open the GIT repo '{}'".format(repo_path)) from e

        # Since I don't know how to update a remote's URL, let's destroy
        # it before re-creating it with the right URL
        try:
            try:
                repo.remotes.delete(remote_name)
            except:
                pass
            remote = repo.remotes.create(remote_name, remote_url)
        except Exception as e:
            raise ValueError("ERROR: could not create the remote '{}'('{}')".format(remote_name, remote_url)) from e

        # Fetch from the controller
        try:
            git_cb = pygit2.RemoteCallbacks(credentials=self.report_upload_credential)
            self.log(Criticality.II, "Git fetch the refs {} from {}".format(fetch_ref, remote_url))
            ret = remote.fetch([fetch_ref], callbacks=git_cb)
            self.log(Criticality.II, "Git fetch complete")
        except Exception as e:
            err = "ERROR: could not fetch the remote {}({})"
            raise ValueError(err.format(remote_name, remote_url)) from e

    def handle_command_queue(self):
        while not self._exit_now:
            try:
                cmd = self._cmd_queue.popleft()
            except IndexError:
                #print("handle_command_queue: Nothing to dequeu, DONE!")
                return

            # Prepare the ACK to the command
            sig = controllerd_pb2.Signal()
            sig.cmd_status.id = cmd.id
            sig.cmd_status.err_code = controllerd_pb2.CmdStatus.OK

            try:
                if cmd.HasField("delete_report"):
                    with self._sbenches_lock:
                        report_name = "{}/{}".format(self.controller_name, cmd.delete_report.name)
                        sbench = self._sbenches.get(report_name, None)
                        if sbench is None:
                            sig.cmd_status.err_code = controllerd_pb2.CmdStatus.ERROR
                            sig.cmd_status.err_msg = "Could not find the report " + cmd.delete_report.name
                        else:
                            sbench.delete()

                    if sig.cmd_status.err_code == controllerd_pb2.CmdStatus.OK:
                        self.update_reports_list()
                elif cmd.HasField("set_work"):
                    self.__set_work__(cmd.set_work, sig)
                else:
                    sig.cmd_status.err_code = controllerd_pb2.CmdStatus.ERROR
                    sig.cmd_status.err_msg = "Could not find any action to do. Software needs to be updated?"
            except Exception as e:
                traceback.print_exc()
                sig.cmd_status.err_code = controllerd_pb2.CmdStatus.ERROR
                sig.cmd_status.err_msg = "Caught an unexpected exception"

            # Now send the cmd_status!
            self.send_msg(sig)

    def sbench_hooks_callback(self, state):
        if state.action == "start_running_tests":
            self.refresh_reports_state()
        elif state.action == "done_running_test":
            # First, make a commit of all the changes and push them if possible
            self.__commit_report__(state.sbench, state.hook_parameters['task'])

            # Handle all the commands we received, to verify if we need to stop
            # doing anything
            self.handle_command_queue()
        elif state.action == "done_running_tests":
            self.refresh_reports_state()
        elif state.action == "reboot_needed":
            msg_sig = controllerd_pb2.Signal()
            msg_sig.reboot.timestamp = datetime.utcnow().timestamp()
            self.send_msg(msg_sig)
        elif state.action == "mode_changed":
            # Do not send the change in state if we are exiting because the
            # pausing is just a way to stop the execution
            if not self._exit_now:
                self.refresh_reports_state()

    def __set_report_in_valid_state__(self, sbench, upload_url, sig):
        # Open the report's git repo
        try:
            repo = pygit2.Repository(sbench.log_folder)
            commit_msg = "New report parameters"
        except (pygit2.GitError, KeyError):
            try:
                repo = pygit2.init_repository(sbench.log_folder, False)
                commit_msg = "Initial creation on the DUT side"
            except OSError as e:
                sig.cmd_status.err_code = controllerd_pb2.CmdStatus.ERROR
                sig.cmd_status.err_msg = "The report cannot be created: git init failure"
                return None

        # Pull the report from the server
        url = "{}/{}".format(Configuration.controller_reports_base_url,
                            upload_url)
        try:
            self.__fetch_repo__(sbench.log_folder,
                                self.__report_remote_name__(), url,
                                '+refs/*:refs/controllerd/*')
        except ValueError as e:
            #traceback.print_exc()
            sig.cmd_status.err_code = controllerd_pb2.CmdStatus.ERROR
            sig.cmd_status.err_msg = "The report cannot be created: git fetch failure"
            return None

        # Make a master branch the remote's master's head
        remote_head = repo.revparse_single("{}/master".format(self.__report_remote_name__()))
        branch = repo.lookup_branch("master")
        if branch is None:
            # The branch does not exist, so create it and checkout the files
            branch = repo.create_branch("master", remote_head, False)
            repo.checkout(branch)
        else:
            # The master branch already exists, make sure that it is
            # fast-forwardable by walking down our local branch's HEAD and
            # stopping only when we find $remote_head
            walker = repo.walk(branch.target)
            fast_forwardable = False
            for commit in walker:
                if str(commit.oid) == str(remote_head.oid):
                    fast_forwardable = True
                    break

            # If we are not fast-forwardable, save the current state to a
            # branch (to not lose data), and then reset the state of master to
            # the remote master
            if not fast_forwardable:
                # Commit any change that would otherwise be lost
                latest = self.__commit_report__(sbench,
                                                "Last commit before switching to the new report",
                                                push=False)

                # Create a new branch that will contain the previous state
                branch_name = "master_sav_" + str(datetime.utcnow().timestamp())
                sav_branch = repo.create_branch(branch_name, repo[latest], False)

                # Reset the branch to the current view
                branch = branch = repo.lookup_branch("master")
                branch.set_target(str(remote_head.oid))

                # Push the new branch
                self.__push_reference__(sbench, repo, "refs/heads/" + branch_name)

                # Get rid of all the files now everything is safe in git
                for entry in os.scandir(path=sbench.log_folder):
                    if entry.is_dir():
                        if entry.name == ".git":
                            continue
                        else:
                            shutil.rmtree(entry.path)
                    else:
                        os.unlink(entry.path)

                # Make the checkouted files match the current state of the master branch
                repo.checkout(repo.lookup_branch("master"), strategy=pygit2.GIT_CHECKOUT_FORCE)

        return commit_msg

    def __create_report__(self, creport, sig):
        name = "{}/{}".format(self.controller_name, creport.name)
        sbench = self.__create_smartezbench__(name)

        # Make sure that we are in a fast-forwardable state before making any
        # change to the report.
        msg = self.__set_report_in_valid_state__(sbench,
                                                 sbench.user_data("upload_url", creport.upload_url),
                                                 sig)
        if msg is None:
            return None, str()

        if sbench.user_data("upload_url", creport.upload_url) != creport.upload_url:
            sig.cmd_status.err_code = controllerd_pb2.CmdStatus.ERROR
            sig.cmd_status.err_msg = "The upload_url cannot be changed"
            return None, str()
        else:
            sbench.set_user_data("upload_url", creport.upload_url)

        if sbench.profile() is None:
            if not sbench.set_profile(creport.profile):
                sig.cmd_status.err_code = controllerd_pb2.CmdStatus.ERROR
                sig.cmd_status.err_msg = "The profile '{}' could not be set".format(creport.profile)
                return None, str()

        sbench.set_user_data("description", creport.description)
        sbench.set_user_data("controller_name", self.controller_name)
        sbench.set_user_data("report_name", creport.name)
        self.__report_set_attributes__(sbench, creport.attributes)

        return sbench, msg

    def __set_work__(self, work, sig):
        # Create the report and reset the work queue
        sbench, commit_msg = self.__create_report__(work.report, sig)
        if sbench is None:
            return

        # Reset the current state of the report
        sbench.reset_work()

        # Fetch the latest changes in the repo
        try:
            self.__fetch_repo__(sbench.repo().repo_path,
                                self.__report_remote_name__(),
                                Configuration.controller_git_repo_url,
                                '+refs/*:refs/controllerd/*')
        except Exception as e:
            # TODO: Do the right thing here
            traceback.print_exc()

        # Queue the work
        repo = pygit2.Repository(sbench.repo().repo_path)
        for commit in work.commits:
            # Check that the commit exists
            try:
                c = repo.revparse_single(commit.id)
            except KeyError:
                msg = "Commit {} does not exist, ignore work queued on it\n\n".format(commit.id)
                sig.cmd_status.err_msg += msg
                continue

            with self._tests_lock:
                has_seen_error = False
                for cmd_test in commit.tests:
                    try:
                        test = self._tests[cmd_test.name]
                    except:
                        if not has_seen_error:
                            sig.cmd_status.err_msg += "Commit {}:\n\tMissing tests:\n".format(commit.id)
                        sig.cmd_status.err_msg += "\t - {}\n".format(cmd_test.name)
                        has_seen_error = True
                        continue
                    sbench.add_test(commit.id, cmd_test.name, cmd_test.rounds)

                if has_seen_error:
                    sig.cmd_status.err_msg += "\n"

            with self._testsets_lock:
                has_seen_error = False
                for cmd_testset in commit.testsets:
                    try:
                        testset = self._testsets[cmd_testset.name]
                    except:
                        if not has_seen_error:
                            sig.cmd_status.err_msg += "\tMissing testsets:\n"
                        sig.cmd_status.err_msg += " - {}\n".format(cmd_testset.name)
                        has_seen_error = True
                        continue
                    sbench.add_testset(commit.id, testset, cmd_testset.rounds)

                if has_seen_error:
                    sig.cmd_status.err_msg += "\n"

            if len(sig.cmd_status.err_msg) > 0:
                sig.cmd_status.err_code = controllerd_pb2.CmdStatus.WARNING

        if len(work.commits) > 0:
            sbench.set_running_mode(RunningMode.RUN)

        self.__commit_report__(sbench, commit_msg)

        self.update_reports_list()

    def sbench_run(self, report_name):
        sbench = self.smartezbench_class(self.ezbench_dir, report_name)
        sbench.schedule_enhancements()

    def run_reports(self):
        while not self._exit_now:
            # Handle all the commands that are available
            self.handle_command_queue()

            with self._sbenches_lock:
                try:
                    next_report = self._report_exec_order[0]
                    self._current_sbench = self._sbenches[next_report]
                except Exception as e:
                    self._run_reports_event.clear()
                    self._current_sbench = None

            # If there were no reports to run, just wait for work
            if self._current_sbench is None:
                self._run_reports_event.wait()
                continue

            # Run the report
            try:
                self._current_sbench.run()

                if Configuration.report_parsing_in_separate_process:
                    # Run the report generation in a separate process because python
                    # is really bad at freeing memory
                    p = multiprocessing.Process(target=self.sbench_run,
                                        args=(self._current_sbench.report_name,))
                    p.start()
                    p.join()
                else:
                    self.sbench_run(self._current_sbench.report_name)

                self.__commit_report__(self._current_sbench, "Scheduled enchancements")

                # If we are DONE, then remove the report from the list of reports
                # to execute
                if self._current_sbench.running_mode() == RunningMode.DONE:
                    with self._sbenches_lock:
                        self._report_exec_order.pop(0)

            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                sys.stderr.write("\n")

    def stop(self):
        self._exit_now = True
        self._run_reports_event.set()
        self._exit_now_req = time.monotonic()

        # Stop the current execution
        with self._sbenches_lock:
            if self._current_sbench is not None:
                self._current_sbench.set_running_mode(RunningMode.PAUSE)

        # Reset the running mode of the current sbench to RUN
        with self._sbenches_lock:
            if self._current_sbench is not None:
                self._current_sbench.set_running_mode(RunningMode.RUN)

        # Close the connection with the controller
        # Ignore errors when running shutdown, as the socket may not be connected
        try:
            self.socket.shutdown(socket.SHUT_RD)
        except OSError as e:
            if e.errno != errno.ENOTCONN and e.errno != errno.EBADF:
                raise e

        # Wait for the controller communication's thread to be done
        self.join()

        # Make sure the socket to the controller has been closed properly
        try:
            self.socket.close()
        except:
            pass

class Configuration:
    controllerd_reconnection_period = 5
    report_parsing_in_separate_process = True

    @classmethod
    def parse_error(cls, msg):
        print("Parsing error: " + msg, file=sys.stderr)
        sys.exit(1)

    @classmethod
    def format_error(cls, category, key, found, expected_format):
        msg = "Invalid format for the field [{}].{}: found '{}', expected {}"
        cls.parse_error(msg.format(category, key, found, expected_format))

    @classmethod
    def parse_string(cls, config, category, key, default=None, optional=False):
        try:
            val = config[category][key]
        except:
            if default is None and not optional:
                cls.parse_error("Field [{}].{} is missing".format(category, key))
            else:
                return default

        return val

    @classmethod
    def parse_bind_addr(cls, config, category, key):
        val = cls.parse_string(config, category, key)

        fields = val.split(":")
        if len(fields) == 2:
            try:
                port = int(fields[1])
            except:
                cls.format_error(category, key, fields[1], "an integer")

            if port < 0 or port > 65535:
                cls.format_error(category, key, port, "0 < port < 65536".format())

            return (fields[0], port)

        else:
            cls.format_error(category, key, "host:port")

    @classmethod
    def load_config(cls, path):
        config = configparser.ConfigParser()
        config.read(path)

        for category in ["General", "Controller", "ControllerCredentials", "Repos"]:
            if category not in config:
                cls.parse_error("The category '{}' is missing".format(category))

        cls.machine_name = cls.parse_string(config, "General", "Name", socket.gethostname())

        cls.controller_host = cls.parse_bind_addr(config, "Controller", "Host")
        cls.controller_reports_base_url = cls.parse_string(config, "Controller", "ReportsBaseURL")
        cls.controller_git_repo_url = cls.parse_string(config, "Controller", "GitRepoURL")

        cls.credentials_user = cls.parse_string(config, "ControllerCredentials", "User")
        cls.credentials_ssh_key_priv = cls.parse_string(config, "ControllerCredentials", "SSHKeyPriv", optional=True)
        cls.credentials_ssh_key_pub = cls.parse_string(config, "ControllerCredentials", "SSHKeyPub", optional=True)
        cls.credentials_ssh_key_pass = cls.parse_string(config, "ControllerCredentials", "SSHKeyPassphrase", optional=True)
        cls.credentials_password = cls.parse_string(config, "ControllerCredentials", "Password", optional=True)

        cls.repo_git_path = cls.parse_string(config, "Repos", "GitRepoPath")

if __name__ == "__main__":
    # parse the options
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", help="Configuration file", default="/etc/ezbench/dutd.ini")
    args = parser.parse_args()

    # Parse the config file
    Configuration.load_config(args.conf)

    # Start the client
    client = ControllerClient(ezbench_dir, timings_db_dir)
    client.start()

    # Exit when the client asks us to
    def stop_handler(signum, frame):
        client.stop()
        print("-- The user requested to abort! --")
        return
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    # Main loop for reports
    client.run_reports()
