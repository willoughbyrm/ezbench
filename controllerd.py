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
from bottle import Bottle, abort, request, response, ServerAdapter
from collections import namedtuple
from datetime import datetime
import socket, threading
import configparser
import traceback
import argparse
import weakref
import pygit2
import shutil
import struct
import fcntl
import copy
import json
import time
import sys
import os

ezbench_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ezbench_dir, 'protocols'))
import controllerd_pb2

def ordered(obj):
    if isinstance(obj, dict):
        return sorted((k, ordered(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return sorted(ordered(x) for x in obj)
    else:
        return obj

class Machine:
    @classmethod
    def __grab_disk_lock(cls):
        cls.lock_fd = open(Configuration.state_file_lock, 'w')
        try:
            fcntl.flock(cls.lock_fd, fcntl.LOCK_EX)
            return True
        except IOError as e:
            print("Machine: Could not lock the DB: " + str(e))
            return False

    @classmethod
    def release_disk_lock(cls):
        try:
            fcntl.flock(cls.lock_fd, fcntl.LOCK_UN)
            cls.lock_fd.close()
        except Exception as e:
            print("Machine: Cannot release the DB: " + str(e))
            pass

    @classmethod
    def reload_state(cls, keep_lock = False):
        cls.__grab_disk_lock()
        try:
            with open(Configuration.state_file, 'rb') as f:
                cls.state.ParseFromString(f.read())
        except IOError as e:
            if not os.path.exists(Configuration.state_file):
                cls.save_state()
            else:
                print("Machine: failed to read the controller's state: " + str(e))

        if not keep_lock:
            cls.release_disk_lock()

    @classmethod
    def __save_state_to_file(cls, state, path):
        try:
            state_tmp = str(path) + ".tmp"
            with open(state_tmp, 'wb') as f:
                f.write(cls.state.SerializeToString())
                f.close()
                os.rename(state_tmp, path)
                return True
        except IOError:
            print("Machine: Could not dump the current state to a file!")
            return False

    @classmethod
    def save_state(cls):
        return cls.__save_state_to_file(cls.state, Configuration.state_file)

    @classmethod
    def init(cls):
        # Reset the state
        cls.lock = threading.Lock()
        cls.state = controllerd_pb2.FullState()
        cls.machines = dict()
        cls.lock_fd = None
        cls.rsync_base = None

        cls.reload_state()
        for machine in cls.state.machines:
            m = Machine(machine.name)

        # Start a thread that will take care of sending messages
        cls.thread_cmd_send_stop = False
        cls.event_cmd_send = threading.Event()
        cls.thread_cmd_send = threading.Thread(target=cls.cmd_sending_thread)
        cls.thread_cmd_send.start()

    @classmethod
    def fini(cls):
        cls.thread_cmd_send_stop = True
        cls.event_cmd_send.set()
        cls.thread_cmd_send.join()

    @classmethod
    def list(cls):
        with cls.lock:
            machines = copy.copy(cls.machines)
            return machines

    @classmethod
    def get(cls, machine_name):
        with cls.lock:
            return cls.machines.get(machine_name, None)

    @classmethod
    def jobs(cls):
        with cls.lock:
            jobs = list()
            for job in cls.state.jobs:
                job_obj = controllerd_pb2.Job()
                job_obj.CopyFrom(job)
                jobs.append(job_obj)
            return jobs

    @classmethod
    def __job_unlocked__(cls, job_id):
        if not cls.lock.locked:
            raise ValueError("The Machine.lock lock is not taken")

        for job in cls.state.jobs:
            if str(job.id) == job_id:
                return job
        return None

    @classmethod
    def job(cls, job_id):
        with cls.lock:
            job = cls.__job_unlocked__(job_id)
            if job is not None:
                job_obj = controllerd_pb2.Job()
                job_obj.CopyFrom(job)
                return job_obj
            else:
                return None

    @classmethod
    def job_machine_report_path(cls, job_id, machine_name):
        rel_upload_url="/{}/{}".format(job_id, machine_name)
        full_path = Configuration.reports_base_path + rel_upload_url
        public_url = Configuration.reports_base_url + rel_upload_url
        return full_path, rel_upload_url, public_url

    @classmethod
    def job_machine_del(cls, job, machine_name):
        if not cls.lock.locked:
            raise ValueError("The Machine.lock lock is not taken")

        # Delete the report folder, if present
        full_path, rel_upload_url, public_url = cls.job_machine_report_path(job.id, machine_name)

        # Send a message to the machine to delete its report
        machine = cls.machines.get(machine_name, None)
        cmd = controllerd_pb2.Cmd()
        cmd.delete_report.name = job.id
        machine.__queue_cmd_unlocked__(cmd)

        if os.path.exists(full_path):
            shutil.rmtree(full_path)

        if machine_name in job.machines:
            job.machines.remove(machine_name)

    @classmethod
    def __make_machine_report_repo__(cls, job, machine, force_reset=False):
                # Generate the path information for the future report
        full_path, rel_upload_url, public_url = cls.job_machine_report_path(job.id, machine)

        # remove any left-over tmp folder, before we re-create it
        if force_reset:
            try:
                shutil.rmtree(full_path)
            except OSError as e:
                if os.path.exists(full_path):
                    msg = "Fatal error: cannot delete the report folder '{}': {}"
                    raise ValueError(msg.format(full_path, e)) from e

        # Abort early if the repo already exist
        try:
            repo = pygit2.Repository(full_path)
            return
        except:
            pass

        # Create a repo for the machine for this job
        try:
            repo = pygit2.init_repository(full_path, True)
        except OSError as e:
            try:
                cls.job_machine_del(job, machine)
            except Exception as cleanup_err:
                msg = "Cannot create the repo '{}' and failed to clean up ({})"
                raise ValueError(msg.format(full_path, cleanup_err)) from e

            msg = "Cannot create the repo '{}'".format(full_path)
            raise ValueError(msg) from e

        # Make a commit with information about when the job was created and more
        job_desc = """= Job parameters =

        Controller name: {cname}
        Job ID         : {job_id}
        Machine        : {machine}

        Description    : {desc}
        Profile        : {profile}
""".format(cname=Configuration.name, job_id=job.id, machine=machine,
                   desc=job.report.description, profile=job.report.profile)

        # Create an initial commit in the repo
        try:
            data_oid = repo.create_blob(job_desc)
            tb = repo.TreeBuilder()
            tb.insert('job_desc', data_oid, pygit2.GIT_FILEMODE_BLOB_EXECUTABLE)
            new_tree = tb.write()
            author = pygit2.Signature('Ezbench controller {}'.format(Configuration.name),
                                    'noone@email.com')
            commit_id = repo.create_commit('refs/heads/master', author, author,
                                        "Job creation", new_tree, [])
            commit = repo.revparse_single(str(commit_id))

            # Update the master branch's target
            branch = repo.lookup_branch("master")
            branch.set_target(str(commit_id))
        except ValueError as e:
            try:
                cls.job_machine_del(job, machine)
            except Exception as cleanup_err:
                msg = "Could not create the initial commit in '{}' and failed to clean up ({})"
                raise ValueError(msg.format(full_path, cleanup_err)) from e

            msg = "Could not create the initial commit in '{}'".format(full_path)
            raise ValueError(msg) from e

    @classmethod
    def job_machine_add(cls, job, machine):
        if not cls.lock.locked:
            raise ValueError("The Machine.lock lock is not taken")

        cls.__make_machine_report_repo__(job, machine, force_reset=True)

        # Add the machine to the list of machines in the job
        job.machines.append(machine)

    @classmethod
    def job_update(cls, job_id, request):
        with cls.lock:
            cls.reload_state(keep_lock=True)

            # find the job
            job = cls.__job_unlocked__(job_id)
            if job is None:
                cls.release_disk_lock()
                raise ValueError("The job '{}' does not exist!".format(job_id))

            # Make sure all the machines exist before changing anything
            for machine_name in request.get('machines', []):
                machine = cls.machines.get(machine_name, None)
                if machine is None:
                    cls.release_disk_lock()
                    raise ValueError("The machine '{}' does not exist!".format(machine_name))

            # Update the description
            if "description" in request:
                job.report.description = request['description']

            # Remove all the machines that are not in the set of new machines
            # and make sure that every other machine has a valid repo set up
            for machine in job.machines:
                if machine not in request.get('machines', []):
                    cls.job_machine_del(job, machine)
                else:
                    cls.__make_machine_report_repo__(job, machine)

            # Add all the new machines
            for machine_name in request.get('machines', []):
                if machine_name not in job.machines:
                    try:
                        cls.job_machine_add(job, machine_name)
                    except Exception as e:
                        traceback.print_exc()
                        cls.release_disk_lock()
                        raise ValueError("The machine '{}' could not be added!".format(machine_name)) from e

            cls.save_state()
            cls.release_disk_lock()

    @classmethod
    def job_create(cls, job_id, request):
        # Sanity checks
        if 'description' not in request or type(request['description']) != str:
            raise ValueError("No description specified!")
        if type(request['description']) != str:
            raise ValueError("The 'description' field's type should be string!")

        if 'profile' not in request:
            raise ValueError("No profile specified!")
        if type(request['profile']) != str:
            raise ValueError("The 'profile' field's type should be string!")

        with cls.lock:
            cls.reload_state(keep_lock=True)
            # Check that the job does not exist already
            if cls.__job_unlocked__(job_id) is not None:
                cls.release_disk_lock()
                raise ValueError("The job already exists!")

            # Verify that all the machines are valid
            for machine_name in request.get('machines', []):
                machine = cls.machines.get(machine_name, None)
                if machine is None:
                    cls.release_disk_lock()
                    raise ValueError("The machine '{}' does not exist!".format(machine_name))

            # Create the job
            job = cls.state.jobs.add()
            job.id = job_id

            # Create the report object
            job.report.name = job_id
            job.report.description = request["description"]
            job.report.profile = request["profile"]

            # Copy the attributes
            for attr in request.get("attributes", dict()):
                for name in controllerd_pb2.ReportAttribute.DESCRIPTOR.fields_by_name:
                    if name == attr:
                        new_attr = job.report.attributes.add()
                        setattr(new_attr, name, request["attributes"][attr])

            cls.save_state()
            cls.release_disk_lock()

        # Now add the machines
        cls.job_update(job_id, request)

    @classmethod
    def job_delete(cls, job_id):
        with cls.lock:
            cls.reload_state(keep_lock=True)

            job = cls.__job_unlocked__(job_id)
            if job is None:
                cls.release_disk_lock()
                raise ValueError("The job {} does not exist!".format(job_id))

            # Check that the job does not exist already
            for job in cls.state.jobs:
                if job.id == job_id:
                    # Queue the delete commands to the machines
                    for machine_name in job.machines:
                        # Delete the report locally
                        cls.job_machine_del(job, machine_name)

                    cls.state.jobs.remove(job)
                    cls.save_state()
                    cls.release_disk_lock()

    @classmethod
    def check_commits_exist(cls, work):
        try:
            repo = pygit2.Repository(Configuration.src_repo_path)
        except Exception as e:
            print("ERROR: Cannot open the git repository '{}': {}".format(Configuration.src_repo_path, e))
            raise ValueError("The DUT repo is improperly set up")

        # Verify that commits are either a SHA1 or a tag
        for commit in work['commits']:
            try:
                rev = repo.revparse_single(commit)
            except Exception as e:
                raise ValueError("The commit '{}' cannot be found in the git repo".format(commit)) from e

            if type(rev) == pygit2.Tag:
                continue
            elif rev.hex != commit:
                # Could be a lightweight tag that points directly to a commit
                try:
                    tagcheckrev = repo.revparse_single("refs/tags/{}".format(commit))
                    continue
                except Exception as e:
                    raise ValueError("The commit '{}' is neither a tag nor a full SHA1".format(commit))

    @classmethod
    def job_work_check(cls, work):
        if 'commits' not in work:
            raise ValueError("The request does not contain the field 'commits'")

        if type(work['commits']) != dict:
            raise ValueError("The field 'commits' in the request is not a dict: {}".format(type(work['commits'])))

        for commit_id in work['commits']:
            if type(work['commits'][commit_id]) != dict:
                raise ValueError("The field 'commits: {}' is not a dict".format(commit_id))

            found_work = False
            for name in ['tests', 'testsets']:
                if name in work['commits'][commit_id]:
                    if type(work['commits'][commit_id][name]) != dict:
                        raise ValueError("The field '{}' is not a dict".format(work['commits'][commit_id][name]))

                    for test in work['commits'][commit_id][name]:
                        if type(test) != str:
                            raise ValueError("The field '{}' is not a str".format(test))

                        if type(work['commits'][commit_id][name][test]) != int:
                            raise ValueError("The round field of the test '{}' is not an int".format(test))

                    found_work = True

            if not found_work:
                raise ValueError("The field 'commits: {}' does not contain work".format(commit_id))

        # Verify that all the commits referenced are stored in the repo
        cls.check_commits_exist(work)

    @classmethod
    def job_work_set_unlocked(cls, job, work):
        # Empty the list of commits
        while len(job.commits) > 0:
            job.commits.pop()

        # Add all the work requested
        for commit_id in work['commits']:
            commit = job.commits.add()
            commit.id = commit_id
            for test_name in work['commits'][commit_id].get('tests', []):
                test = commit.tests.add()
                test.name = test_name
                test.rounds = work['commits'][commit_id]['tests'][test_name]
            for testset_name in work['commits'][commit_id].get('testsets', []):
                testset = commit.testsets.add()
                testset.name = testset_name
                testset.rounds = work['commits'][commit_id]['testsets'][testset_name]

        # Queue a message for the machines to get updated!
        for machine in job.machines:
            cmd = controllerd_pb2.Cmd()
            cmd.set_work.report.CopyFrom(job.report)

            # Create the git repo for the report and compute its URL
            full_path, rel_upload_url, public_url = cls.job_machine_report_path(job.id, machine)
            cmd.set_work.report.upload_url = rel_upload_url

            for commit in job.commits:
                new_commit = cmd.set_work.commits.add()
                new_commit.CopyFrom(commit)
            cls.machines[machine].__queue_cmd_unlocked__(cmd)

    @classmethod
    def job_work_put(cls, job_id, work):
        job = cls.__job_unlocked__(job_id)

        # Validate the architecture of the work request
        cls.job_work_check(work)

        with cls.lock:
            cls.reload_state(keep_lock=True)
            try:
                job = cls.__job_unlocked__(job_id)
                if job is None:
                    raise ValueError("The job '{}' does not exist!".format(job_id))

                cls.job_work_set_unlocked(job, work)
                cls.save_state()
            finally:
                cls.release_disk_lock()

    @classmethod
    def job_work_get(cls, job):
        state = dict()
        state['commits'] = dict()
        for commit in job.commits:
            state['commits'][commit.id] = dict()
            if len(commit.tests) > 0:
                state['commits'][commit.id]['tests'] = dict()
                for test in commit.tests:
                    state['commits'][commit.id]['tests'][test.name] = test.rounds
            if len(commit.testsets) > 0:
                state['commits'][commit.id]['testsets'] = dict()
                for testset in commit.testsets:
                    state['commits'][commit.id]['testsets'][testset.name] = testset.rounds
        return state

    @classmethod
    def __work_add__(cls, work, commit, work_type, name, rounds):
        if commit not in work['commits']:
            work['commits'][commit] = dict()

        if work_type not in work['commits'][commit]:
            work['commits'][commit][work_type] = dict()

        if name not in work['commits'][commit][work_type]:
            work['commits'][commit][work_type][name] = rounds
        else:
            work['commits'][commit][work_type][name] += rounds

        if work['commits'][commit][work_type][name] <= 0:
            del work['commits'][commit][work_type][name]
        if len(work['commits'][commit][work_type]) == 0:
            del work['commits'][commit][work_type]
        if len(work['commits'][commit]) == 0:
            del work['commits'][commit]

    @classmethod
    def job_work_patch(cls, job_id, work):
        # Validate the architecture of the work request
        cls.job_work_check(work)

        with cls.lock:
            cls.reload_state(keep_lock=True)
            try:
                job = cls.__job_unlocked__(job_id)
                if job is None:
                    raise ValueError("The job '{}' does not exist!".format(job_id))

                # read the current work and add the wanted work
                cur_work = cls.job_work_get(job)
                for commit_id in work['commits']:
                    for work_type in ['tests', 'testsets']:
                        for test_name in work['commits'][commit_id].get(work_type, []):
                            rounds = work['commits'][commit_id][work_type][test_name]
                            cls.__work_add__(cur_work, commit_id, work_type, test_name, rounds)

                cls.job_work_set_unlocked(job, cur_work)
                cls.save_state()
            finally:
                cls.release_disk_lock()

    # Requires taking the global lock
    @classmethod
    def __find_machine__(cls, machine_name):
        for machine in cls.state.machines:
            if machine.name == machine_name:
                return machine
        return None

    def __init__(self, machine_name):
        self.name = machine_name
        self._client = None

        with self.lock:
            self.reload_state(keep_lock=True)
            if machine_name not in self.machines:
                machine = self.state.machines.add()
                machine.name = machine_name
                machine.next_cmd_id = 0
                machine.recv_sig_count = 0
                self.save_state()
            self.release_disk_lock()

            # Add the machine to the list of tests
            self.machines[machine_name] = self

    def send_next_cmd(self):
        client = self.client()
        if client is None:
            return False

        with self.lock:
            self.reload_state(keep_lock=True)
            machine = self.__find_machine__(self.name)

            now = datetime.utcnow().timestamp()

            # TODO: start from the end to reduce the overhead of the search
            for qcmd in machine.queued_cmds:
                if qcmd.acknowledged == 0:
                    # Only try to resend the command if the client re-connected
                    # or we already waited more than the resend_period (seconds)
                    if (client.connect_time > qcmd.last_sent or
                        now - qcmd.last_sent > Configuration.resend_period):
                        # We are ready to send, update the status of this command
                        qcmd.last_sent = now
                        self.save_state()
                        self.release_disk_lock()

                        # Now send the message
                        client.send_msg(qcmd.cmd)

                        return True
                    else:
                        break

        self.release_disk_lock()
        return False

    @classmethod
    def cmd_sending_thread(cls):
        while not cls.thread_cmd_send_stop:
            # Wait a command to need to be sent
            cls.event_cmd_send.wait(Configuration.resend_period + 0.01)

            # Exit immediately if we got asked to stop
            if cls.thread_cmd_send_stop:
                return

            # Get the current list of machines and get a reference on each object
            machines = list()
            with cls.lock:
                for machine in cls.machines:
                    machines.append(cls.machines[machine])

                # We got all the machines, so we can ACK the event and process
                cls.event_cmd_send.clear()

            # Now send the messages of all the machines
            for machine in machines:
                machine.send_next_cmd()

    def __queue_cmd_unlocked__(self, cmd):
        machine = self.__find_machine__(self.name)

        # allocate an ID for the command
        cmd.id = machine.next_cmd_id
        machine.next_cmd_id += 1

        # add the command to the queue
        queued_cmd = machine.queued_cmds.add()
        queued_cmd.cmd.CopyFrom(cmd)

        # Wake up the command-sending thread
        self.event_cmd_send.set()

    def queue_cmd(self, cmd):
        with self.lock:
            self.reload_state(keep_lock=True)

            self.__queue_cmd_unlocked__(cmd)

            # make sure everything is on disk before sending anything
            self.save_state()
            self.release_disk_lock()

    def recv_msg(self, sig):
        with self.lock:
            self.reload_state(keep_lock=True)
            machine = self.__find_machine__(self.name)

            if sig.HasField("cmd_status"):
                for qcmd in machine.queued_cmds:
                    if qcmd.cmd.id == sig.cmd_status.id:
                        qcmd.status.CopyFrom(sig.cmd_status)
                        qcmd.acknowledged = datetime.utcnow().timestamp()
                        exec_time_ms = (qcmd.acknowledged - qcmd.last_sent) * 1000.0

                        # Update the ping of the machine, if the command was a ping
                        if qcmd.cmd.HasField('ping'):
                            machine.ping = exec_time_ms

                        msg = "{}: cmd {} got executed in {:.2f} ms: {} {}"
                        err_code = controllerd_pb2.CmdStatus.CmdErrorCode.Name(qcmd.status.err_code)
                        err_msg = ""
                        if len(qcmd.status.err_msg) > 0:
                            err_msg = ("({})".format(qcmd.status.err_msg))
                        print(msg.format(self.name, qcmd.cmd.id, exec_time_ms,
                                         err_code, err_msg))
            elif sig.HasField("reports"):
                machine.reports.CopyFrom(sig.reports)
            elif sig.HasField("tests"):
                machine.tests.CopyFrom(sig.tests)
            elif sig.HasField("testsets"):
                machine.testsets.CopyFrom(sig.testsets)
            elif sig.HasField("log"):
                print("More logs for {}: {}".format(sig.log.report, sig.log.msg))

            machine.last_seen = int(datetime.utcnow().timestamp())
            machine.recv_sig_count += 1
            self.save_state()
            self.release_disk_lock()

    def available_tests(self):
        with self.lock:
            tests = dict()
            for test in self.__find_machine__(self.name).tests.tests:
                test_obj = controllerd_pb2.AvailableTests.Test()
                test_obj.CopyFrom(test)
                tests[test.name] = test_obj
            return tests

    def available_testsets(self):
        with self.lock:
            testsets = dict()
            for testset in self.__find_machine__(self.name).testsets.testsets:
                testset_obj = controllerd_pb2.AvailableTestsets.Testset()
                testset_obj.CopyFrom(testset)
                testsets[testset.name] = testset_obj
            return testsets

    def queued_cmds(self, pending_only=True):
        with self.lock:
            machine = self.__find_machine__(self.name)
            if not pending_only:
                return machine.queued_cmds

            cmds = list()
            for qcmd in machine.queued_cmds:
                if qcmd.status.err_code == controllerd_pb2.CmdStatus.NON_ACK:
                    qcmd_obj = controllerd_pb2.QueuedCmd()
                    qcmd_obj.CopyFrom(qcmd)
                    cmds.append(qcmd_obj)

            return cmds

    def last_seen(self):
        with self.lock:
            return self.__find_machine__(self.name).last_seen

    def recv_sig_count(self):
        with self.lock:
            return self.__find_machine__(self.name).recv_sig_count

    def reports(self):
        with self.lock:
            reports = list()
            for report in self.__find_machine__(self.name).reports.reports:
                report_obj = controllerd_pb2.ReportState()
                report_obj.CopyFrom(report)
                reports.append(report_obj)
            return reports

    def report(self, report_name):
        with self.lock:
            machine = self.__find_machine__(self.name)
            for report in machine.reports.reports:
                if report.name == report_name:
                    report_obj = controllerd_pb2.ReportState()
                    report_obj.CopyFrom(report)
                    return report_obj
        return None

    def ping(self):
        with self.lock:
            return self.__find_machine__(self.name).ping

    def send_ping(self):
        cmd = controllerd_pb2.Cmd()
        cmd.ping.requested = True
        self.queue_cmd(cmd)

    def set_client(self, client):
        if client is not None:
            if self._client is not None:
                cur_client = self._client()
                if cur_client is not None and cur_client.is_online():
                    return False

            # Get a weak reference to the client and store it
            self._client = weakref.ref(client)

            # Update the state
            with self.lock:
                self.reload_state(keep_lock=True)
                machine = self.__find_machine__(self.name)
                machine.last_seen = int(datetime.utcnow().timestamp())
                self.save_state()
                self.release_disk_lock()

            # Kick the processing of the cmd queue
            self.event_cmd_send.set()
        else:
            self._client = None

        return True

    def client(self):
        if self._client is not None:
            return self._client()
        else:
            return None

    def is_online(self):
        client = self.client()
        if client is not None:
            return client.is_online()
        else:
            return False

class ClientThread(threading.Thread):
    def __init__(self, ip, port, client_socket, quiet=False):
        threading.Thread.__init__(self)
        self.machine_name = None
        self.ip = ip
        self.port = port
        self.socket = client_socket
        self.quiet = quiet
        self.connect_time = datetime.utcnow().timestamp()

        self.online = True

        # Tell the OS to never buffer as we will always send small packets
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def recv_msg(self, msg):
        msg_len_buf = self.socket.recv(4, socket.MSG_WAITALL)
        if len(msg_len_buf) != 4:
            return None

        msg_len = struct.unpack('>I', msg_len_buf)[0]
        if msg_len == 0:
            print("WARNING: Receiving a 0-length buffer!")

        buf = bytes()
        while len(buf) < msg_len:
            buf += self.socket.recv(msg_len - len(buf), socket.MSG_WAITALL)

        if len(buf) == msg_len:
            msg.ParseFromString(buf)
            return msg
        else:
            return None

    def send_msg(self, msg, close_on_failure=True):
        if not self.online:
            return

        buf = msg.SerializeToString()

        try:
            controllerd_pb2.Cmd().ParseFromString(buf)
        except:
            if type(msg) == controllerd_pb2.Cmd():
                print("Error: protobuf generated an invalid serialized string: msg =", msg)
                if close_on_failure:
                    print("[+] Machine {}: Invalid message to be sent. Shut down the connection...".format(self.machine_name))
                    self.close()
                    return False

        packet_size = struct.pack('>I', len(buf))

        #print("msg sent (size={}):".format(len(buf)), buf)

        # Try to send all the bytes, or close the connection
        try:
            self.socket.sendall(packet_size + buf, socket.MSG_DONTWAIT)
            return True
        except:
            if close_on_failure:
                print("[+] Machine {}'s send buffer is full. Shut down the connection...".format(self.machine_name))
                self.close()
            return False

    def __exit(self):
        self.online = False
        self.socket.close()

    def run(self):
        machine = None
        try:
            # Read the client hello message
            hello = self.recv_msg(controllerd_pb2.ClientHello())
            if hello is None:
                self.socket.close()
                return

            # Set the client immediately to also update its last_seen
            self.machine_name = hello.machine_name
            machine = Machine.get(hello.machine_name)
            if machine is None:
                machine = Machine(hello.machine_name)
            if not machine.set_client(self):
                client = machine.client()
                msg = "[+] Duplicate machine '{}' connected from {}:{} using version {} - already connected from {}:{}"
                print(msg.format(hello.machine_name, self.ip, self.port, hello.version,
                            client.ip, client.port))
                # Ping the machine in case it's a zombie connection
                machine.send_ping()
                self.__exit()
                return False

            # Send the hello message
            hello_back = controllerd_pb2.ServerHello()
            hello_back.version = 1
            hello_back.controller_name = Configuration.name
            if not self.send_msg(hello_back):
                if not self.quiet:
                    print("[+] Machine '{}' got disconnected...".format(machine.name))
                self.__exit()
                return

            queue_length = len(machine.queued_cmds(pending_only=True))
            if not self.quiet:
                msg = "[+] Machine '{}' connected from {}:{} using version {} - {} commands queued"
                print(msg.format(hello.machine_name, self.ip, self.port, hello.version,
                                queue_length))

            while self.online:
                sig = self.recv_msg(controllerd_pb2.Signal())
                if sig is None:
                    break

                machine.recv_msg(sig)
        except ConnectionResetError:
            pass
        finally:
            if not self.quiet:
                if machine is not None:
                    print("[+] Machine '{}' got disconnected...".format(machine.name))
                else:
                    print("Handshake failed from {}:{}...".format(self.ip, self.port))
            self.__exit()

    def is_online(self):
        return self.online

    def close(self):
        try:
            self.socket.shutdown(socket.SHUT_RD)
        except OSError:
            pass
        self.join()

class AsyncDUTServer:
    def __init__(self, bind_params, quiet=False):
        self.bind_params = bind_params
        self.quiet = quiet

        self._exit_now = False
        self.ready_for_clients = False

        self.thread_tcp_server = threading.Thread(target = self.run)

        self.clients_lock = threading.Lock()
        self.clients = []

        # Initialize the machine's state
        Machine.init()

        # Wait for the server to have started
        self.thread_tcp_server.start()
        while not self.ready_for_clients:
            time.sleep(0.001)

    def __del__(self):
        self.close()

    def run(self):
        self.tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcpsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcpsock.bind(self.bind_params)

        # Start listening for clients
        if not self.quiet:
            print("Listening for incoming connections...")
        self.tcpsock.listen(10)
        self.ready_for_clients = True
        while not self._exit_now:
            try:
                (clientsock, (ip, port)) = self.tcpsock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            newthread = ClientThread(ip, port, clientsock, quiet=self.quiet)
            newthread.start()
            with self.clients_lock:
                self.clients.append(weakref.ref(newthread))

        # We are exiting, prevent new connections
        self._exit_now = True
        self.tcpsock.close()

        # Now kill all the clients
        with self.clients_lock:
            for client in self.clients:
                ref = client()
                if ref is not None:
                    ref.close()

    def close(self):
        if not self._exit_now:
            self._exit_now = True
            self.tcpsock.shutdown(socket.SHUT_RDWR)
        self.thread_tcp_server.join()

        # Make sure the Machine class is done with all its work
        Machine.fini()

class AsyncRestServer:
    def __init__(self, bind_params, debug=False, quiet=False):
        self.app = app = Bottle()

        protocol_version = 1

        @app.route('/')
        def list_jobs():
            state = dict()
            state['service'] = "ezbench_controllerd"
            state['name'] = Configuration.name
            state['machines_url'] = "/machines"
            state['jobs_url'] = "/jobs"
            state['version'] = protocol_version
            return state

        @app.route('/version')
        def version():
            state = dict()
            state['version'] = protocol_version
            return state

        @app.route('/machines')
        def machines():
            state=dict()
            state['machines'] = dict()

            machines = Machine.list()
            for machine in machines:
                state['machines'][machine] = dict()
                state['machines'][machine]['online'] = machines[machine].is_online()
                state['machines'][machine]['last_seen'] = machines[machine].last_seen()
                state['machines'][machine]['pending_commands'] = len(machines[machine].queued_cmds(pending_only=True))
                state['machines'][machine]['url'] = "/machines/{}".format(machine)
            return state

        @app.route('/machines/<machine_name>')
        def machine(machine_name):
            machine = Machine.get(machine_name)
            if machine is None:
                abort(404, "No such machine.")

            state=dict()
            state['name'] = machine.name
            state['online'] = machine.is_online()
            state['last_seen'] = machine.last_seen()
            state['recv_sig_count'] = machine.recv_sig_count()
            state['ping'] = machine.ping()

            state['reports'] = dict()
            for report in machine.reports():
                state['reports'][report.name] = dict()
                state['reports'][report.name]['state'] = report.state
                state['reports'][report.name]['url'] = "/machines/{}/reports/{}".format(machine_name, report.name)

            state['tests'] = dict()
            for test in machine.available_tests():
                state['tests'][test] = dict()
                state['tests'][test]['url'] = "/machines/{}/tests/{}".format(machine_name, test)

            state['testsets'] = dict()
            testsets = machine.available_testsets()
            for testset in testsets:
                state['testsets'][testset] = dict()
                state['testsets'][testset]['description'] = testsets[testset].description
                state['testsets'][testset]['url'] = "/machines/{}/testsets/{}".format(machine_name, testset)

            state['queued_cmds'] = list()
            for qcmd in machine.queued_cmds(pending_only=True):
                cmd = dict()
                cmd['url'] = "/machines/{}/commands/{}".format(machine_name, qcmd.cmd.id)
                cmd['id'] = qcmd.cmd.id
                cmd['description'] = str(qcmd.cmd)
                cmd['last_sent'] = qcmd.last_sent
                cmd['acknowledged'] = qcmd.acknowledged
                cmd['err_code'] = str(controllerd_pb2.CmdStatus.CmdErrorCode.Name(qcmd.status.err_code))
                if len(qcmd.status.err_msg) > 0:
                    cmd['err_msg'] = qcmd.status.err_msg
                state['queued_cmds'].append(cmd)

            return state

        @app.route('/machines/<machine_name>/commands/<cmd_id>')
        def command(machine_name, cmd_id):
            machine = Machine.get(machine_name)
            if machine is None:
                abort(404, "No such machine.")

            for qcmd in machine.queued_cmds(pending_only=False):
                if str(qcmd.cmd.id) == cmd_id:
                    state=dict()
                    state['id'] = qcmd.cmd.id
                    state['description'] = str(qcmd.cmd)
                    state['last_sent'] = qcmd.last_sent
                    state['acknowledged'] = qcmd.acknowledged
                    state['err_code'] = str(controllerd_pb2.CmdStatus.CmdErrorCode.Name(qcmd.status.err_code))
                    if len(qcmd.status.err_msg) > 0:
                        state['err_msg'] = qcmd.status.err_msg
                    return state

            abort(404, "No such command.")

        @app.route('/machines/<machine_name>/reports/<report_name>')
        def report(machine_name, report_name):
            machine = Machine.get(machine_name)
            if machine is None:
                abort(404, "No such machine.")

            for report in machine.reports():
                if report.name == report_name:
                    state=dict()
                    state['name'] = report.name
                    state['profile'] = report.profile
                    state['machine'] = machine_name
                    state['state'] = report.state
                    state['state_disk'] = report.state_disk
                    state['build_time'] = report.build_time
                    state['deploy_time'] = report.deploy_time
                    return state

            abort(404, "No such report.")

        @app.route('/machines/<machine_name>/tests/<test_name>')
        def test(machine_name, test_name):
            machine = Machine.get(machine_name)
            if machine is None:
                abort(404, "No such machine.")

            tests = machine.available_tests()
            for tname in tests:
                test = tests[tname]
                if test.name == test_name:
                    state=dict()
                    state['name'] = test.name
                    state['exec_time_median'] = test.exec_time_median
                    state['exec_time_max'] = test.exec_time_max
                    state['machine'] = machine_name
                    return state

            abort(404, "No such test.")

        @app.route('/machines/<machine_name>/testsets/<testset_name>')
        def testset(machine_name, testset_name):
            machine = Machine.get(machine_name)
            if machine is None:
                abort(404, "No such machine.")

            testsets = machine.available_testsets()
            for tname in testsets:
                testset = testsets[tname]
                if testset.name == testset_name:
                    state=dict()
                    state['name'] = testset.name
                    state['description'] = testset.description
                    state['machine'] = machine_name
                    state['tests'] = dict()
                    for test in testset.tests:
                        state['tests'][test.name] = test.rounds
                    return state

            abort(404, "No such testset.")

        @app.route('/jobs')
        def list_jobs():
            state = dict()
            state['jobs'] = dict()
            for job in Machine.jobs():
                state['jobs'][job.id] = dict()
                state['jobs'][job.id]['description'] = job.report.description
                state['jobs'][job.id]['url'] = '/jobs/{}'.format(job.id)

            return state

        @app.route('/jobs/<job_id>', ['GET', 'POST', 'PATCH', 'DELETE'])
        def job(job_id):
            job = Machine.job(job_id)
            if job is None and request.method != 'POST':
                abort(404, "No such job.")

            if request.method == 'GET':
                if job is None:
                    abort(404, "No such job.")

                state = dict()
                state['id'] = job.id
                state['description'] = job.report.description
                state['profile'] = job.report.profile

                state['attributes'] = dict()
                for attr in job.report.attributes:
                    for name in attr.DESCRIPTOR.fields_by_name:
                        if attr.HasField(name):
                            state['attributes'][name] = getattr(attr, name)

                state['machines'] = dict()
                for machine_name in job.machines:
                    state['machines'][machine_name] = dict()
                    state['machines'][machine_name]['report'] = "/machines/{}/reports/{}".format(machine_name, job.id)

                    full_path, rel_upload_url, public_url = Machine.job_machine_report_path(job_id, machine_name)
                    state['machines'][machine_name]['clone_url'] = public_url

                    machine = Machine.get(machine_name)
                    if machine is None:
                        continue
                    state['machines'][machine_name]['online'] = machine.is_online()
                    report = machine.report(job.id)
                    if report is None:
                        state['machines'][machine_name]['state'] = "MISSING"
                    else:
                        state['machines'][machine_name]['state'] = report.state

                return state
            elif request.method == "DELETE":
                try:
                    Machine.job_delete(job_id)
                except ValueError as e:
                    abort(400, "Invalid request: {}".format(e))
            else:
                if request.json is None:
                    abort(400, "Invalid request: Request format not 'application/json'")

                if request.method == 'POST':
                    try:
                        Machine.job_create(job_id, request.json)
                        response.status = 201
                    except ValueError as e:
                        abort(400, "Invalid request: {}".format(e))
                elif request.method == "PATCH":
                    try:
                        Machine.job_update(job_id, request.json)
                    except ValueError as e:
                        abort(400, "Invalid request: {}".format(e))

        @app.route('/jobs/<job_id>/work', ["GET", "PUT", "PATCH"])
        def work(job_id):
            job = Machine.job(job_id)
            if job is None:
                abort(404, "No such job.")

            if request.method == 'GET':
                return Machine.job_work_get(job)
            else:
                if request.json is None:
                    abort(400, "Invalid request: Request format not 'application/json'")

                if request.method == 'PUT':
                    try:
                        Machine.job_work_put(job_id, request.json)
                    except ValueError as e:
                        abort(400, "Invalid request: {}".format(e))
                elif request.method == 'PATCH':
                    try:
                        Machine.job_work_patch(job_id, request.json)
                    except ValueError as e:
                        abort(400, "Invalid request: {}".format(e))

        @app.route('/jobs/<job_id>/tasks', method="GET")
        def tasks():
            """{
                "machines": {
                    "mperes_DESK": [
                        {
                            "description": "commit abcdef: glxgears:window 1/3"
                            "expected_completion_time": 300,
                            "elapsed_time": 12.3,
                            "rounds": 3,
                            "current_round": 1
                        },
                        {
                            "description": "commit abcdef: glxgears:window 1/3"
                            "expected_completion_time": 300,
                            "elapsed_time": 12.3,
                            "rounds": 3,
                            "current_round": 0
                        },
                    ]
                }
            }"""
            print(request.json)
            response.status = 201
            return "CREATED!"

        # Suggested in http://stackoverflow.com/questions/11282218/bottle-web-framework-how-to-stop
        class StoppableWSGIRefServer(ServerAdapter):
            server = None
            ready_for_clients = False

            def run(self, handler):
                from wsgiref.simple_server import make_server, WSGIRequestHandler
                if self.quiet:
                    class QuietHandler(WSGIRequestHandler):
                        def log_request(*args, **kw): pass
                    self.options['handler_class'] = QuietHandler
                self.server = make_server(self.host, self.port, handler, **self.options)
                self.ready_for_clients = True
                self.server.serve_forever()

            def stop(self):
                self.server.socket.shutdown(socket.SHUT_RDWR)
                self.server.shutdown()
                self.server.socket.close()

        self.server = StoppableWSGIRefServer(host=bind_params[0], port=bind_params[1])
        self.thread_app = threading.Thread(target=app.run,
                                           kwargs={"debug": debug, "quiet": quiet,
                                                   "server": self.server})
        self.thread_app.start()

        # Wait for the server to have started
        while not self.server.ready_for_clients:
            time.sleep(0.001)

    def __del__(self):
        self.close()

    def close(self):
        if self.thread_app.is_alive():
            self.server.stop()
            self.app.close()
            self.thread_app.join()

class Configuration:
    resend_period = 30

    @classmethod
    def parse_error(cls, msg):
        print("Parsing error: " + msg, file=sys.stderr)
        sys.exit(1)

    @classmethod
    def format_error(cls, category, key, found, expected_format):
        msg = "Invalid format for the field [{}].{}: found '{}', expected {}"
        cls.parse_error(msg.format(category, key, found, expected_format))

    @classmethod
    def parse_string(cls, config, category, key):
        try:
            val = config[category][key]
        except:
            cls.parse_error("Field [{}].{} is missing".format(category, key))

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

        for category in ["General", "RESTServer", "DUTServer", "Reports", "Repos"]:
            if category not in config:
                cls.parse_error("The category '{}' is missing".format(category))

        cls.name = cls.parse_string(config, "General", "Name")
        cls.state_file = cls.parse_string(config, "General", "StateFile")
        cls.state_file_lock = cls.parse_string(config, "General", "StateFileLock")

        cls.rest_bind = cls.parse_bind_addr(config, "RESTServer", "BindAddress")

        cls.dut_bind = cls.parse_bind_addr(config, "DUTServer", "BindAddress")

        cls.reports_base_path = cls.parse_string(config, "Reports", "ReportsBasePath")
        cls.reports_base_url = cls.parse_string(config, "Reports", "ReportsBaseURL")

        cls.src_repo_path = cls.parse_string(config, "Repos", "GitRepoPath")

def start_controllerd(debug=False, quiet=False):
    # Print information about the parameters
    print("Controller's name           : {}".format(Configuration.name))
    print("Controller's state file     : {}".format(Configuration.state_file))
    print("Controller's state file lock: {}".format(Configuration.state_file_lock))
    print()
    print("DUT server binding point    : {}".format(Configuration.dut_bind))
    print("REST server binding point   : {}".format(Configuration.rest_bind))
    print()
    print("Reports' base GIT path      : {}".format(Configuration.reports_base_path))
    print("Reports' base GIT public URL: {}".format(Configuration.reports_base_url))
    print()

    # Start the TCP server that will listen for machines' incoming connections
    tcpserver = AsyncDUTServer(Configuration.dut_bind, quiet=quiet)

    # Start the REST server that will allow people to schedule work
    restserver = AsyncRestServer(Configuration.rest_bind, debug, quiet)

    return tcpserver, restserver

from pstats import Stats
if __name__ == "__main__":
    # parse the options
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", help="Configuration file", default="/etc/ezbench/controllerd.ini")
    parser.add_argument("--debug", help="Enable debug mode", action="store_true", default=False)
    parser.add_argument("--quiet", help="Enable quiet mode", action="store_true", default=False)
    args = parser.parse_args()

    # Parse the config file
    Configuration.load_config(args.conf)

    # Start the server
    tcpserver, restserver = start_controllerd(args.debug, args.quiet)

    # Wait for a keyboard signal before quitting
    while True:
        try:
            time.sleep(1000)
        except KeyboardInterrupt:
            break

    restserver.close()
    tcpserver.close()

