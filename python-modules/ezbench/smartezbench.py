"""
Copyright (c) 2015, Intel Corporation

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
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import multiprocessing
import statistics
import subprocess
import threading
import pprint
import fcntl
import time
import json
import glob
import copy
import math
import sys
import gc
import os
import re

ezbench_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(os.path.join(ezbench_dir, 'timing_DB'))

from ezbench.testset import *
from ezbench.report import *
from ezbench.runner import *
from timing import *

# Smart-ezbench-related classes
class Criticality(Enum):
    DD = 4
    II = 3
    WW = 2
    EE = 1

class RunningMode(Enum):
    INITIAL = 0
    RUN = 1
    PAUSE = 2
    ERROR = 3
    ABORT = 4

    # Intermediate steps, going from RUN to RUNNING or RUNNING to PAUSE/ABORT
    INTERMEDIATE = 100
    RUNNING = 101
    PAUSING = 102
    ABORTING = 103


def list_smart_ezbench_report_names(ezbench_dir, updatedSince = 0):
    log_dir = ezbench_dir + '/logs'
    state_files = glob.glob("{log_dir}/*/smartezbench.state".format(log_dir=log_dir));

    reports = []
    for state_file in state_files:
        if updatedSince > 0 and os.path.getmtime(state_file) < updatedSince:
            continue

        start = len(log_dir) + 1
        stop = len(state_file) - 19
        reports.append(state_file[start:stop])

    return reports

class TaskEntry:
    def __init__(self, commit, test, rounds):
        self.commit = commit
        self.test = test
        self.rounds = rounds
        self.start_date = None
        self.exec_time = None
        self.build_time = None

    def started(self):
        self.start_date = datetime.now()

    def predicted_completion_time(self):
        b = 0
        if self.build_time is not None:
            b = self.build_time

        e = 0
        if self.exec_time is not None:
            e = self.exec_time

        return timedelta(0, b + e)

    def set_timing_information(self, timingsDB, compilation_time = None,
                               available_versions = {}):
        if compilation_time is not None and self.commit not in available_versions:
            self.build_time = compilation_time

        time = timingsDB.data("test", self.test)
        if len(time) > 0:
            self.exec_time = statistics.median(time) * self.rounds
        else:
            self.exec_time = None

    def remaining_time(self):
        if self.start_date is not None:
            elapsed = datetime.now() - self.start_date
        else:
            elapsed = timedelta(0)
        return self.predicted_completion_time() - elapsed

    def __str__(self):
        string = "{}: {}: {} run(s)".format(self.commit, self.test, self.rounds)

        total_delta = self.predicted_completion_time()
        if total_delta.total_seconds() > 0:
            remaining = self.remaining_time()

            if self.start_date is not None:
                progress = 100.0 - (remaining.total_seconds() * 100 / total_delta.total_seconds())

                if remaining.total_seconds() > 0:
                    remaining_str = str(timedelta(0, math.ceil(remaining.total_seconds()))) + "s remaining"
                else:
                    remaining_str = str(timedelta(0, math.floor(-remaining.total_seconds()))) + "s overtime"

                string += "({:.2f}%, {})".format(progress, remaining_str)
            else:
                rounded_total_delta = timedelta(0, math.ceil(total_delta.total_seconds()))
                string += "(estimated completion time: {}s)".format(rounded_total_delta)
        else:
            if self.start_date is not None:
                string += "(started {} ago)".format(datetime.now() - self.start_date)
            else:
                string += "(no estimation available)"

        return string

GitCommit = namedtuple('GitCommit', 'sha1 timestamp')

class SmartEzbench:
    def __init__(self, ezbench_dir, report_name, readonly = False):
        self.readonly = readonly
        self.ezbench_dir = ezbench_dir
        self.report_name = report_name
        self.log_folder = ezbench_dir + '/logs/' + report_name
        self.smart_ezbench_state = self.log_folder + "/smartezbench.state"
        self.smart_ezbench_lock = self.log_folder + "/smartezbench.lock"
        self.smart_ezbench_log = self.log_folder + "/smartezbench.log"
        self._report_cached = None

        self.state = dict()
        self.state['commits'] = dict()
        self.state['mode'] = RunningMode.INITIAL.value

        self._task_lock = threading.Lock()
        self._task_current = None
        self._task_list = None
        self._events_str = None

        self.min_criticality = Criticality.II

        # Create the log directory
        first_run = False
        if not readonly and not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
            first_run = True

        # Open the log file as append
        self.log_file = open(self.smart_ezbench_log, "a")

        # Add the welcome message
        if first_run or not self.__reload_state():
            if readonly:
                raise RuntimeError("The report {} does not exist".format(report_name))
            self.__save_state()
            self.__log(Criticality.II,
                    "Created report '{report_name}' in {log_folder}".format(report_name=report_name,
                                                                            log_folder=self.log_folder))

    def __log(self, error, msg):
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = "{time}: ({error}) {msg}\n".format(time=time, error=error.name, msg=msg)
        if error.value <= self.min_criticality.value:
            print(log_msg, end="")
            if not self.readonly:
                self.log_file.write(log_msg)
                self.log_file.flush()

    def __grab_lock(self):
        if self.readonly:
            return
        self.lock_fd = open(self.smart_ezbench_lock, 'w')
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX)
            return True
        except IOError as e:
            self.__log(Criticality.EE, "Could not lock the report: " + str(e))
            return False

    def __release_lock(self):
        if self.readonly:
            return

        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            self.lock_fd.close()
        except Exception as e:
            self.__log(Criticality.EE, "Cannot release the lock: " + str(e))
            pass

    def __reload_state_unlocked(self):
        # check if a report already exists
        try:
            with open(self.smart_ezbench_state, 'rt') as f:
                self.state_read_time = time.time()
                try:
                    self.state = json.loads(f.read())
                except Exception as e:
                    self.__log(Criticality.EE, "Exception while reading the state: " + str(e))
                    pass
                return True
        except IOError as e:
            self.__log(Criticality.WW, "Cannot open the state file: " + str(e))
            pass
        return False

    def __reload_state(self, keep_lock = False):
        self.__grab_lock()
        ret = self.__reload_state_unlocked()
        if not keep_lock:
            self.__release_lock()
        return ret

    def __save_state(self):
        if self.readonly:
            return

        try:
            state_tmp = str(self.smart_ezbench_state) + ".tmp"
            with open(state_tmp, 'wt') as f:
                f.write(json.dumps(self.state, sort_keys=True, indent=4, separators=(',', ': ')))
                f.close()
                os.rename(state_tmp, self.smart_ezbench_state)
                return True
        except IOError:
            self.__log(Criticality.EE, "Could not dump the current state to a file!")
            return False

    def __create_ezbench(self, ezbench_path = None, profile = None, report_name = None):
        if profile is None:
            profile = self.profile()

        return Ezbench(ezbench_dir = self.ezbench_dir, profile = profile,
                       report_name = self.report_name,
                       run_config_scripts = self.conf_scripts())

    def __read_attribute_unlocked__(self, attr, default = None):
        if attr in self.state:
            return self.state[attr]
        else:
            return default

    def __read_attribute__(self, attr, default = None):
        self.__reload_state(keep_lock=False)
        return self.__read_attribute_unlocked__(attr, default)

    def __write_attribute_unlocked__(self, attr, value, allow_updates = False):
        if allow_updates or attr not in self.state or self.state['beenRunBefore'] == False:
            self.state[attr] = value
            self.__save_state()
            return True
        return False

    def __write_attribute__(self, attr, value, allow_updates = False):
        self.__reload_state(keep_lock=True)
        ret = self.__write_attribute_unlocked__(attr, value, allow_updates)
        self.__release_lock()
        return ret

    def __running_mode_unlocked__(self, check_running = True):
        mode = self.__read_attribute_unlocked__('mode', RunningMode.INITIAL.value) % RunningMode.INTERMEDIATE.value

        if check_running and Ezbench(self.ezbench_dir, report_name=self.report_name).reportIsLocked():
            mode += RunningMode.INTERMEDIATE.value

        return RunningMode(mode)

    def running_mode(self, check_running = True):
        self.__reload_state(keep_lock=True)
        ret = self.__running_mode_unlocked__()
        self.__release_lock()
        return ret

    def set_running_mode(self, mode):
        if mode.value >= RunningMode.INTERMEDIATE.value:
            self.__log(Criticality.EE, "Ezbench mode cannot manually be set to '{}'".format(mode.name))
            return False

        self.__reload_state(keep_lock=True)

        # Request an early exit if we go from RUNNING to PAUSE or
        cur_mode = self.__running_mode_unlocked__()
        if cur_mode.value > RunningMode.INTERMEDIATE.value and mode != RunningMode.RUN:
            Ezbench.requestEarlyExit(self.ezbench_dir, self.report_name)

        self.__write_attribute_unlocked__('mode', mode.value, allow_updates = True)
        self.__log(Criticality.II, "Ezbench running mode set to '{mode}'".format(mode=mode.name))
        self.__release_lock()

        return True

    def profile(self):
        return self.__read_attribute__('profile')

    def set_profile(self, profile):
        self.__reload_state(keep_lock=True)
        if 'beenRunBefore' not in self.state or self.state['beenRunBefore'] == False:
            # Check that the profile exists!
            ezbench = self.__create_ezbench(profile = profile)
            run_info = ezbench.run(["HEAD"], [], [], dry_run=True)
            if not run_info.success():
                if run_info.exit_code == EzbenchExitCode.ARG_PROFILE_INVALID:
                    self.__log(Criticality.EE,
                               "Invalid profile name '{profile}'.".format(profile=profile))
                else:
                    self.__log(Criticality.EE,
                               "The following error arose '{error}'.".format(error=run_info.exit_code.name))
                self.__release_lock()
                return

            self.state['profile'] = profile
            self.__log(Criticality.II, "Ezbench profile set to '{profile}'".format(profile=profile))
            self.__save_state()
        else:
            self.__log(Criticality.EE, "You cannot change the profile of a report that already has results. Start a new one.")
        self.__release_lock()

    def conf_scripts(self):
        return self.__read_attribute__('conf_scripts', [])

    def add_conf_script(self, conf_script):
        self.__reload_state(keep_lock=True)
        if 'beenRunBefore' not in self.state or self.state['beenRunBefore'] == False:
            if "conf_scripts" not in self.state:
                self.state['conf_scripts'] = list()

            if conf_script not in self.state['conf_scripts']:
                self.__log(Criticality.II, "Add configuration script '{0}'".format(conf_script))
                self.state['conf_scripts'].append(conf_script)
                self.__save_state()
        else:
             self.__log(Criticality.EE, "You cannot change the set of scripts of a report that already has results. Start a new one.")
        self.__release_lock()

    def remove_conf_script(self, conf_script):
        self.__reload_state(keep_lock=True)
        if 'beenRunBefore' not in self.state or self.state['beenRunBefore'] == False:
            if "conf_scripts" in self.state:
                try:
                    self.state['conf_scripts'].remove(conf_script)
                    self.__log(Criticality.II, "Remove configuration script '{0}'".format(conf_script))
                    self.__save_state()
                except:
                    pass
        else:
             self.__log(Criticality.EE, "You cannot change the set of scripts of a report that already has results. Start a new one.")
        self.__release_lock()

    def commit_url(self):
        return self.__read_attribute__('commit_url')

    def set_commit_url(self, commit_url):
        self.__write_attribute__('commit_url', commit_url, allow_updates = True)
        self.__log(Criticality.II, "Report commit URL has been changed to '{}'".format(commit_url))

    def __add_test_unlocked__(self, commit, test, rounds = None):
        if commit not in self.state['commits']:
            self.state['commits'][commit] = dict()
            self.state['commits'][commit]["tests"] = dict()

        if rounds is None:
            rounds = 3
        else:
            rounds = int(rounds)

        if test not in self.state['commits'][commit]['tests']:
            self.state['commits'][commit]['tests'][test] = dict()
            self.state['commits'][commit]['tests'][test]['rounds'] = rounds
        else:
            self.state['commits'][commit]['tests'][test]['rounds'] += rounds

        # if the number of rounds is equal to 0 for a test, delete it
        if self.state['commits'][commit]['tests'][test]['rounds'] <= 0:
            del self.state['commits'][commit]['tests'][test]

        # Delete a commit that has no test
        if len(self.state['commits'][commit]['tests']) == 0:
            del self.state['commits'][commit]

    def add_test(self, commit, test, rounds = None):
        self.__reload_state(keep_lock=True)
        self.__add_test_unlocked__(commit, test, rounds)
        self.__save_state()
        self.__release_lock()

    def add_testset(self, commit, testset, rounds = None):
        self.__reload_state(keep_lock=True)

        if rounds is None:
            rounds = 1
        else:
            rounds = int(rounds)

        for test in sorted(testset.tests.keys()):
            self.__add_test_unlocked__(commit, test,
                                            testset.tests[test] * rounds)

        self.__save_state()
        self.__release_lock()

    def __force_test_rounds_unlocked__(self, commit, test, at_least):
        if at_least < 1:
            return 0
        else:
            at_least = int(at_least)

        if commit not in self.state['commits']:
            self.state['commits'][commit] = dict()
            self.state['commits'][commit]["tests"] = dict()

        if test not in self.state['commits'][commit]['tests']:
            self.state['commits'][commit]['tests'][test] = dict()
            self.state['commits'][commit]['tests'][test]['rounds'] = 0

        to_add = at_least - self.state['commits'][commit]['tests'][test]['rounds']

        if to_add > 0:
            self.__log(Criticality.WW,
                       "Schedule {} more runs for the test {} on commit {}".format(to_add, test, commit))

            self.state['commits'][commit]['tests'][test]['rounds'] += to_add

        if to_add > 0:
            return to_add
        else:
            return 0

    def force_test_rounds(self, commit, test, at_least):
        self.__reload_state(keep_lock=True)
        ret = self.__force_test_rounds_unlocked__(commit, test, at_least)
        self.__save_state()
        self.__release_lock()

        return ret

    def task_info(self):
        self._task_lock.acquire()
        tl = copy.deepcopy(self._task_list)
        c = copy.deepcopy(self._task_current)
        self._task_lock.release()

        db = TimingsDB(self.ezbench_dir + "/timing_DB")

        # Get information about the build time and the available versions
        ezbench = self.__create_ezbench()
        versions = set(ezbench.available_versions())
        c_ts = db.data("build", self.profile())
        if len(c_ts) > 0:
            c_t = statistics.median(c_ts)
        else:
            c_t = None

        # the current task already has the timing information
        if c is not None:
            versions |= set([c.commit])
        if tl is not None:
            for t in tl:
                t.set_timing_information(db, c_t, versions)
                versions |= set([t.commit])

        return c, tl, self._events_str

    def __prioritize_runs(self, task_tree, deployed_version):
        task_list = list()

        # Aggregate all the subtests
        for commit in task_tree:
            test_subtests = dict()
            test_rounds = dict()

            # First, read all the tests and aggregate them
            for test in task_tree[commit]["tests"]:
                basename, subtests, metric = Test.parse_name(test)
                if basename not in test_subtests:
                    test_subtests[basename] = set()
                test_subtests[basename] |= set(subtests)
                test_rounds[basename] = max(test_rounds.get(basename, 0),
                                       task_tree[commit]["tests"][test]["rounds"])

            # Destroy the state before reconstructing it!
            task_tree[commit]["tests"] = dict()
            for basename in test_subtests:
                full_name = Test.partial_name(basename, list(test_subtests[basename]))
                task_tree[commit]["tests"][full_name] = dict()
                task_tree[commit]["tests"][full_name]["rounds"] = test_rounds[basename]

        # Schedule the tests using the already-deployed version
        if deployed_version is not None and deployed_version in task_tree:
            for test in task_tree[deployed_version]["tests"]:
                rounds = task_tree[deployed_version]["tests"][test]["rounds"]
                task_list.append(TaskEntry(deployed_version, test, rounds))
            del task_tree[deployed_version]

        # Add all the remaining tasks in whatever order!
        for commit in task_tree:
            for test in task_tree[commit]["tests"]:
                rounds = task_tree[commit]["tests"][test]["rounds"]
                task_list.append(TaskEntry(commit, test, rounds))

        return task_list

    def __change_state_to_run__(self):
        self.__reload_state(keep_lock=True)
        ret = False
        running_state=self.__running_mode_unlocked__()
        if running_state == RunningMode.INITIAL or running_state == RunningMode.RUNNING:
            self.__write_attribute_unlocked__('mode', RunningMode.RUN.value, allow_updates = True)
            self.__log(Criticality.II, "Ezbench running mode set to RUN")
            ret = True
        elif running_state != RunningMode.RUN:
            self.__log(Criticality.II,
                       "We cannot run when the current running mode is {mode}.".format(mode=running_state.name))
            ret = False
        else:
            ret = True
        self.__release_lock()
        return ret

    def __done_running__(self):
        self._task_current = None
        self._task_list = None
        self._task_lock.release()

    @classmethod
    def __remove_task_from_tasktree__(self, task_tree, commit, full_name, rounds):
        if commit.sha1 not in task_tree:
            return False
        if full_name not in task_tree[commit.sha1]["tests"]:
            return False

        task_tree[commit.sha1]["tests"][full_name]['rounds'] -= rounds

        if task_tree[commit.sha1]["tests"][full_name]['rounds'] <= 0:
            del task_tree[commit.sha1]["tests"][full_name]

        if len(task_tree[commit.sha1]["tests"]) == 0:
            del task_tree[commit.sha1]

        return True

    @classmethod
    def __generate_task_and_events_list__(cls, q, state, log_folder, git_history):
        exit_code = 1
        task_tree = list()
        events_str = []

        # Make sure we catch *any* error, because we need to send stuff in the
        # Queue if we do not want the parent process to get stuck
        try:
            # Generate the report, order commits based on the git history
            try:
                report = Report(log_folder, silentMode = True)
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                sys.stderr.write("\n")
                pass

            # Get the list of events
            events_str = []
            for event in report.events:
                events_str.append(str(event))

            # Walk down the report and get rid of every run that has already been made!
            task_tree = copy.deepcopy(state['commits'])
            for commit in report.commits:
                for result in commit.results:
                    for key in result.results():
                        full_name = Test.partial_name(result.test.full_name, [key])
                        SmartEzbench.__remove_task_from_tasktree__(task_tree, commit, full_name, len(result.result(key)))

            # Delete the tests on commits that do not compile
            for commit in report.commits:
                if commit.build_broken() and commit.sha1 in task_tree:
                    del task_tree[commit.sha1]

            exit_code = 0
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            sys.stderr.write("\n")
            pass

        # Return the result
        q.put((exit_code, task_tree, events_str))

    def run(self):
        self.__log(Criticality.II, "----------------------")
        self.__log(Criticality.II, "Starting a run: {report} ({path})".format(report=self.report_name, path=self.log_folder))

        # Change state to RUN or fail if we are not in the right mode
        if not self.__change_state_to_run__():
            return False

        self.__log(Criticality.II, "Checking the dependencies:")

        # check for dependencies
        if 'profile' not in self.state:
            self.__log(Criticality.EE, "    - Ezbench profile: Not set. Abort...")
            return False
        else:
            profile = self.state["profile"]
            self.__log(Criticality.II, "    - Ezbench profile: '{0}'".format(profile))

        self.__log(Criticality.II, "    - Configuration scripts: '{0}'".format(self.conf_scripts()))

        # Create the ezbench runner
        ezbench = self.__create_ezbench()
        run_info = ezbench.run(["HEAD"], [], [], dry_run=True)
        self.__log(Criticality.II, "    - Deployed version: '{0}'".format(run_info.deployed_commit))
        self.__log(Criticality.II, "All the dependencies are met, generate a report...")

        # Generate a report to compare the goal with the current state. Run it
        # in a separate process because python is really bad at freeing memory
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=SmartEzbench.__generate_task_and_events_list__,
                                    args=(q, self.state, self.log_folder, self.git_history()))
        p.start()
        exit_code, task_tree, self._events_str = q.get()
        p.join()

        if len(task_tree) == 0:
            self.__log(Criticality.II, "Nothing left to do, exit")
            return False

        task_tree_str = pprint.pformat(task_tree)
        self.__log(Criticality.II, "Task list: {tsk_str}".format(tsk_str=task_tree_str))

        # Lock the report for further changes (like for profiles)
        self.__write_attribute__('beenRunBefore', True)

        # Prioritize --> return a list of commits to do in order
        self._task_lock.acquire()
        self._task_list = self.__prioritize_runs(task_tree, run_info.deployed_commit)

        # Start generating ezbench calls
        while len(self._task_list) > 0:
            running_mode = self.running_mode(check_running = False)
            if running_mode != RunningMode.RUN:
                self.__log(Criticality.II,
                       "Running mode changed from RUN(NING) to {mode}. Exit...".format(mode=running_mode.name))
                self.__done_running__()
                return False

            self._task_current = e = self._task_list.pop(0)
            short_name=e.test[:80].rsplit('|', 1)[0]+'...'
            self.__log(Criticality.DD,
                       "make {count} runs for test {test} using commit {commit}".format(count=e.rounds,
                                                                                                  commit=e.commit,
                                                                                                  test=short_name))

            # Until we can read the output of core.sh on the fly, let's update
            # the timings for the current task by emulating what EzBench does
            ezbench = self.__create_ezbench()
            versions = set(ezbench.available_versions())
            db = TimingsDB(self.ezbench_dir + "/timing_DB")
            build_times = db.data("build", self.profile())
            if len(build_times) > 0:
                total_time = statistics.median(build_times)
            else:
                total_time = 0
            e.set_timing_information(db, total_time, versions)

            # Start the task
            self._task_current.started()
            self._task_lock.release()
            run_info = ezbench.run([e.commit], [e.test + '$'], rounds=e.rounds)
            self._task_lock.acquire()

            if run_info.success():
                continue

            # We got an error, let's see what we can do about it!
            if run_info.exit_code.value < 40:
                # Error we cannot do anything about, probably a setup issue
                # Let's mark the run as aborted until the user resets it!
                self.set_running_mode(RunningMode.ERROR)
            elif (run_info.exit_code == EzbenchExitCode.COMPILATION_FAILED or
                  run_info.exit_code == EzbenchExitCode.DEPLOYMENT_FAILED):
                # Cancel any other test on this commit
                self._task_list = [x for x in self._task_list if not x.commit == e.commit]

        self._task_current = None

        self.__done_running__()
        self.__log(Criticality.II, "Done")

        return True

    def git_history(self):
        git_history = list()

        # Get the repo directory
        ezbench = self.__create_ezbench()
        run_info = ezbench.run(["HEAD"], [], [], dry_run=True)

        if not run_info.success() or run_info.repo_dir == '':
            return git_history

        # Get the list of commits and store their position in the list in a dict
        output = subprocess.check_output(["/usr/bin/git", "log", "--first-parent", "--format=%h %ct"],
                                          cwd=run_info.repo_dir).decode().split('\n')

        for line in output:
            fields = line.split(' ')
            if len(fields) == 2:
                git_history.append(GitCommit(fields[0], fields[1]))

        return git_history

    def report(self, git_history=list(), reorder_commits = True,
               restrict_to_commits = []):
        if reorder_commits and len(git_history) == 0:
            git_history = self.git_history()

        # Generate the report, order commits based on the git history
        r = Report(self.log_folder, silentMode = True,
                                 restrict_to_commits = restrict_to_commits)
        r.enhance_report([c.sha1 for c in git_history])

        # Update the list of events with the most up to date report we have
        events_str = []
        for event in r.events:
            events_str.append(str(event))
        self._events_str = events_str

        return r

    def __find_middle_commit__(self, git_history, old, new):
        if not hasattr(self, "__find_middle_commit__cache"):
            self.__find_middle_commit__cache = dict()

        key = "{}->{}".format(old, new)
        if key in self.__find_middle_commit__cache:
            return self.__find_middle_commit__cache[key]

        old_idx = git_history.index(old)
        new_idx = git_history.index(new)
        middle_idx = int(old_idx - ((old_idx - new_idx) / 2))
        if middle_idx != old_idx and middle_idx != new_idx:
            middle = git_history[middle_idx]
        else:
            middle = None

        self.__find_middle_commit__cache[key] = middle
        return middle

    # WARNING: test may be None!
    def __score_event__(self, git_history, commit_sha1, test, severity):
        commit_weight = 1 - (git_history.index(commit_sha1) / len(git_history))

        test_weight = 1
        if test is not None and hasattr(test, 'score_weight'):
            test_weight = test.score_weight

        return commit_weight * test_weight * severity

    def schedule_enhancements(self, git_history=None, max_variance = 0.025,
                              perf_diff_confidence = 0.99, smallest_perf_change=0.005,
                              max_run_count = 20, commit_schedule_max = 1):
        self.__log(Criticality.II, "Start enhancing the report")

        # Generate the report, order commits based on the git history
        if git_history is None:
            git_history = self.git_history()
        commits_rev_order = [c.sha1 for c in git_history]
        r = Report(self.log_folder, silentMode = True)
        r.enhance_report(commits_rev_order, max_variance, perf_diff_confidence,
                         smallest_perf_change)

        # Check all events
        tasks = []
        for e in r.events:
            commit_sha1 = None
            test = None
            event_prio = 1
            severity = 0 # should be a value in [0, 1]
            test_name_to_run = ""
            runs = 0
            if type(e) is EventBuildBroken:
                if e.commit_range.old is None or e.commit_range.is_single_commit():
                    continue
                middle = self.__find_middle_commit__(commits_rev_order,
                                                     e.commit_range.old.sha1,
                                                     e.commit_range.new.sha1)
                if middle is None:
                    continue

                # Schedule the work
                commit_sha1 = middle
                severity = 1
                event_prio = 0.5
                test_name_to_run = "no-op"
                runs = 1
            elif type(e) is EventBuildFixed:
                if e.fixed_commit_range.is_single_commit():
                    continue
                middle = self.__find_middle_commit__(commits_rev_order,
                                                     e.fixed_commit_range.old.sha1,
                                                     e.fixed_commit_range.new.sha1)
                if middle is None:
                    continue

                # Schedule the work
                commit_sha1 = middle
                severity = 1
                event_prio = 0.5
                test_name_to_run = "no-op"
                runs = 1
            elif type(e) is EventPerfChange or type(e) is EventRenderingChange:
                if e.commit_range.is_single_commit():
                    continue

                # ignore commits which have a big variance
                result_new = r.find_result(e.commit_range.new, e.test).result()
                if result_new.margin() > max_variance:
                    continue
                result_old = r.find_result(e.commit_range.old, e.test).result()
                if result_old.margin() > max_variance:
                    continue

                middle = self.__find_middle_commit__(commits_rev_order,
                                                     e.commit_range.old.sha1,
                                                     e.commit_range.new.sha1)
                if middle is None:
                    continue

                # FIXME: handle the case where the middle commit refuses to build

                # Schedule the work
                commit_sha1 = middle
                test = e.test
                severity = min(abs(e.diff()), 1) * e.confidence
                event_prio = 0.75

                test_name_to_run = test.full_name
                runs = (len(result_old) + len(result_new)) / 2
            elif isinstance(e, EventResultNeedsMoreRuns):
                commit_sha1 = e.result.commit.sha1
                missing_runs = max(1, e.wanted_n() - len(e.result)) # Schedule at least 1 more runs
                severity = min(missing_runs / len(e.result), 1)
                event_prio = 1

                test_name_to_run = e.result.subtest_fullname()
                additional_runs = min(20, missing_runs) # cap the maximum amount of runs to play nice

                # Make sure we do not schedule more than the maximum amount of run
                runs = len(e.result) + additional_runs
                if runs > max_run_count:
                    runs = max_run_count - len(e.result)
                    if runs == 0:
                        continue
            elif type(e) is EventUnitResultChange:
                if e.commit_range.is_single_commit():
                    continue

                # Find the middle commit
                middle = self.__find_middle_commit__(commits_rev_order,
                                                     e.commit_range.old.sha1,
                                                     e.commit_range.new.sha1)
                if middle is None:
                    continue

                # Schedule the work
                commit_sha1 = middle
                severity = 1
                event_prio = 1
                test_name_to_run = str(e.full_name)
                runs = np.math.ceil((len(e.old_result) + len(e.new_result)) / 2)
            elif type(e) is EventUnitResultUnstable:
                # Nothing to do, for now
                continue
            else:
                print("schedule_enhancements: unknown event type {}".format(type(e).__name__))
                continue

            score = self.__score_event__(commits_rev_order, commit_sha1, test, severity)
            score *= event_prio

            tasks.append((score, commit_sha1, test_name_to_run, runs, e))

        # If we are using the throttle mode, only schedule the commit with the
        # biggest score to speed up bisecting of the most important issues
        tasks_sorted = sorted(tasks, key=lambda t: t[0])
        scheduled_commits = added = 0
        self.__reload_state(keep_lock=True)
        added = 0
        while len(tasks_sorted) > 0 and scheduled_commits < commit_schedule_max:
            commit = tasks_sorted[-1][1]
            self.__log(Criticality.DD, "Add all the tasks using commit {}".format(commit))
            for t in tasks_sorted:
                if t[1] == commit:
                    added += self.__force_test_rounds_unlocked__(t[1], t[2], t[3])
            if added > 0:
                self.__log(Criticality.II, "{}".format(t[4]))
                scheduled_commits += 1
            else:
                self.__log(Criticality.DD, "No work scheduled using commit {}, try another one".format(commit))
            del tasks_sorted[-1]
        if added > 0:
            self.__save_state()
        self.__release_lock()

        self.__log(Criticality.II, "Done enhancing the report")

        return r
