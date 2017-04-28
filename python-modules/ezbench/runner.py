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

from datetime import datetime
from enum import Enum
import subprocess
import fcntl
import os
import re

# Ezbench runs
class EzbenchExitCode(Enum):
    UNKNOWN = -1
    NO_ERROR = 0
    UNKNOWN_ERROR = 1
    CORE_ALREADY_RUNNING = 5
    REPORT_LOCKED = 6
    ARG_PROFILE_NAME_MISSING = 11
    ARG_PROFILE_INVALID = 12
    ARG_OPTARG_MISSING = 13
    ARG_REPO_MISSING = 14
    OS_SHELL_GLOBSTAT_MISSING = 30
    OS_LOG_FOLDER_CREATE_FAILED = 31
    OS_CD_REPO = 32
    GIT_INVALID_COMMIT_ID = 50
    ENV_SETUP_ERROR = 60
    COMP_DEP_UNK_ERROR = 70
    COMPILATION_FAILED = 71
    DEPLOYMENT_FAILED = 72
    DEPLOYMENT_ERROR = 73
    REBOOT_NEEDED = 74
    TEST_INVALID_NAME = 100
    UNK_ERROR = 255

class EzbenchRun:
    def __init__(self, commits, tests, avail_versions, predicted_execution_time, repo_type, repo_dir, repo_head, deployed_commit, exit_code):
        self.commits = commits
        self.tests = tests
        self.avail_versions = avail_versions
        self.predicted_execution_time = predicted_execution_time
        self.repo_type = repo_type
        self.repo_dir = repo_dir
        self.repo_head = repo_head
        self.deployed_commit = deployed_commit
        self.exit_code = EzbenchExitCode(exit_code)

    def success(self):
        return self.exit_code == EzbenchExitCode.NO_ERROR

class Ezbench:
    def __init__(self, ezbench_dir, profile = None, repo_path = None,
                 make_command = None, report_name = None, tests_folder = None,
                 run_config_scripts = []):
        self.ezbench_dir = ezbench_dir
        self.ezbench_path = "{}/core.sh".format(ezbench_dir)
        self.profile = profile
        self.repo_path = repo_path
        self.make_command = make_command
        self.report_name = report_name
        self.tests_folder = tests_folder
        self.run_config_scripts = run_config_scripts

        self.abortFileName = None
        if report_name is not None:
            self.abortFileName = "{}/logs/{}/requestExit".format(ezbench_dir, report_name)

    @classmethod
    def requestEarlyExit(self, ezbench_dir, report_name):
        abortFileName = "{}/logs/{}/requestExit".format(ezbench_dir, report_name)
        try:
            f = open(abortFileName, 'w')
            f.close()
            return True
        except IOError:
            return False

    def __ezbench_cmd_base(self, tests = [], test_excludes = [], rounds = None, dry_run = False, list_tests = False, list_built_versions = False):
        ezbench_cmd = []
        ezbench_cmd.append(self.ezbench_path)

        if list_tests:
            ezbench_cmd.append("-l")
            return ezbench_cmd, ""

        if self.profile is not None:
            ezbench_cmd.append("-P"); ezbench_cmd.append(self.profile)

        if list_built_versions:
            ezbench_cmd.append("-L")
            ezbench_cmd.append("-k")
            return ezbench_cmd, ""

        if self.repo_path is not None:
            ezbench_cmd.append("-p"); ezbench_cmd.append(self.repo_path)

        if len(tests) > 0:
            ezbench_cmd.append("-b"); ezbench_cmd.append("-")

        for test_excl in test_excludes:
            ezbench_cmd.append("-B"); ezbench_cmd.append(test_excl)

        if rounds is not None:
            ezbench_cmd.append("-r"); ezbench_cmd.append(str(int(rounds)))

        if self.make_command is not None:
            ezbench_cmd.append("-m"); ezbench_cmd.append(self.make_command)
        if self.report_name is not None:
            ezbench_cmd.append("-N"); ezbench_cmd.append(self.report_name)
        if self.tests_folder is not None:
            ezbench_cmd.append("-T"); ezbench_cmd.append(self.tests_folder)
        for script in self.run_config_scripts:
            ezbench_cmd.append("-c"); ezbench_cmd.append(script)

        if dry_run:
            ezbench_cmd.append("-k")

        stdin = ""
        for test in tests:
            stdin += test + "\n"

        return ezbench_cmd, stdin

    def __run_ezbench(self, cmd, stdin, dry_run = False, verbose = False):
        exit_code = None

        if verbose:
            print(cmd); print(stdin)

        # Remove the abort file before running anything as it would result in an
        # immediate exit
        if not dry_run and self.abortFileName is not None:
            try:
                os.remove(self.abortFileName)
            except FileNotFoundError:
                pass

        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT,
                                             universal_newlines=True,
                                             input=stdin)
            exit_code = EzbenchExitCode.NO_ERROR
        except subprocess.CalledProcessError as e:
            exit_code = EzbenchExitCode(e.returncode)
            output = e.output
            pass

        # we need to parse the output
        commits= []
        tests = []
        avail_versions = []
        pred_exec_time = 0
        deployed_commit = ""
        repo_type = ""
        repo_dir = ""
        head_commit = ""
        re_commit_list = re.compile('^Testing \d+ versions: ')
        re_repo = re.compile('^Repo type = (.*), directory = (.*), version = (.*), deployed version = (.*)$')
        for line in output.split("\n"):
            m_commit_list = re_commit_list.match(line)
            m_repo = re_repo.match(line)
            if line.startswith("Tests that will be run:"):
                tests = line[24:].split(" ")
            elif line.startswith("Available tests:"):
                tests = line[17:].split(" ")
            elif line.startswith("Available versions:"):
                avail_versions = line[19:].strip().split(" ")
            elif line.find("estimated finish date:") >= 0:
                pred_exec_time = ""
            elif m_repo is not None:
                repo_type, repo_dir, head_commit, deployed_commit = m_repo.groups()
            elif m_commit_list is not None:
                commits = line[m_commit_list.end():].split(" ")
                while '' in commits:
                    commits.remove('')
            elif exit_code == EzbenchExitCode.TEST_INVALID_NAME and line.endswith("do not exist"):
                print(line)

        if len(tests) > 0 and tests[-1] == '':
            tests.pop(-1)

        if exit_code != EzbenchExitCode.NO_ERROR:
            print("\n\nERROR: The following command '{}' failed with the error code {}. Here is its output:\n\n'{}'".format(" ".join(cmd), exit_code, output))

        return EzbenchRun(commits, tests, avail_versions, pred_exec_time, repo_type, repo_dir, head_commit, deployed_commit, exit_code)

    def run(self, commits, tests, test_excludes = [],
                    rounds = None, dry_run = False, verbose = False):
        ezbench_cmd, ezbench_stdin = self.__ezbench_cmd_base(tests, test_excludes, rounds, dry_run)

        for commit in commits:
            ezbench_cmd.append(commit)

        return self.__run_ezbench(ezbench_cmd, ezbench_stdin, dry_run, verbose)

    def available_tests(self):
        ezbench_cmd, ezbench_stdin = self.__ezbench_cmd_base(list_tests = True, dry_run = True)
        return self.__run_ezbench(ezbench_cmd, ezbench_stdin, dry_run = True).tests

    def available_versions(self):
        ezbench_cmd, ezbench_stdin = self.__ezbench_cmd_base(list_built_versions = True)
        return self.__run_ezbench(ezbench_cmd, ezbench_stdin, dry_run = True).avail_versions

    def reportIsLocked(self):
        if self.report_name is None:
            return False

        lockFileName = "{}/logs/{}/lock".format(self.ezbench_dir, self.report_name)

        try:
            with open(lockFileName, 'w') as lock_fd:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
                except IOError:
                    return True

                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except Exception as e:
                    pass

                return False
        except Exception:
            return False
            pass


class RunnerErrorCode(Enum):
    UNKNOWN = -1
    NO_ERROR = 0
    UNKNOWN_ERROR = 1
    CORE_ALREADY_RUNNING = 5
    REPORT_LOCKED = 6

    CMD_INVALID = 10
    CMD_PARAMETER_ALREADY_SET = 11
    CMD_PROFILE_INVALID = 12
    CMD_PROFILE_MISSING = 13
    CMD_REPORT_CREATION = 14
    CMD_REPORT_MISSING = 15
    CMD_TESTING_NOT_STARTED = 16
    CMD_TEST_EXEC_TYPE_UNSUPPORTED = 17
    CMD_TEST_EXEC_TYPE_NEED_VALID_RESULT_FILE = 18
    CMD_RESULT_ALREADY_COMPLETE = 19

    OS_SHELL_GLOBSTAT_MISSING = 30
    OS_LOG_FOLDER_CREATE_FAILED = 31
    OS_CD_REPO = 32

    GIT_INVALID_COMMIT_ID = 50

    ENV_SETUP_ERROR = 60

    COMP_DEP_UNK_ERROR = 70
    COMPILATION_FAILED = 71
    DEPLOYMENT_FAILED = 72
    DEPLOYMENT_ERROR = 73
    REBOOT_NEEDED = 74

    EARLY_EXIT = 80

    TEST_INVALID_NAME = 100

class RunnerError(Exception):
    pass

class RunnerTest:
    def __init__(self, name, test_type, unit, inverted, time_estimation):
        """
        Construct a RunnerTest, which contains information about a runner.sh
        test.

        Args:
            name: The name of the test (for example: glxgears:window)
            test_type: The type of the test (for example: unified)
            unit: The unit used by the test (for example: FPS). Only applicable
                  to the non-unified test types.
            inverted: True if lower is better, False otherwise. Only applicable
                  to the non-unified test types.
            time_estimation: Very rough estimation of how long the test may run.
        """
        self.name = name
        self.test_type = test_type
        self.unit = unit
        self.inverted = inverted
        self.time_estimation = time_estimation

    def __str__(self):
        return self.name

class RunnerCmdResult:
    def __init__(self, cmd, timestamp, err_code, err_str, exec_time, cmd_output):
        """
        Construct a RunnerCmdResult, which contains information about the
        execution of a command by

        Args:
            cmd: The command sent to runner.sh
            timestamp: Time at which the command has been sent
            err_code: Error code returned by the command. It is an instance of
                      a RunnerErrorCode
            err_str: String representing the error
            exec_time: Exact execution time of the command
            cmd_output: Output of the command
        """
        self.cmd = cmd
        self.timestamp = timestamp
        self.err_code = err_code
        self.err_str = err_str
        self.exec_time = exec_time
        self.cmd_output = cmd_output

class Runner:
    def __init__(self, ezbench_dir):
        """
        Construct an ezbench Runner.

        Args:
            ezbench_dir: Absolute path to the ezbench directory
        """
        self.ABI = 1
        self.ezbench_dir = ezbench_dir
        self.runner_path = "{}/runner.sh".format(ezbench_dir)

        # Start the runner!
        self.runner = subprocess.Popen([self.runner_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)

        runner_version = self.version()
        if runner_version != self.ABI:
            raise ValueError("The runner implements a different version ({}) than we do ({})".format(runner_version, self.ABI))

    def __del__(self):
        try:
            self.done()
        except:
            # Silently ignore errors
            pass

    def __send_command__(self, cmd):
        """
        Execute a command in the current runner, generate an exception if it
        does not execute properly.

        Returns a RunnerCmdResult object

        Args:
            cmd: The command to execute
        """
        if self.runner is None:
            return

        # Send the wanted command
        timestamp = datetime.now()
        try:
            self.runner.stdin.write(cmd + "\n")
            self.runner.stdin.flush()
        except Exception as e:
            err = "Could not send the command '{}'\n".format(cmd)
            raise RunnerError(dict({"msg": err, "err_str": err,
                                "err_code":RunnerErrorCode.UNKNOWN})) from e

        # Read the output
        cmd_output = []
        for line in self.runner.stdout:
            fields = line.rstrip().split(',')
            if fields[0] == "-->":
                pass
            elif fields[0] == "<--":
                useless, errno, errstr, time = fields

                exec_time = float(time.split(' ')[0]) / 1000.0
                err_code = RunnerErrorCode(int(errno))
                if err_code != RunnerErrorCode.NO_ERROR:
                    msg="The runner returned the error code {} ({}) for the command '{}'"
                    raise RunnerError(dict({"msg":msg.format(errno, errstr, cmd),
                                            "err_code":err_code, "err_str":errstr}))

                return RunnerCmdResult(cmd, timestamp, err_code, errstr,
                                       exec_time, cmd_output)
            else:
                cmd_output.append(line.rstrip())

        raise RunnerError(dict({"msg":"Incomplete command '{}'. Partial output is:\n{}".format(cmd, cmd_output),
                                "err_code":RunnerErrorCode.UNKNOWN,
                                "err_str":"Stream ended before we got '<--'"}))

    def __parse_cmd_output__(self, cmd_output):
        state = dict()
        for line in cmd_output:
            fields = line.split("=")
            if len(fields) == 2:
                state[fields[0]] = fields[1]
        return state

    def conf_script(self):
        errno, errstr, time, cmd_output = self.__send_command__("conf_script")
        return cmd_output

    def add_conf_script(self, path):
        self.__send_command__("conf_script,{}".format(path))

    def done(self):
        if self.runner is not None:
            self.__send_command__("done")
            self.runner.wait()
            self.runner = None

    def list_cached_versions(self):
        r = self.__send_command__("list_cached_versions")
        return r.cmd_output

    def list_tests(self):
        r = self.__send_command__("list_tests")

        tests=[]
        for t in r.cmd_output:
            name, test_type, unit, inverted, time_estimation = t.split(',')
            tests.append(RunnerTest(name, test_type, unit,
                                    (inverted == "1"), float(time_estimation)))

        return tests

    def profile(self):
        r = self.__send_command__("profile")
        return r.cmd_output[0]

    def set_profile(self, profile):
        self.__send_command__("profile,{}".format(profile))

    def reboot(self):
        self.__send_command__("reboot")

    def repo_info(self):
        r = self.__send_command__("repo")
        return self.__parse_cmd_output__(r.cmd_output)

    def report_name(self):
        r = self.__send_command__("report")
        return r.cmd_output[0]

    def set_report_name(self, report):
        self.__send_command__("report,{}".format(report))

    def resume(self, commit, test, result_file, verbose):
        if verbose:
            verbose = 1
        else:
            verbose = 0
        cmd = "resume,{},{},{},{}".format(commit, test, result_file, verbose)
        r = self.__send_command__(cmd)
        return r.exec_time, r.cmd_output[0]

    def run(self, commit, test, verbose):
        if verbose:
            verbose = 1
        else:
            verbose = 0
        r = self.__send_command__("run,{},{},{}".format(commit, test, verbose))
        return r.exec_time, r.cmd_output[0]

    def start_testing(self):
        self.__send_command__("start_testing")

    def version(self):
        r = self.__send_command__("version")
        return int(r.cmd_output[0])
