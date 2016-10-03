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
                 run_config_script = None):
        self.ezbench_dir = ezbench_dir
        self.ezbench_path = "{}/core.sh".format(ezbench_dir)
        self.profile = profile
        self.repo_path = repo_path
        self.make_command = make_command
        self.report_name = report_name
        self.tests_folder = tests_folder
        self.run_config_script = run_config_script

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
        if self.run_config_script is not None:
            ezbench_cmd.append("-c"); ezbench_cmd.append(self.run_config_script)

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
        return self.__run_ezbench(ezbench_cmd, ezbench_stdin).avail_versions

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


# Test sets, needed by SmartEzbench
class Testset:
    def __init__(self, filepath, name):
        self.filepath = filepath
        self.name = name
        self.description = "No description"
        self.tests = dict()

        self._ln = -1

    def __print__(self, msg, silent = False):
        if not silent:
            print("At {}:{}, {}".format(self.filepath, self._ln, msg))

    def __include_set__(self, availableTestSet, reg_exp, rounds, silent = False):
        # Convert the rounds number to integer and validate it
        try:
            rounds = int(rounds)
            if rounds < 0:
                self.__print__("the number of rounds cannot be negative ({})".format(rounds), silent)
                return False
        except ValueError:
            self.__print__("the number of rounds is invalid ({})".format(rounds), silent)
            return False

        # Now add the tests needed
        try:
            inc_re = re.compile(reg_exp)
        except Exception as e:
            self.__print__("invalid regular expression --> {}".format(e), silent)
        tests_added = 0
        for test in availableTestSet:
            if inc_re.search(test):
                self.tests[test] = rounds
                tests_added += 1

        if tests_added == 0:
            self.__print__("no tests got added", silent)
            return False
        else:
            return True

    def __exclude_set__(self, reg_exp, silent = False):
        # Now remove the tests needed
        try:
            inc_re = re.compile(reg_exp)
        except Exception as e:
            self.__print__("invalid regular expression --> {}".format(e), silent)

        to_remove = []
        for test in self.tests:
            if inc_re.search(test):
                to_remove.append(test)

        if len(to_remove) > 0:
            for entry in to_remove:
                del self.tests[entry]
        else:
            self.__print__("exclude '{}' has no effect".format(reg_exp), silent)

        return True

    def parse(self, availableTestSet, silent = False):
        try:
            with open(self.filepath) as f:
                self._ln = 1
                for line in f.readlines():
                    fields = line.split(" ")
                    if fields[0] == "description":
                        if len(fields) < 2:
                            self.__print__("description takes 1 argument", silent)
                            return False
                        self.description = " ".join(fields[1:])
                    elif fields[0] == "include":
                        if availableTestSet is None:
                            continue
                        if len(fields) != 3:
                            self.__print__("include takes 2 arguments", silent)
                            return False
                        if not self.__include_set__(availableTestSet, fields[1], fields[2], silent):
                            return False
                    elif fields[0] == "exclude":
                        if availableTestSet is None:
                            continue
                        if len(fields) != 2:
                            self.__print__("exclude takes 1 argument", silent)
                            return False
                        if not self.__exclude_set__(fields[1].strip(), silent):
                            return False
                    elif fields[0] != "\n" and fields[0][0] != "#":
                        self.__print__("invalid line", silent)
                    self._ln += 1

                return True
        except EnvironmentError:
            return False

    @classmethod
    def list(cls, ezbench_dir):
        testsets = []
        for root, dirs, files in os.walk(ezbench_dir + '/testsets.d/'):
            for f in files:
                if f.endswith(".testset"):
                    testsets.append(cls(root + f, f[0:-8]))

        return testsets

    @classmethod
    def open(cls, ezbench_dir, name):
        filename = name + ".testset"
        for root, dirs, files in os.walk(ezbench_dir + '/testsets.d/'):
            if filename in files:
                return cls(root + '/' + filename, name)
        return None
