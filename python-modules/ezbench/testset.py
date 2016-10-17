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

import os
import re

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
