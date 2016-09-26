#!/usr/bin/env python3

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

import sys
import os

# Import ezbench from the utils/ folder
ezbench_dir = os.path.abspath(sys.path[0]+'/../')
sys.path.append(ezbench_dir+'/utils/')
sys.path.append(ezbench_dir+'/utils/env_dump')
from ezbench import *
from env_dump_parser import *

if __name__ == "__main__":
    import argparse

    # parse the options
    parser = argparse.ArgumentParser()
    parser.add_argument("log_folder")
    args = parser.parse_args()

    report = Report(args.log_folder, silentMode=True)
    report.enhance_report([])

    print("Test name, cairo image perf, xlib perf, cairo image energy, xlib energy")
    for result in report.commits[0].results:
        test_name = result.test.full_name

        if not test_name.startswith("x11:cairo:xlib:"):
            continue

        img_res = report.find_result_by_name(report.commits[0], test_name.replace("x11:cairo:xlib:", "x11:cairo:image:"))
        if img_res is None:
            img_res = report.find_result_by_name(report.commits[0], test_name.replace("x11:cairo:xlib:", "x11:cairo:ximage:"))

        test_name = test_name.replace(":xlib:", ':')
        if img_res is None:
            print("could not find the cpu result for test '{}'".format(test_name))

        perf_cpu = img_res.result().mean()
        perf_gpu = result.result().mean()

        pwr_cpu = img_res.result("metric_rapl0.package-0:energy").mean()
        pwr_gpu = result.result("metric_rapl0.package-0:energy").mean()

        print("{},{},{},{},{}".format(test_name, perf_cpu, perf_gpu, pwr_cpu, pwr_gpu))
