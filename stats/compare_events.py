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

from mako.lookup import TemplateLookup
from mako.template import Template
from mako import exceptions
import argparse
import sys
import os

# Import ezbench from the utils/ folder
ezbench_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(ezbench_dir, 'python-modules'))
sys.path.append(ezbench_dir)

from utils.env_dump.env_dump_parser import *
from ezbench.smartezbench import *
from ezbench.report import *

def reports_to_html(reports, output, title=None, verbose=False, embed=True):
    # Parse the results and then create one report with the following structure:
    # commit -> report_name -> test -> bench results
    events = Report.event_tree(reports)

    if title is None:
        report_names = [r.name for r in reports]
        title = "Performance report on the runs named '{run_name}'".format(
            run_name=report_names)

    templatedir = os.path.join(ezbench_dir, 'templates')

    lookup = TemplateLookup(directories=[templatedir])
    template = lookup.get_template('report.mako')

    try:
        html = template.render(title=title, events=events, embed=embed)
    except:
        html = exceptions.html_error_template().render().decode()

    with open(output, 'w') as f:
        f.write(html)
        if verbose:
            print("Output HTML generated at: {}".format(output))


def gen_report(log_folder, restrict_commits, quiet):
    report_name = os.path.basename(os.path.abspath(log_folder))

    try:
        sbench = SmartEzbench(ezbench_dir, report_name, readonly=True)
        report = sbench.report(restrict_to_commits=restrict_commits, silentMode=not quiet)
    except RuntimeError:
        report = Report(log_folder, restrict_to_commits=restrict_commits)
        report.enhance_report(NoRepo(log_folder))

    return report

if __name__ == "__main__":
    # parse the options
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", help="Set the title for the report")
    parser.add_argument("--output", help="Report html file path")
    parser.add_argument("--quiet", help="Be quiet when generating the report", action="store_true")
    parser.add_argument("--restrict_commits", help="Restrict commits to this list (space separated)")
    parser.add_argument("log_folder", nargs='+')
    args = parser.parse_args()

    # Set the output folder if not set
    if args.output is None:
        if len(args.log_folder) == 1:
            args.output = os.path.join(args.log_folder[0], 'index.html')
        else:
            print("Error: The output html file has to be specified when comparing multiple reports!")
            sys.exit(1)

    # Restrict the report to this list of commits
    restrict_commits = []
    if args.restrict_commits is not None:
        restrict_commits = args.restrict_commits.split(' ')

    reports = []
    for log_folder in set(args.log_folder):
        reports.append(gen_report(log_folder, restrict_commits, not args.quiet))

    reports_to_html(reports, args.output, args.title, not args.quiet)
