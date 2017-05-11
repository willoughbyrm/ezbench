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
import collections
import sys
import os

# Import ezbench from the utils/ folder
ezbench_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(ezbench_dir, 'python-modules'))
sys.path.append(ezbench_dir)

from utils.env_dump.env_dump_parser import *
from ezbench.smartezbench import *
from ezbench.report import *

# constants
html_name="index.html"

def __env_add_result__(db, human_envs, report, commit, result):
	if result.test.full_name not in human_envs:
		for run in result.runs:
			envfile = run.env_file
			if envfile is not None:
				fullpath = report.log_folder + "/" + envfile
				human_envs[result.test.full_name] = EnvDumpReport(fullpath, True)
	if result.test.full_name not in db['env_sets']:
		db['env_sets'][result.test.full_name] = list()
	for e in range(0, len(result.runs)):
		# Create the per-run information
		envfile = result.runs[e].env_file
		if envfile is None:
			continue

		fullpath = report.log_folder + "/" + envfile
		r = EnvDumpReport(fullpath, False).to_set(['^DATE',
													'^ENV.ENV_DUMP_FILE',
													'^ENV.ENV_DUMP_METRIC_FILE',
													'^ENV.EZBENCH_CONF_.*\.key$',
													'_PID',
													'SHA1$',
													'.pid$',
													'X\'s pid$',
													'extension count$',
													'window id$'])
		tup = dict()
		tup['log_folder'] = report.name
		tup['commit'] = commit
		tup['run'] = e

		# Compare the set to existing ones
		found = False
		for r_set in db['env_sets'][result.test.full_name]:
			if r  == r_set['set']:
				r_set['users'].append(tup)
				found = True

		# Add the report
		if not found:
			new_entry = dict()
			new_entry['set'] = r
			new_entry['users'] = [tup]
			db['env_sets'][result.test.full_name].append(new_entry)

def reports_to_html(reports, output, output_unit = None, title = None,
			   commit_url = None, verbose = False, reference_report = None,
			   reference_commit = None, embed = True):

	# select the right unit
	if output_unit is None:
		output_unit = "FPS"

	# Parse the results and then create one report with the following structure:
	# commit -> report_name -> test -> bench results
	db = dict()
	db["commits"] = collections.OrderedDict()
	db["reports"] = list()
	db["events"] = dict()
	db["tests"] = list()
	db["metrics"] = dict()
	db['env_sets'] = dict()
	db["envs"] = dict()
	db["targets"] = dict()
	db["targets_raw"] = dict()
	db["target_result"] = dict()
	human_envs = dict()

	if reference_report is None and reference_commit is not None:
		reference_report = reports[0]

	# set all the targets
	if reference_report is not None and len(reference_report.commits) > 0:
		if reference_commit is not None:
			ref_commit = reference_report.find_commit_by_id(reference_commit)
		else:
			ref_commit = reference_report.commits[-1]

		db['reference_name'] = "{} ({})".format(reference_report.name, ref_commit.label)
		db['reference'] = reference_report
		for result in ref_commit.results.values():
			average_raw = result.result().mean()
			average = convert_unit(average_raw, result.unit, output_unit)
			average = float("{0:.2f}".format(average))
			average_raw = float("{0:.2f}".format(average_raw))
			if (not result.test.full_name in db["targets"] or
				db["targets"][result.test.full_name] == 0):
					db["targets"][result.test.full_name] = average
					db["targets_raw"][result.test.full_name] = average_raw
					db["target_result"][result.test.full_name] = result

			__env_add_result__(db, human_envs, reference_report, ref_commit, result)

	for report in reports:
		report.events = [e for e in report.events if not isinstance(e, EventResultNeedsMoreRuns)]

	db["events"] = Report.event_tree(reports)

	for report in reports:
		db["reports"].append(report)

		# make sure all the tests are listed in db["envs"]
		for test in report.tests:
			db["envs"][test.full_name] = dict()

		for event in report.events:
			if type(event) is EventPerfChange:
				for result in event.commit_range.new.results.values():
					if result.test.full_name != event.test.full_name:
						continue
					result.annotation = str(event)

		# add all the commits
		for commit in report.commits:
			if len(commit.results) == 0 and not hasattr(commit, 'annotation'):
				continue

			if not commit.label in db["commits"]:
				db["commits"][commit.label] = dict()
				db["commits"][commit.label]['reports'] = dict()
				db["commits"][commit.label]['commit'] = commit
				if not commit.build_broken():
					db["commits"][commit.label]['build_color'] = "#00FF00"
				else:
					db["commits"][commit.label]['build_color'] = "#FF0000"
				db["commits"][commit.label]['build_error'] = str(RunnerErrorCode(commit.compil_exit_code)).split('.')[1]
			db["commits"][commit.label]['reports'][report.name] = dict()

			# Add the results and perform some stats
			score_sum = 0
			count = 0
			for result in commit.results.values():
				if not result.test.full_name in db["tests"]:
					db["tests"].append(result.test.full_name)
					db["metrics"][result.test.full_name] = []
				db["commits"][commit.label]['reports'][report.name][result.test.full_name] = result
				average_raw = result.result().mean()
				if average_raw is not None and result.unit is not None:
					average = convert_unit(average_raw, result.unit, output_unit)
				else:
					average_raw = 0
					average = 0
					result.unit = "unknown"
				score_sum += average
				count += 1

				result.average_raw = float("{0:.2f}".format(average_raw))
				result.average = float("{0:.2f}".format(average))
				result.margin_str = float("{0:.2f}".format(result.result().margin() * 100))

				# Compare to the target
				if (not result.test.full_name in db["targets"] or
				(db["targets"][result.test.full_name] == 0 and 'reference_name' not in db)):
					db["targets"][result.test.full_name] = result.average
					db["targets_raw"][result.test.full_name] = result.average_raw
				result.diff_target = compute_perf_difference(output_unit,
				                                             db["targets"][result.test.full_name],
				                                             result.average)

				# Get the metrics
				result.metrics = dict()
				for metric in result.results(BenchSubTestType.METRIC):
					if metric not in db["metrics"][result.test.full_name]:
						db["metrics"][result.test.full_name].append(metric)

					result.metrics[metric] = result.result(metric)


				# Environment
				__env_add_result__(db, human_envs, report, commit, result)

			if count > 0:
				avg = score_sum / count
			else:
				avg = 0
			db["commits"][commit.label]['reports'][report.name]["average"] = float("{0:.2f}".format(avg))
			db["commits"][commit.label]['reports'][report.name]["average_unit"] = output_unit

	# Generate the environment
	for bench in human_envs:
		env = human_envs[bench]
		if env is not None:
			for key in sorted(list(env.values)):
				if not bench in db['envs']:
					continue
				cur = db['envs'][bench]
				fields = key.split(":")
				for f in range(0, len(fields)):
					field = fields[f].strip()
					if f < len(fields) - 1:
						if field not in cur:
							cur[field] = dict()
						cur = cur[field]
					else:
						cur[field] = env.values[key]

	# Generate the environment diffs
	db['env_diff_keys'] = dict()
	for bench in db['env_sets']:
		final_union = set()
		for report in db['env_sets'][bench]:
			diff = db['env_sets'][bench][0]['set'] ^ report['set']
			final_union = final_union | diff
		db['env_diff_keys'][bench] = sorted(dict(final_union).keys())

	# Sort the tests by name to avoid ever-changing layouts
	db["tests"] = np.sort(db["tests"])

	# Support creating new URLs
	if commit_url is not None:
		db["commit_url"] = commit_url

	# Generate the report
	html_template="""
	<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
	"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">

	<%! import os %>
	<%! import cgi %>
	<%! import html %>
	<%! import base64 %>
	<%! from ezbench.smartezbench import compute_perf_difference, BenchSubTestType, Event, EventRenderingChange %>

	<html xmlns="http://www.w3.org/1999/xhtml">
		<head>
			<title>${title}</title>
			<meta http-equiv="content-type" content="text/html; charset=utf-8" />
			<style>
				body { font-size: 10pt; }
				table { font-size: 8pt; }

				/* http://martinivanov.net/2011/09/26/css3-treevew-no-javascript/ */
				.css-treeview input + label + ul
				{
					display: none;
				}
				.css-treeview input:checked + label + ul
				{
					display: block;
				}
				.css-treeview input
				{
					position: absolute;
					opacity: 0;
				}
				.css-treeview label,
				.css-treeview label::before
				{
					cursor: pointer;
				}
				.css-treeview input:disabled + label
				{
					cursor: default;
					opacity: .6;
				}
				table{
					border-collapse:collapse;
				}
				table td{
					padding:5px; border:#4e95f4 1px solid;
				}
				table tr:nth-child(odd){
					background: #b8d1f3;
				}
				table tr:nth-child(even){
					background: #dae5f4;
				}

				.tree_node:hover {
					cursor: pointer;
					text-decoration: underline;
				}

				.close_button {
					color: black;
					background-color: grey;
					cursor:pointer;
				}
				.close_button:hover {
					text-decoration:underline;
				}

				/* env_dump*/
				.ed_c {
					color: black;
				}
				.ed_nc {
					color: gray;
				}
			</style>
			<script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
			<script type="text/javascript">
				google.charts.load('current', {'packages':['corechart', 'table']});
				google.charts.setOnLoadCallback(drawTrend);
				google.charts.setOnLoadCallback(drawDetails);
				google.charts.setOnLoadCallback(drawTable);

				var currentCommit = "${default_commit}";

				function showColumn(dataTable, chart, activColumns, series, col, show) {
					seriesCol = 0
					for (i = 0; i < col; i++)
						if (dataTable.getColumnType(i) == 'number')
							seriesCol++
					if (!show) {
						activColumns[col] = {
							label: dataTable.getColumnLabel(col),
							type: dataTable.getColumnType(col),
							calc: function () {
								return null;
							}
						};
						series[seriesCol].color = '#CCCCCC';
					}
					else {
						activColumns[col] = col;
						series[seriesCol].color = null;
					}
				}

				function showAllColumns(dataTable, chart, activColumns, series, show) {
					for (var i = 1; i < dataTable.getNumberOfColumns(); i++) {
						if (dataTable.getColumnType(i) == 'number')
							showColumn(dataTable, chart, activColumns, series, i, show)
					}
				}

				function handle_selection(sel, dataTable, series, activColumns, chart) {
					var col = sel[0].column;

					var allActive = true;
					for (var i = 1; i < dataTable.getNumberOfColumns(); i++) {
						if (dataTable.getColumnType(i) == 'number' && activColumns[i] != i) {
							allActive = false;
						}
					}
					if (activColumns[col] == col) {
						// The clicked columns is active
						if (allActive) {
							showAllColumns(dataTable, chart, activColumns, series, false);
							showColumn(dataTable, chart, activColumns, series, col, true);
						} else {
							showColumn(dataTable, chart, activColumns, series, col, false);
						}
					}
					else {
						// The clicked columns is inactive, show it
						showColumn(dataTable, chart, activColumns, series, col, true);
					}

					var activeColsCount = 0;
					for (var i = 1; i < dataTable.getNumberOfColumns(); i++) {
						if (dataTable.getColumnType(i) == 'number' && activColumns[i] == i) {
							activeColsCount++;
						}
					}
					if (activeColsCount == 0)
						showAllColumns(dataTable, chart, activColumns, series, true);

					return activeColsCount
				}

				function adjustChartSize(id_chart, reportsCount, testCount) {
					var size = 75 + reportsCount * (25 + (testCount * 8));
					id_chart.style.height = size + "px";
					id_chart.style.width = "100%";
				}

				function trendUnselect() {
					trend_chart.setSelection(null);
				}

				function drawTrend() {
					var dataTable = new google.visualization.DataTable();

	<%def name="tooltip_commit_table(commit)">\\
	<h3>${db["commits"][commit]['commit'].full_name.replace('"', '&quot;')} <span class='close_button' onclick='javascript:trendUnselect();' title='Close this tooltip'>[X]</span></h3>\\
	<h4>Commit\\
	% if 'commit_url' in db:
	(<a href='${db["commit_url"].format(commit=commit)}' target='_blank'>URL</a>)\\
	% endif
	</h4><table>\\
	<tr><td><b>Author:</b></td><td>${cgi.escape(db["commits"][commit]['commit'].author)}</td></tr>\\
	<tr><td><b>Commit date:</b></td><td>${db["commits"][commit]['commit'].commit_date}</td></tr>\\
	<tr><td><b>Build exit code:</b></td><td bgcolor='${db["commits"][commit]['build_color']}'><center>${db["commits"][commit]['build_error']}</center></td></tr>\\
	% if len(db["commits"][commit]['commit'].bugs) > 0:
	<tr><td><b>Referenced bugs</b></td><td><ul>\\
	% for bug in db["commits"][commit]['commit'].bugs:
	<li><a href='${bug.replace('"', '&quot;')}' target='_blank'>${bug.replace('"', '&quot;')}</a></li>\\
	% endfor
	</ul></td></tr>\\
	% endif
	% if hasattr(db["commits"][commit]['commit'], "annotation_long"):
	<tr><td><b>Annotation:</b></td><td>${cgi.escape(db["commits"][commit]['commit'].annotation_long)}</td></tr>\\
	% endif
	</table>\\
	</%def>

	% if len(db['reports']) > 1:
					dataTable.addColumn('string', 'Commit');
					dataTable.addColumn({type: 'string', role: 'tooltip', p: { html: true }});
					% for report in db["reports"]:
					dataTable.addColumn('number', '${report.name}');
					% endfor
					dataTable.addRows([
					% for commit in db["commits"]:
						['${commit}', "${tooltip_commit_table(commit)}<h4>Perf</h4><table>\\
	% for report in db["reports"]:
	% if report.name in db["commits"][commit]['reports']:
	<tr><td><b>${report.name}:</b></td><td>${db["commits"][commit]['reports'][report.name]["average"]} ${output_unit}</td></tr>\\
	% endif
	% endfor
	</table><p></p>"\\
							% for report in db["reports"]:
								% if report.name in db["commits"][commit]['reports']:
	, ${db["commits"][commit]['reports'][report.name]["average"]}\\
								% else:
	, null\\
								% endif
							% endfor
	],
					% endfor
					]);
	% else:
					<%
						report = db["reports"][0].name
					%>
					dataTable.addColumn('string', 'Commits');
					dataTable.addColumn({type: 'string', role:'annotation'});
					% for test in db["tests"]:
					dataTable.addColumn('number', '${test}');
					dataTable.addColumn({type: 'string', role: 'tooltip', p: { html: true }});
					% endfor

					dataTable.addRows([
					% for commit in db["commits"]:
	["${commit}"\\
						% if hasattr(db["commits"][commit]['commit'], 'annotation'):
	, "${db["commits"][commit]['commit'].annotation}"\\
						%else:
	, null\\
						% endif
						% for test in db["tests"]:
							% if test in db["commits"][commit]['reports'][report]:
	<%
		result = db["commits"][commit]['reports'][report][test]
		diff_target = "{0:.2f}".format(result.diff_target)
	%>\\
	, ${diff_target}, "${tooltip_commit_table(commit)}<h4>Perf</h4><table><tr><td><b>Test</b></td><td>${test}</td></tr><tr><td><b>Target</b></td><td>${db['targets'][test]} ${output_unit} (${diff_target}%)</td></tr><tr><td><b>Raw value</b></td><td>${result.average_raw} ${result.unit} +/- ${result.margin_str}% (n=${len(result.result())})</td></tr><tr><td><b>Converted value</b></td><td>${result.average} ${output_unit} +/- ${result.margin_str}% (n=${len(result.result())})</td></tr></table><br/>"\\
								% else:
	, null, "${test}"\\
								% endif
						% endfor
	],
					% endfor
					]);
	% endif

					var activColumns = [];
					var series = {};
					for (var i = 0; i < dataTable.getNumberOfColumns(); i++) {
						activColumns.push(i);
						if (i > 0) {
							series[i - 1] = {};
						}
					}

					var options = {
					chart: {
							title: 'Performance trend across multiple commits'
						},
					% if len(db['reports']) > 1:
						focusTarget: 'category',
						vAxis: {title: 'Average result (${output_unit})'},
					% else:
						annotations: {style: 'line', textStyle: {fontSize: 12}},
						vAxis: {title: '% of target (%)'},
					% endif
						legend: { position: 'top', textStyle: {fontSize: 12}, maxLines: 3},
						tooltip: {trigger: 'selection', isHtml: true},
						crosshair: { trigger: 'both' },
						hAxis: {title: 'Commits', slantedText: true, slantedTextAngle: 45},
						series: series,
						chartArea: {left:"6%", width:"95%"}
					};

					trend_chart = new google.visualization.LineChart(document.getElementById('trends_chart'));
					trend_chart.draw(dataTable, options);

					google.visualization.events.addListener(trend_chart, 'select', function () {
						var sel = trend_chart.getSelection();
						// See https://developers.google.com/chart/interactive/docs/reference#visgetselection
						if (sel.length > 0 && typeof sel[0].row === 'object') {
							handle_selection(sel, dataTable, series, activColumns, trend_chart)

							// Redraw the chart with the masked columns
							var view = new google.visualization.DataView(dataTable);
							view.setColumns(activColumns);
							trend_chart.draw(view, options);
						}

						if (sel.length > 0 && typeof sel[0].row === 'number') {
							// update the other graph if there were changes
							var commit = dataTable.getValue(sel[0].row, 0)
							if (commit != currentCommit) {
								currentCommit = commit;
								drawDetails();
								drawTable();
							}
						}

						if (sel.length == 0) {
							trend_chart.setSelection(null);
						}
					});
				}

			function drawDetails() {
					var dataTable = new google.visualization.DataTable();
					dataTable.addColumn('string', 'Report');
					dataTable.addColumn('number', 'Average');
					dataTable.addColumn({type: 'string', role: 'tooltip', p: { html: true }});
					% for test in db["tests"]:
					dataTable.addColumn('number', '${test}');
					dataTable.addColumn({type: 'string', role: 'tooltip', p: { html: true }});
					% endfor

					% for commit in db["commits"]:
					if (currentCommit == "${commit}") {
						dataTable.addRows([
						% for report in db["reports"]:
							% if report.name in db["commits"][commit]['reports']:
								["${report.name}", ${db["commits"][commit]['reports'][report.name]["average"]}, "<h3>${report.name} - Average</h3><p>\\
								% for r in db["reports"]:
	<%
										if not r.name in db["commits"][commit]:
											continue
										diff = compute_perf_difference(output_unit, db["commits"][commit]['reports'][report.name]["average"], db["commits"][commit]['reports'][r.name]["average"])
										diff = float("{0:.2f}".format(diff))
										btag = btagend = ""
										if r.name == report.name:
											btag="<b>"
											btagend="</b>"
									%>\\
	${btag}${r.name}: ${db["commits"][commit]['reports'][r.name]["average"]} ${output_unit} (${diff}%)${btagend}<br/>\\
								% endfor
	</p>"\\
								% for test in db["tests"]:
									% if report.name in db["commits"][commit]['reports'] and test in db["commits"][commit]['reports'][report.name]:
	, ${db["commits"][commit]['reports'][report.name][test].average}, "<h3>${report.name} - ${test}</h3><p>\\
										% for r in db["reports"]:
	<%
												if not r.name in db["commits"][commit]['reports'] or test not in db["commits"][commit]['reports'][r.name]:
													continue
												diff = compute_perf_difference(output_unit, db["commits"][commit]['reports'][report.name][test].average, db["commits"][commit]['reports'][r.name][test].average)
												diff = float("{0:.2f}".format(diff))
												btag = btagend = ""
												if r.name == report.name:
													btag="<b>"
													btagend="</b>"
											%>\\
	${btag}${r.name}: ${db["commits"][commit]['reports'][r.name][test].average} ${output_unit} (${diff}%)${btagend}<br/>\\
										% endfor
	</p>"\\
							% else:
	, null, "${test}"\\
							% endif
							% endfor
	],
							% endif
						% endfor
						]);
					}
					% endfor

					// adjust the size of the chart to fit the data
					adjustChartSize(details_chart, dataTable.getNumberOfRows(), Math.floor(dataTable.getNumberOfColumns() / 2));

					var activColumns = [];
					var series = {};
					for (var i = 0; i < dataTable.getNumberOfColumns(); i++) {
						activColumns.push(i);
						if (i > 0) {
							series[i - 1] = {};
						}
					}
					series[0] = {type: 'line'};


					var options = {
						title : 'Performance of commit ' + currentCommit,
						legend: {textStyle: {fontSize: 12}},
						tooltip: {trigger: 'focus', isHtml: true},
						vAxis: {title: 'Reports', textStyle: {fontSize: 12}},
						hAxis: {title: 'Average result (${output_unit})', textStyle: {fontSize: 12}},
						seriesType: 'bars',
						orientation: 'vertical',
						series: series
					};

					var chart = new google.visualization.ComboChart(document.getElementById('details_chart'));
					chart.draw(dataTable, options);

					google.visualization.events.addListener(chart, 'select', function () {
						var sel = chart.getSelection();
						// See https://developers.google.com/chart/interactive/docs/reference#visgetselection
						if (sel.length > 0 && typeof sel[0].row === 'object') {
							activeCols = handle_selection(sel, dataTable, series, activColumns, chart)

							// reduce the size of the chart to fit the data
							adjustChartSize(details_chart, dataTable.getNumberOfRows(), activeCols);

							// Redraw the chart with the masked columns
							var view = new google.visualization.DataView(dataTable);
							view.setColumns(activColumns);
							chart.draw(view, options);
						}

						if (sel.length == 0) {
							chart.setSelection(null);
						}
					});
				}

				function drawTable() {
					% if len(db["reports"]) > 1:
						var dataTable = new google.visualization.DataTable();
						dataTable.addColumn('string', 'Test');
						dataTable.addColumn('string', 'Report 1');
						dataTable.addColumn('string', 'Report 2');
						dataTable.addColumn('number', '%');
						dataTable.addColumn('string', 'Comments');

						% for commit in db["commits"]:
						if (currentCommit == "${commit}") {
							% for report1 in db["reports"]:
								% if report1.name in db["commits"][commit]['reports']:
									% for report2 in db["reports"]:
										% if report2.name != report1.name and report2.name in db["commits"][commit]['reports']:
											% for test in db["tests"]:
												% if (test in db["commits"][commit]['reports'][report1.name] and test in db["commits"][commit]['reports'][report2.name]):
												<%
													r1 = db["commits"][commit]['reports'][report1.name][test]
													r2 = db["commits"][commit]['reports'][report2.name][test]
													perf_diff = compute_perf_difference(r1.unit, r1.average_raw, r2.average_raw)
													perf_diff = "{0:.2f}".format(perf_diff)
												%>
							dataTable.addRows([['${test}', '${report1.name}', '${report2.name}', ${perf_diff}, "${r1.average_raw} => ${r2.average_raw} ${r1.unit}"]])
												% endif
											% endfor
										% endif
									% endfor
								% endif
							% endfor
						}
						%endfor
					% else:
						var dataTable = new google.visualization.DataTable();
						dataTable.addColumn('string', 'Test');
						dataTable.addColumn('string', 'Report');
						dataTable.addColumn('number', '% of target');
						dataTable.addColumn('string', 'Comments');

						% for commit in db["commits"]:
						if (currentCommit == "${commit}") {
							% for report1 in db["reports"]:
								% if report1.name in db["commits"][commit]['reports']:
									% for test in db["tests"]:
										% if (test in db["commits"][commit]['reports'][report1.name] and test in db["targets"]):
										<%
											r1 = db["commits"][commit]['reports'][report1.name][test]
											perf_diff = compute_perf_difference(r1.unit, db["targets_raw"][test], r1.average_raw)
											perf_diff = "{0:.2f}".format(perf_diff)
										%>\\
dataTable.addRows([['${test}', '${report1.name}', ${perf_diff}, "${r1.average_raw}(${report1.name}) => ${db["targets_raw"][test]}(target) ${r1.unit}"]])
										% endif
									% endfor
								% endif
							% endfor
						}
						%endfor
					% endif
					var chart = new google.visualization.Table(document.getElementById('details_table'));
					chart.draw(dataTable, {showRowNumber: true, width: '100%', height: '100%'});
				}
			</script>
		</head>

		<body>
			<h1>${title}</h1>

			% if 'reference_name' in db:
				<p>With targets taken from: ${db['reference_name']}</p>
			% endif

			<h2>Trends</h2>

			<center><div id="trends_chart" style="width: 100%; height: 500px;"></div></center>

			<h2>Details</h2>

			<center><div id="details_chart" style="width: 100%; height: 500px;"></div></center>

			<center><div id="details_table" style="width: 100%; height: 500px;"></div></center>

			<h2>Events</h2>
			<%include file="events.mako"/>

			<h2>Tests</h2>

				% for test in db["tests"]:
					<h3>${test}</h3>

					<div class="css-treeview">
						<%def name="outputtreenode(node, id, label, attr = '')">
							<li><input type="checkbox" id="${id}" ${attr}/><label class="tree_node" for="${id}">+${label}</label><ul>
								<table>
								% for child in sorted(node):
									% if type(node[child]) is str:
										<tr><td>${child}</td><td>${node[child]}</td></tr>
									% endif
								% endfor
								</table>
								% for child in sorted(node):
									% if type(node[child]) is not str:
										${outputtreenode(node[child], "{}.{}".format(id, child.replace(' ', '_')), child, '')}
									% endif
								% endfor
							</ul></li>
						</%def>

						<ul>
							${outputtreenode(db["envs"][test], test + "_envroot", "Environment", 'checked="checked"')}
						</ul>
					</div>

					<table>
						<tr>
							<th>Key</th>
							% for env_set in db["env_sets"][test]:
							<%
								users = ""
								for user in env_set['users']:
									if len(users) > 0:
										users += "<br/>"
									users += "{}.{}#{}".format(user['log_folder'], user['commit'].label, user['run'])
							%>\\
							<th>${users}</th>
							% endfor
						</tr>
						% for key in db["env_diff_keys"][test]:
						<tr>
							<td>${key}</td>
							<%
								prev = None
							%>
							% for env_set in db["env_sets"][test]:
							<%
								if key in dict(env_set['set']):
									env_val = dict(env_set['set'])[key]
								else:
									env_val = "MISSING"

								if prev is None or env_val != prev:
									css_class = "ed_c"
								else:
									css_class = "ed_nc"
								prev = env_val
							%>
							<td class="${css_class}">${env_val}</td>
							% endfor
						</tr>
						% endfor
					</table>

					<h3>Metrics</h3>
					% for report in db["reports"]:
						<h4>${report.name}</h4>
						<table>
							<tr><th>Metric Name</th>
							% if 'reference' in db:
							<th>Target</th>
							%endif
							% for commit in db["commits"]:
							% if report.name in db["commits"][commit]['reports'] and test in db["commits"][commit]['reports'][report.name]:
							<th>${commit}</th>
							% endif
							% endfor

							% for metric in sorted(db["metrics"][test]):
<% ref_metric = None %>\\
								<tr><td>${metric}</td>
								% if 'reference' in db:
									% if (test in db["target_result"] and (metric in db["target_result"][test].results(BenchSubTestType.METRIC))):
									<%
										ref_metric = db["target_result"][test].result(metric)
									%>
									<td>${str(ref_metric)}</td>
									% else:
									<td>N/A</td>
									% endif
								%endif
								% for commit in db["commits"]:
									% if report.name in db["commits"][commit]['reports'] and test in db["commits"][commit]['reports'][report.name]:
									% if metric in db["commits"][commit]['reports'][report.name][test].results(BenchSubTestType.METRIC):
									<%
										m = db["commits"][commit]['reports'][report.name][test].metrics[metric]
									%>
										<td>${str(m)}\\
										% if ref_metric is not None:
<%
											diff = compute_perf_difference(unit, ref_metric.mean(), m.mean())
										%>${" ({:.2f}%)".format(diff)}\\
										% endif
</td>
									% else:
										<td>N/A</td>
									% endif
									% endif
								% endfor
								</tr>
							% endfor
						</table>
					%endfor

					<h3>Results</h3>
					<%
						unit_results = []
						stats_status = dict()
						statuses = set()

						target_changes = dict()
						changes = set()

						# Add the target report in the list of reports if it
						# contains tests for this test
						target_result = None
						if 'target_result' in db and test in db['target_result']:
							subtests = db['target_result'][test].results(BenchSubTestType.SUBTEST_STRING)
							if len(subtests) > 0:
								target_result = db['target_result'][test]
								target_result.name = "Target"
								stats_status[target_result.name] = dict()
								unit_results.append(target_result)

						for report in db['reports']:
							for commit in report.commits:
								for result in commit.results.values():
									if result.test.full_name != test:
										continue
									if result.test_type != "unit":
										continue
									result.name = "{}.{}".format(report.name, commit.label)
									stats_status[result.name] = dict()
									target_changes[result.name] = dict()
									unit_results.append(result)

						all_tests = set()
						for result in unit_results:
							all_tests |= set(result.results(BenchSubTestType.SUBTEST_STRING))
							result.unit_results = dict()

						unit_tests = set()
						for test in all_tests:
							value = None
							for result in unit_results:
								if "<" in test: # Hide subsubtests
									continue
								subtest = result.result(test)
								if subtest is None or len(subtest) == 0:
									status = "missing"
								else:
									if len(subtest.to_set()) == 1:
										status = subtest[0]
									else:
										status = "unstable"
								result.unit_results[test] = status

								# Collect stats on all the status
								if status not in stats_status[result.name]:
									stats_status[result.name][status] = 0
									statuses |= set([status])
								stats_status[result.name][status] += 1

								if value == None and status != "missing":
									value = status
									continue
								if value != status and status != "missing":
									unit_tests |= set([test])

								if (target_result is None or result == target_result or
									target_result.unit_results[test] == status):
									continue

								change = "{} -> {}".format(target_result.unit_results[test],
								                           status)
								if change not in target_changes[result.name]:
									target_changes[result.name][change] = 0
									changes |= set([change])
								target_changes[result.name][change] += 1

						all_tests = []
					%>

					% if len(unit_results) > 0:
					<h4>Unit tests</h4>
						<h5>Basic stats</h4>
						<table>
						<tr><th>Version</th>
						% for status in sorted(statuses):
							<th>${status}</th>
						% endfor
						</tr>

						% for result in stats_status:
						<tr><td>${result}</td>\\
							% for status in sorted(statuses):
								% if status in stats_status[result]:
<td>${stats_status[result][status]}</td>\\
								% else:
<td>0</td>\\
								% endif
							% endfor
						</tr>
						% endfor
						</table>

						% if 'target_result' in db and test in db['target_result']:
						<h5>Status changes</h4>
						<table>
						<tr><th>Version</th>
						% for result in target_changes:
							<th>${result}</th>
						% endfor
						</tr>

						% for change in sorted(changes):
						<tr><td>${change}</td>
							% for result in target_changes:
								% if change in target_changes[result]:
							<td>${target_changes[result][change]}</td>
								% else:
							<td>0</td>
								% endif
							% endfor
						</tr>
						% endfor
						</table>
						% endif

						<h5>Changes</h4>
						<div style='overflow:auto; width:100%;max-height:1000px;'>
						<table>
							<tr><th>test name (${len(unit_tests)})</th>
							% for result in unit_results:
							<th>${result.name}</th>
							% endfor
							</tr>

							% for test in sorted(unit_tests):
	<tr><td>${html.escape(test)}</td>\\
								% for result in unit_results:
	<td>${result.unit_results[test]}</td>\\
								% endfor
	</tr>
							% endfor
						</table>
						</div>
					% endif
				% endfor
		</body>

	</html>
	"""
	# Check that we have commits
	if len(db["commits"]) == 0 and verbose:
		print("No commits were found, cancelling...")
	else:
		# Create the html file
		if verbose:
			print("Generating the HTML")

		if title is None:
			report_names = [r.name for r in reports]
			title = "Performance report on the runs named '{run_name}'".format(run_name=report_names)

		templatedir = os.path.join(ezbench_dir, 'templates')

		lookup = TemplateLookup(directories=[templatedir])
		template = Template(html_template, lookup=lookup)
		try:
			html = template.render(ezbench_dir=ezbench_dir, title=title, db=db,
			                       output=output, output_unit=output_unit,
					       default_commit=list(db["commits"])[-1],
					       events=db["events"], embed=embed)
		except:
			html = exceptions.html_error_template().render().decode()

		with open(output, 'w') as f:
			f.write(html)
			if verbose:
				print("Output HTML generated at: {}".format(output))

def gen_report(log_folder, restrict_commits):
	report_name = os.path.basename(os.path.abspath(log_folder))

	try:
		sbench = SmartEzbench(ezbench_dir, report_name, readonly=True)
		report = sbench.report(restrict_to_commits = restrict_commits, silentMode=False)
	except RuntimeError:
		report = Report(log_folder, restrict_to_commits = restrict_commits)
		report.enhance_report(NoRepo(log_folder))

	return report

if __name__ == "__main__":
	import argparse

	# parse the options
	parser = argparse.ArgumentParser()
	parser.add_argument("--title", help="Set the title for the report")
	parser.add_argument("--unit", help="Set the output unit (Default: FPS)")
	parser.add_argument("--output", help="Report html file path")
	parser.add_argument("--commit_url", help="HTTP URL pattern, {commit} contains the SHA1")
	parser.add_argument("--quiet", help="Be quiet when generating the report", action="store_true")
	parser.add_argument("--reference", help="Compare the test results to this reference report")
	parser.add_argument("--reference_commit", help="Compare the test results to the specified commit of the reference report")
	parser.add_argument("--restrict_commits", help="Restrict commits to this list (space separated)")
	parser.add_argument("log_folder", nargs='+')
	args = parser.parse_args()

	# Set the output folder if not set
	if args.output is None:
		if len(args.log_folder) == 1:
			args.output = "{}/index.html".format(args.log_folder[0])
		else:
			print("Error: The output html file has to be specified when comparing multiple reports!")
			sys.exit(1)

	# Restrict the report to this list of commits
	restrict_commits = []
	if args.restrict_commits is not None:
		restrict_commits = args.restrict_commits.split(' ')

	reports = []
	for log_folder in set(args.log_folder):
		reports.append(gen_report(log_folder, restrict_commits))

	# Reference report
	reference = None
	if args.reference is not None:
		reference = gen_report(args.reference, [])

	reports_to_html(reports, args.output, args.unit, args.title,
			   args.commit_url, not args.quiet, reference, args.reference_commit)
