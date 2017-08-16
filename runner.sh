#!/bin/bash

# Copyright (c) 2015-2017, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# The script is known to work with recent versions of bash.
# Authors:
# - Martin Peres <martin.peres@intel.com>
# - Chris Wilson <chris@chris-wilson.co.uk>

# Uncomment the following to track all the executed commands
#set -o xtrace

# Start by defining all the errors
typeset -A errors
errors[0]="No Error"
errors[1]="Unknown error"
errors[5]="An instance of runner.sh is already running"
errors[6]="The report is already locked"

# Commands parsing
errors[10]="Invalid command"
errors[11]="Parameter already set"
errors[12]="The profile does not exist"
errors[13]="The profile is not specified"
errors[14]="The report cannot be opened/created"
errors[15]="The report is missing"
errors[16]="Testing has not been started"
errors[17]="Test execution type unsupported"
errors[18]="Test execution type requires a valid result file"
errors[19]="Result already complete"

# OS
errors[30]="The shell does not support globstat"
errors[31]="Cannot create the log folder"
errors[32]="Cannot move to the repo directory"

# SCM
errors[50]="Invalid version ID"

# Environment
errors[60]="Failed to deploy the environment"

# Compilation and deployment
errors[70]="Compilation or deployment failed"
errors[71]="Compilation failed"
errors[72]="Deployment failed"
errors[73]="The deployed version does not match the wanted version"
errors[74]="A reboot is necessary"

# Running
errors[80]="Early exit requested"

# Test
errors[100]="The test does not exist"

function lock {
    # Inspired from http://www.kfirlavi.com/blog/2012/11/06/elegant-locking-of-bash-program/
    local lock_file="$1"
    local fd="$2"

    eval "exec $fd>$lock_file"

    flock -w 0.1 -x $fd
    return $?
}

function unlock {
    local fd="$1"
    flock -u $fd
    return $?
}

function use_report {
    local name=$1

    [ -n "$reportName" ] && return 11

    # Keep the stupid name for the rapl_check profile
    local folder="$ezBenchDir/logs/${name:-$(date +"%Y-%m-%d-%T")}"

    # Create the folder
    [ -d $folder ] || mkdir -p $folder || return 30
    echo "report,$name" >> $folder/runner.log
    exec > >(tee -a $folder/runner.log)
    exec 2>&1

    reportName=$name
    logsFolder=$folder

    # generate the early-exit filename
    abortFile="$logsFolder/requestExit"

    return 0
}

function use_profile {
    local prof=$1

    [ -n "$profile" ] && return 11

    # Check if the profile exists
    profileDir="$ezBenchDir/profiles.d/$prof"
    if [ ! -d "$profileDir" ]; then
        return 12
    fi

    # The profile name is valid, set some environment variables
    PROFILE_TMP_BUILD_DIR="$DEPLOY_BASE_DIR/$prof/tmp"
    PROFILE_DEPLOY_BASE_DIR="$DEPLOY_BASE_DIR/$prof/builds"
    PROFILE_DEPLOY_DIR="$DEPLOY_BASE_DIR/$prof/cur"

    # Default user options
    for conf in $profileDir/conf.d/**/*.conf; do
        [ "$conf" = "$profileDir/conf.d/**/*.conf" ] && continue
        source "$conf"
    done
    source "$profileDir/profile"

    profile=$prof

    return 0
}

function parse_tests_profiles  {
    [ $available_tests_parsed -eq 1 ] && return

    # Parse the list of tests
    for test_dir in ${testsDir:-$ezBenchDir/tests.d}; do
        for test_file in $test_dir/**/*.test; do
            unset test_name
            unset test_unit
            unset test_type
            unset test_invert
            unset test_exec_time

            source "$test_file" || continue

            # Sanity checks on the file
            [ -z "$test_name" ] && continue
            [ -z "$test_exec_time" ] && continue

            # Set the default unit to FPS
            [ -z "$test_unit" ] && test_unit="FPS"

            # Set the default type to bench
            [ -z "$test_type" ] && test_type="bench"

            # Set the default type to bench
            [ -z "$test_invert" ] && test_invert="0"

            # Set the default error handling capabiliy
            [ -z "$test_has_exit_code" ] && test_has_exit_code="0"

            for test in $test_name; do
                # TODO: Check that the run function exists

                idx=${#availTestNames[@]}
                availTestNames[$idx]=$test
                availTestUnits[$idx]=$test_unit
                availTestTypes[$idx]=$test_type
                availTestIsInvert[$idx]=$test_invert
                availTestExecTime[$idx]=$test_exec_time
                availTestHasExitCode[$idx]=$test_has_exit_code
            done
        done
    done
    unset test_name
    unset test_unit
    unset test_type
    unset test_invert
    unset test_exec_time

    available_tests_parsed=1
}

function done_testing {
    [ "$testing_ready" -eq 1 ] || return

    # Execute the user-defined post hook
    callIfDefined ezbench_post_hook

    # Delete the abort file
    if [ -n "$abortFile" ]; then
        rm $abortFile 2> /dev/null
    fi

    unlock 201
    unlock 200

    # Clear the traps before calling the finish hook
    trap - EXIT
    trap - INT # Needed for zsh

    testing_ready=0
}

function start_testing {
    [ "$testing_ready" -eq 0 ] || return
    [ -n "$profile" ] || return 13
    [ -n "$reportName" ] || return 15

    # Lock against concurent testing with the runner
    lock "$ezBenchDir/lock" 200 || return 5

    # Lock the report, as it allows knowing which report is being worked on
    if ! lock "$logsFolder/lock" 201; then
        unlock 200
        return 6
    fi

    # Execute the user-defined environment deployment hook
    callIfDefined ezbench_env_deploy_hook

    trap done_testing EXIT
    trap done_testing INT # Needed for zsh

    testing_ready=1

    # Write in the journal what current version is deployed
    write_to_journal deployed $(profile_repo_deployed_version)

    return 0
}

function compile_and_deploy {
    # Accessible variables
    # $version      [RO]: SHA1 id of the current version
    # $versionName  [RO]: Name of the version

    profile_repo_get_patch $version > "$logsFolder/$1.patch"

    # Select the version of interest
    local versionListLog="$logsFolder/commit_list"
    if [ -z "$(grep ^"$version" "$versionListLog" 2> /dev/null)" ]
    then
        local title=$(profile_repo_version_title "$version")
        echo "$version $title" >> "$versionListLog"
    fi

    # early exit if the deployed version is the wanted version
    deployed_version=$(profile_repo_deployed_version)
    are_same_versions "$version" "$deployed_version" && return 0

    local compile_logs=$logsFolder/${version}_compile_log

    # Compile the version and check for failure. If it failed, go to the next version.
    export REPO_COMPILE_AND_DEPLOY_VERSION=$version
    eval "$makeAndDeployCmd" >> "$compile_logs" 2>&1
    local exit_code=$?
    unset REPO_COMPILE_AND_DEPLOY_VERSION

    # Write the compilation status to the compile log.
    # The exit code 74 actually means everything is fine but we need to reboot
    if [ $exit_code -eq 74 ]
    then
        printf "Exiting with error code 0\n" >> "$compile_logs"
    else
        printf "Exiting with error code $exit_code\n" >> "$compile_logs"
    fi

    # Add to the log that we are now in the deployment phase
    write_to_journal deploy $version

    # Check for compilation errors
    if [ "$exit_code" -ne '0' ]; then
        if [[ $exit_code -lt 71 || $exit_code -gt 74 ]]; then
            return 70
        else
            return $exit_code
        fi
    fi

    # Check that the deployed image is the right one
    are_same_versions "$version" "$(profile_repo_deployed_version)" || return 73

    # Write in the journal that the version is deployed
    write_to_journal deployed $version

    return 0
}

function find_test {
    # Accessible variables
    # $availTestNames    [RO]: Array containing the names of available tests
    # $availTestUnits    [RO]: Array containing the unit of available tests
    # $availTestTypes    [RO]: Array containing the result type of available tests
    # $availTestIsInvert [RO]: Array containing the execution time of available tests
    # $availTestExecTime [RO]: Array containing the execution time of available tests

    local test=$1
    local basetest=$(echo "$test" | cut -d [ -f 1)
    local subtests=$(echo $test | cut -s -d '[' -f 2- | rev | cut -d ']' -f 2- | rev)

    # Try to find the test in the list of available tests
    for (( a=0; a<${#availTestNames[@]}; a++ )); do
        if [[ ${availTestNames[$a]} == $basetest ]]; then
            # We do not accept running subtests on non-complete matches of names
            [[ $basetest != ${availTestNames[$a]} && -n $subtests ]] && continue

            testName="${availTestNames[$a]}"
            testSubTests="$subtests"
            testUnit="${availTestUnits[$a]}"
            testType="${availTestTypes[$a]}"
            testInvert="${availTestIsInvert[$a]}"
            testExecTime="${availTestExecTime[$a]}"
            testHasExitCode="${availTestHasExitCode[$a]}"

            return 0
        fi
    done

    return 1
}

function execute_test {
    local testExecutionType=$1
    local versionName=$2
    local testName=$3
    local verboseOutput=$4
    local run_log_file_name=$5

    [ -n "$versionName" ] || return 10
    [ -n "$testName" ] || return 10
    [ -n "$verboseOutput" ] || return 10
    [ -n "$profile" ] || return 13
    [ -n "$reportName" ] || return 15
    [ "$testing_ready" -eq 1 ] || return 16

    # Verify that, if specified, the run_log_file_name exits
    [[ -n "$run_log_file_name" && ! -f "$logsFolder/$run_log_file_name" ]] && return 18

    # Find the test
    parse_tests_profiles
    find_test "$testName" || return 100

    # Get the full version name
    local version
    version=$(profile_repo_version_from_human $versionName)
    if [ $? != 0 ]; then
	return 50
    fi

    # verify that the type of execution is available
    local execFuncName=${testName}_${testExecutionType}
    if ! function_available "$execFuncName"; then
        # If the type was resume, mark the test as tested because there is
        # nothing else we can do for this run
        if [ "$testExecutionType" == "resume" ]; then
            write_to_journal tested "$version" "${testName}" "$run_log_file_name"
        fi
        return 17
    fi

    # compile and deploy the version
    compile_and_deploy $version || return $?

    # Exit if asked to
    [ -e "$abortFile" ] && return 80

    # Generate the logs file names
    local reportFileName="${version}_${testType}_${testName}"
    local reportFile="$logsFolder/$reportFileName"

    # Only generate the run_log_file_name if it is unspecified
    if [ -z "$run_log_file_name" ]; then
        # Find the first run id available
        if [ -f "$reportFile" ] || [ ${testType} == "unified" ]; then
            # The logs file exist, look for the number of runs
            local run=0
            while [ -f "${reportFile}#${run}" ]
            do
                local run=$((run+1))
            done
        else
            if [ -z "${testInvert}" ]; then
                direction="more is better"
            else
                direction="less is better"
            fi
            echo "# ${testUnit} ($direction) of '${testName}' using version ${version}" > "$reportFile"
            local run=0
        fi
        local run_log_file_name="${reportFileName}#$run"
    fi

    # compute the different hook names
    local preHookFuncName=${testName}_run_pre_hook
    local postHookFuncName=${testName}_run_post_hook

    local run_log_file="$logsFolder/$run_log_file_name"
    IFS='|' read -a run_sub_tests <<< "$testSubTests"

    echo "$run_log_file_name"

    callIfDefined "$preHookFuncName"
    callIfDefined benchmark_run_pre_hook

    # This function will return multiple fps readings
    write_to_journal test "$version" "${testName}" "$run_log_file_name"
    if [ "$verboseOutput" -eq 1 ]; then
        "$execFuncName" > >(tee "$run_log_file")
    else
        "$execFuncName" > "$run_log_file" 2> /dev/null
    fi
    local exit_code=$?
    if [ "$testHasExitCode" -eq 1 ]; then
        # If the testing requested a reboot etc, we don't really want to mark it
        # as completely tested. The exit codes signaling a completed test are
        # 0 (no error) and 19 (already complete).
        if [[ ! "$exit_code" -eq 0 && ! "$exit_code" -eq 19 ]]; then
            callIfDefined benchmark_run_post_hook
            callIfDefined "$postHookFuncName"
            return "$exit_code"
        fi
    fi

    write_to_journal tested "$version" "${testName}" "$run_log_file_name"

    callIfDefined benchmark_run_post_hook
    callIfDefined "$postHookFuncName"

    if [ ${testType} != "unified" ]; then
        if [ -s "$run_log_file" ]; then
            if [ ${testType} == "bench" ]; then
                # Add the reported values before adding the result to the average values for
                # the run.
                run_avg=$(awk '{sum=sum+$1} END {print sum/NR}' $run_log_file)
            elif [ ${testType} == "unit" ]; then
                run_avg=$(head -n 1 $run_log_file)
            elif [ ${testType} == "imgval" ]; then
                run_avg=0.0
            fi
            echo "$run_avg" >> "$reportFile"

        else
            echo "0" >> "$run_log_file"
            echo "0" >> "$reportFile"
        fi
    fi

    # If the test returns exit codes, then use it
    if [ "$testHasExitCode" -eq 1 ]; then
        return $exit_code
    else
        return 0
    fi
}

# set the default run_bench function which can be overriden by the profiles:
# Bash variables: $run_log_file : filename of the log report of the current run
#                 $testName : Name of the test
#                 $testSubTests : List of subtests
# Arguments: $1 : timeout (set to 0 for infinite wait)
#            $2+: command line executing the test AND NOTHING ELSE!
function run_bench {
    timeout=$1
    shift
    cmd="LIBGL_DEBUG=verbose vblank_mode=0 stdbuf -oL timeout $timeout"
    bench_binary=$(echo "$1" | rev | cut -d '/' -f 1 | rev)

    env_dump_path="$ezBenchDir/utils/env_dump"
    env_dump_lib="$env_dump_path/env_dump.so"
    env_dump_launch="$env_dump_path/env_dump.sh"
    env_dump_extend="$env_dump_path/env_dump_extend.sh"

    if [ -f "$env_dump_lib" ]; then
        run_log_file_env_dump="$run_log_file"
        cmd="$cmd $env_dump_launch "$run_log_file" $@"
    else
        cmd="$cmd $@"
    fi

    run_log_file_stdout="$run_log_file.stdout"
    run_log_file_stderr="$run_log_file.stderr"

    callIfDefined run_bench_pre_hook
    local time_before=$(date +%s.%N)

    eval $cmd > "$run_log_file_stdout" 2> "$run_log_file_stderr"
    local exit_code=$?

    if [ -f "$env_dump_lib" ]; then
        $env_dump_extend "$SHA1_DB" "$run_log_file.env_dump"
    fi

    local time_after=$(date +%s.%N)
    test_exec_time=$(echo "$time_after - $time_before" | bc -l)
    callIfDefined run_bench_post_hook

    # If the test does not have subtests, then store the execution time if the run was successful.
    # Success exit codes are 0 (no error) and 19 (already complete).
    if [[ -z "$testSubTests" && "$testExecutionType" != "resume" ]] && [[ "$exit_code" -eq "0" || "$exit_code" -eq "19" ]]; then
        "$ezBenchDir/timing_DB/timing.py" -n test -k "$testName" -a $test_exec_time
    fi

    # delete the log files if they are empty
    if [ ! -s "$run_log_file_stdout" ] ; then
        rm "$run_log_file_stdout"
    else
        cat "$run_log_file_stdout"
    fi
    if [ ! -s "$run_log_file_stderr" ] ; then
        rm "$run_log_file_stderr"
    else
        cat "$run_log_file_stderr" >&2
    fi

    return $exit_code
}

function function_available() {
    if [ "$(type -t "$1")" == 'function' ]; then
        return 0
    else
        return 1
    fi
}

function callIfDefined() {
    if [ "$(type -t "$1")" == 'function' ]; then
        local funcName=$1
        shift
        $funcName "$@"
        return 0
    else
        return 1
    fi
}

function write_to_journal {
    local journal="$logsFolder/journal"
    local operation=$1
    local key=$2
    shift 2

    [ -d "$logsFolder" ] || return

    local time=$(date +"%s.%6N")

    # if the operation is "deployed", check how long it took to deploy
    if [[ "$operation" == "deployed" ]]; then
        local deploy_time=$(tail -n 1 "$logsFolder/journal" 2> /dev/null | awk -F ',' "{    \
            if (\$2 == \"deploy\" && \$3 == \"$key\") {                          \
                print $time - \$1                                              \
            }                                                                  \
        }")

        if [ -n "$deploy_time" ]; then
            "$ezBenchDir/timing_DB/timing.py" -n deploy -k "$profile" -a "$deploy_time"
        fi
    fi

    echo -n "$time,$operation,$key" >> "$journal" 2> /dev/null

    while [ "${1+defined}" ]; do
        echo -n ",$1" >> "$journal" 2> /dev/null
        shift
    done
    echo "" >> "$journal" 2> /dev/null

    sync
}

function show_help {
    echo "This tool is meant to be run by machines, but if you really want to deal with this"
        echo "manually, here is the list of commands that can be used:"
        echo ""
        echo "  - side-effect-free commands:"
        echo "    - conf_script:"
        echo "          return the list of configuration scripts set"
        echo ""
        echo "    - echo:"
        echo "          does nothing, useful as a fence mechanism"
        echo ""
        echo "    - help:"
        echo "          displays this help message"
        echo ""
        echo "    - list_cached_versions:"
        echo "          list all the compiled versions for the profile"
        echo ""
        echo "    - list_tests:"
        echo "          list the set of available tests"
        echo ""
        echo "    - profile:"
        echo "          return the current profile"
        echo ""
        echo "    - repo:"
        echo "          return information about the profile's repository"
        echo ""
        echo "    - report:"
        echo "          return the current profile"
        echo ""
        echo "    - version:"
        echo "          return the implemented version of the protocol"
        echo ""
        echo ""
        echo "  - state-changing commands:"
        echo "    - conf_script,<path>:"
        echo "          set the profile for the current session"
        echo ""
        echo "    - done:"
        echo "          tear down the test environment and exit"
        echo ""
        echo "    - profile,<profile>:"
        echo "          set the profile for the current session"
        echo ""
        echo "    - reboot:"
        echo "          reboot the machine"
        echo ""
        echo "    - report,<report_name>:"
        echo "          set the report for the current session"
        echo ""
        echo "    - resume,<version>,<test>,<result_file>,<verbose>:"
        echo "          Run a test, verbose=1 for reading the output"
        echo ""
        echo "    - run,<version>,<test>,<verbose>:"
        echo "          Run a test, verbose=1 for reading the output"
        echo ""
        echo "    - start_testing:"
        echo "          start the test environment"
        echo ""
        echo "That's all, folks!"
}

shopt -s globstar || {
    echo "ERROR: ezbench requires bash 4.0+ or zsh with globstat support."
    exit 30
}

# Printf complains about floating point numbers having , as a delimiter otherwise
LC_NUMERIC="C"

# Get the current folder
ezBenchDir=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# initial cleanup
mkdir "$ezBenchDir/logs" 2> /dev/null

# Read the user parameters
source "$ezBenchDir/user_parameters.sh"

# Initialise the default state
typeset -A availTestNames
typeset -A availTestUnits
typeset -A availTestTypes
typeset -A availTestIsInvert
typeset -A availTestExecTime
typeset -A availTestHasExitCode
typeset -A confs
protocol_version=1
reportName=""
profile=""
available_tests_parsed=0
testing_ready=0
dry_run=0

# Now start the main loop
while read line
do
    start_time=$(date +%s.%N)
    break=0
    error=1

    echo "-->,$line"
    cmd=$(echo $line | cut -d ',' -f 1)

    if [ "$cmd" == "help" ]; then
        show_help
        error=0

    elif [ "$cmd" == "report" ]; then
        arg=$(echo "$line" | cut -d ',' -s -f 2)
        if [ -n "$arg" ]; then
            use_report "$arg"
            error=$?
        else
            echo "$reportName"
            error=0
        fi

        reportName=$arg

    elif [ "$cmd" == "profile" ]; then
        arg=$(echo "$line" | cut -d ',' -s -f 2)
        if [ -n "$arg" ]; then
            use_profile "$arg"
            error=$?
        else
            echo "$profile"
            error=0
        fi

    elif [ "$cmd" == "repo" ]; then
        if [ -n "$profile" ]; then
            echo "type=$(profile_repo_type)"
            echo "path=$repoDir"
            echo "head=$(profile_repo_version)"
            echo "deployed_version=$(profile_repo_deployed_version)"
            error=$?
        else
            error=13
        fi

    elif [ "$cmd" == "conf_script" ]; then
        arg=$(echo "$line" | cut -d ',' -s -f 2)
        if [ -n "$arg" ]; then
            source "$arg"
            idx=${#confs[@]}
            confs[$idx]="$arg"
            error=0
        else
            for (( c=0; c<${#confs[@]}; c++ )); do
                echo "${confs[$c]}"
            done
            error=0
        fi

    elif [ "$cmd" == "list_tests" ]; then
        parse_tests_profiles

        for (( a=0; a<${#availTestNames[@]}; a++ )); do
            echo "${availTestNames[$a]},${availTestTypes[$a]},${availTestUnits[$a]},${availTestIsInvert[$a]},${availTestExecTime[$a]}"
        done
        error=0

    elif [ "$cmd" == "list_cached_versions" ]; then
        if [ -n "$profile" ]; then
            for v in $(profile_get_built_versions); do
                echo $v
            done
            error=0
        else
            error=13
        fi

    elif [ "$cmd" == "start_testing" ]; then
        start_testing
        error=$?

    elif [[ "$cmd" == "done" || "$cmd" == "reboot" ]]; then
        break=1
        error=0

    elif [ "$cmd" == "resume" ]; then
        version=$(echo "$line" | cut -d ',' -s -f 2)
        test=$(echo "$line" | cut -d ',' -s -f 3)
        result_file=$(echo "$line" | cut -d ',' -s -f 4)
        verbose=$(echo "$line" | cut -d ',' -s -f 5)

        execute_test resume "$version" "$test" "$verbose" "$result_file"
        error=$?

    elif [ "$cmd" == "run" ]; then
        version=$(echo "$line" | cut -d ',' -s -f 2)
        test=$(echo "$line" | cut -d ',' -s -f 3)
        verbose=$(echo "$line" | cut -d ',' -s -f 4)

        execute_test run "$version" "$test" "$verbose"
        error=$?

    elif [ "$cmd" == "echo" ]; then
        error=0

    elif [ "$cmd" == "version" ]; then
        echo "$protocol_version"
        error=0

    else
        error=10
    fi

    t=$(echo "($(date +%s.%N) - $start_time) * 1000 / 1" | bc)
    echo "<--,$error,${errors[$error]},$t ms"

    # Exit, if requested
    [ "$break" == 1 ] && break
done < "${1:-/dev/stdin}"

# Make sure we tear-down everything we started
done_testing

# If the last command received was reboot, do it!
if [ "$cmd" == "reboot" ]; then
    # Default to 120 seconds if not set in user_parameters.sh
    if [ -z "$WATCHDOG_TIMEOUT_SYNC" ]; then
	WATCHDOG_TIMEOUT_SYNC=120
    fi
    if [ -z "$WATCHDOG_TIMEOUT_REBOOT" ]; then
	WATCHDOG_TIMEOUT_REBOOT=120
    fi

    # Sync disks. Use owatch if available to time out with a power off
    # if filesystems are jammed.
    owcmdline=
    owatchbin="$ezBenchDir/utils/owatch/owatch"
    if [ -x "$owatchbin" ]; then
	owcmdline="$owatchbin $WATCHDOG_TIMEOUT_SYNC"
    fi
    $owcmdline sync

    # Use reboot_wd to time out with a power off.
    rwdbin="$ezBenchDir/utils/reboot_wd/reboot_wd"
    if [ -x "$rwdbin" ]; then
	sudo "$rwdbin" "$WATCHDOG_TIMEOUT_REBOOT"
    fi
    sudo reboot
fi

exit 0
