# Copyright (c) 2015, Intel Corporation
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

function profile_repo_deployment_version_dir() {
    echo "$PROFILE_DEPLOY_BASE_DIR/$version"
}

# MANDATORY: Default version of the function that deploys a previously-compiled
# version $version.
# Inputs:
#   - $version: the version to deploy
# Outputs:
#   - All the environment variables set to work as expected
function repo_deploy_version() {
    local dep_version_dir=$(profile_repo_deployment_version_dir)
    [ -d "$dep_version_dir" ] || return 72

    callIfDefined ezbench_deploy_pre_hook

    rm $PROFILE_DEPLOY_DIR 2> /dev/null
    ln -s $dep_version_dir $PROFILE_DEPLOY_DIR || return 72

    return 0
}

function auto_deploy_make_and_deploy() {
    # Return error codes:
    # 71: Compilation error
    # 72: Deployment error

    # Check that the deploy folder exists
    if [ ! -d "$DEPLOY_BASE_DIR" ]; then
        echo "ERROR: Please set DEPLOY_BASE_DIR ($DEPLOY_BASE_DIR) to an acceptable directory in user_parameters.sh."
        return 72
    fi

    # Call the user-defined pre-compile hook
    callIfDefined compile_pre_hook

    # Default to saying that there was a deployment error
    local compile_error=72

    # First, check if we already have compiled the wanted version
    repo_deploy_version
    local depl_version=$(profile_repo_deployed_version)

    # If we did not get the expected version, let's compile it
    if [[ "$depl_version" != "$version" ]]; then
        # Verify that the commit is not marked as broken
        local dep_version_dir=$(profile_repo_deployment_version_dir)
        local broken_commit_file="$dep_version_dir/ezbench_marked_broken"
        if [[ ! -e "$broken_commit_file" ]]; then
            echo "$(date +"%m-%d-%Y-%T"): Start compiling version $version"
            local compile_start=$(date +%s)

            profile_repo_compile_start $version
            local exit_code=$?
            [ $exit_code -ne 0 ] && return $exit_code

            repo_compile_version
            local compile_error=$?

            profile_repo_compile_stop

            if [[ $compile_error -eq 0 || $compile_error -eq 74 ]]; then
                # compute the compilation time
                local compile_end=$(date +%s)
                local build_time=$(($compile_end-$compile_start))
                echo "$(date +"%m-%d-%Y-%T"): Done compiling version $version (exit code=$compile_error). Build time = $build_time."

                # Update our build time estimator
                "$ezBenchDir/timing_DB/timing.py" -n build -k "$profile" -a $build_time
                local avgBuildTime=$(profile_repo_compilation_time)
                local avgBuildTime=$(bc <<< "0.75*$avgBuildTime + 0.25*$build_time")
                profile_repo_set_compilation_time $avgBuildTime

                # Call the user-defined post-compile hook
                callIfDefined deploy_pre_hook

                # Now deploy the version that we compiled
                repo_deploy_version

                # If compile_error == 0, this means that the deployment was successful
                # and did not require a reboot. Verify that the version is the
                # wanted one and, if not, say the deployment failed.
                if [[ $compile_error -eq 0 ]]; then
                    local depl_version=$(profile_repo_deployed_version)
                    if [[ "$depl_version" != "$version" ]]; then
                        local compile_error=72

                        # Mark the commit as broken, to avoid recompilations
                        touch "$broken_commit_file"
                    fi
                fi
            fi
        else
            echo "$(date +"%m-%d-%Y-%T"): Commit marked as broken. Remove '$broken_commit_file' if this is incorrect."
        fi
    else
        echo "$(date +"%m-%d-%Y-%T"): Found a cached version of the compilation, re-use it!"
    fi

    # Call the user-defined post-compile hook
    callIfDefined compile_post_hook

    return $compile_error
}

# TODO: Add a function that says how long it would take to test

makeAndDeployCmd="auto_deploy_make_and_deploy"
