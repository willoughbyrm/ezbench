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

import struct
import socket
import sys
import os

import pygit2

unit_tests_dir = os.path.dirname(os.path.abspath(__file__))
ezbench_dir = os.path.dirname(unit_tests_dir)
timings_db_dir = os.path.join(ezbench_dir, 'timing_DB')

sys.path.insert(0, os.path.join(ezbench_dir, 'python-modules'))
sys.path.insert(0, os.path.join(ezbench_dir, 'protocols'))
sys.path.insert(0, timings_db_dir)
sys.path.insert(0, ezbench_dir)

tmp_folder = ezbench_dir + "/unit_tests/tmp"

def send_msg(client, msg):
    buf = msg.SerializeToString()
    if len(buf) == 0:
        print("WARNING: Trying to send 0 bytes!")
        traceback.print_stack()
    pack1 = struct.pack('>I', len(buf))
    client.sendall(pack1 + buf)

def recv_msg(client, msg):
    msg_len_buf = client.recv(4, socket.MSG_WAITALL)
    if len(msg_len_buf) != 4:
        return None

    msg_len = struct.unpack('>I', msg_len_buf)[0]
    buf = bytes()
    while len(buf) < msg_len:
        buf += client.recv(msg_len - len(buf), socket.MSG_WAITALL)

    if len(buf) == msg_len:
        msg.ParseFromString(buf)
        return msg
    else:
        print("Received less than expected (got {} instead of {})".format(len(buf), msg_len), buf)
        return None

class GitRepoFactory:
    def __init__(self, path, bare=False):
        # Create a test dut-repo and a couple of commits
        self.repo_path = path
        self.is_bare = bare
        self.src_repo = pygit2.init_repository(path, self.is_bare)

    def create_commit(self, title, files=[], parents=[], branch_name="master"):
        # Create the data
        tb = self.src_repo.TreeBuilder()
        for filepath, data in files:
            data_oid = self.src_repo.create_blob(data)
            tb.insert(filepath, data_oid, pygit2.GIT_FILEMODE_BLOB)

        # Make the commit
        author = pygit2.Signature("EzBench Test Author", "test@author.com")
        reference="refs/heads/" + branch_name
        commit_id = self.src_repo.create_commit(reference, author, author, title,
                                       tb.write(), parents)
        commit = self.src_repo.revparse_single(str(commit_id))

        # Update the master branch
        try:
            branch = self.src_repo.lookup_branch(branch_name)
        except:
            branch = self.src_repo.create_branch(branch_name, commit, False)
        branch.set_target(str(commit_id))

        # Now update the working copy to the current branch
        if not self.is_bare:
            self.src_repo.checkout(reference)

        return str(commit_id)

    def create_tag(self, name, commit, msg="Nothing to say here"):
        author = pygit2.Signature("EzBench Test Author", "test@author.com")
        return self.src_repo.create_tag(name, commit, pygit2.GIT_OBJ_COMMIT,
                                        author, msg)

