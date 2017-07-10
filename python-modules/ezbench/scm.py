"""
Copyright (c) 2016, Intel Corporation

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

from collections import namedtuple, deque
import pygit2
import copy
import time
import sys

class ResultsDAG:
    __slots__ = ['scm', '_parents', '_children', '_results', '_scores',
                 '_scores_head', '_scores_sorted', '_edge_graphs',
                 '_edge_results', '_nodes_cache']

    def __init__(self, scm):
        """
        Construct a ResultsDAG object which is backed by the $scm Source control
        management.

        Args:
            scm: A SCM object
        """

        self.scm = scm
        self._parents = dict()
        self._children = dict()
        self._results = dict()
        self._scores = dict()
        self._scores_head = None
        self._scores_sorted = None
        self._edge_graphs = dict()
        self._edge_results = dict()

        self._nodes_cache = set()

    def add_edge(self, parent_id, child_id, graph = None, results = None):
        """
        Add a directed edge from $parent_id to $child_id. It is possible to
        add a graph as a parameter that will hold another ResultsDAG which
        would represent how $parent_id and $child_id are connected in another
        DAG. This is useful when the current DAG is an overlay.

        Args:
            parent_id: a node id
            child_id: a node id
            graph: another ResultsDAG indicating how the edge connects the two nodes
            results: a set of results
        """

        self._nodes_cache = None

        if parent_id not in self._children:
            self._children[parent_id] = set()
        self._children[parent_id].add(child_id)

        if child_id not in self._parents:
            self._parents[child_id] = set()
        self._parents[child_id].add(parent_id)

        if graph is not None:
            k = (parent_id, child_id)
            if k not in self._edge_graphs:
                self._edge_graphs[k] = deque()
            self._edge_graphs[k].append(graph)

        if results is not None:
            k = (parent_id, child_id)
            if k not in self._edge_results:
                self._edge_results[k] = set()
            self._edge_results[k] |= results

    def set_results(self, node_id, results):
        """
        Add a set of results to $node_id. useful in conjunction with the
        find_closest_nodes_with_results method.

        Args:
            node_id: a node id
            results: a set of results to associate to $node_id
        """
        self._results[node_id] = results

    def parents(self, node_id):
        """
        Return the list of nodes which have an edge going to $node_id.

        Args:
            node_id: a node id
        """

        return self._parents.get(node_id, [])

    def children(self, node_id):
        """
        Return the list of nodes which have an edge going from $node_id.

        Args:
            node_id: a node id
        """

        return self._children.get(node_id, [])

    def results(self, node_id):
        """
        Return the results associated to $node_id.

        Args:
            node_id: a node id to get the results from
        """

        return self._results.get(node_id, set())

    def score(self, node_id):
        """
        Return the score associated to $node_id, generated during the last time
        bisecting_scores got called.

        Args:
            node_id: a node id to get the score of
        """

        return self._scores.get(node_id, None)

    def edge_graphs(self, parent_id, child_id):
        """
        Return a list of graphs associated to the edge $parent_id->$child_id.

        Args:
            parent_id: the origin node of the edge
            child_id: the destination node of the edge
        """

        return self._edge_graphs.get((parent_id, child_id), [])

    def edge_results(self, parent_id, child_id):
        """
        Return a list of graphs associated to the edge $parent_id->$child_id.

        Args:
            parent_id: the origin node of the edge
            child_id: the destination node of the edge
        """

        return self._edge_results.get((parent_id, child_id), set())

    def nodes(self):
        """
        Return the list of nodes found in the DAG.
        """
        if self._nodes_cache is None:
            self._nodes_cache = set(self._children.keys()) | set(self._parents.keys())

        return self._nodes_cache

    def __len__(self):
        """
        Return the length of the list of nodes found in the DAG.
        """

        return len(self.nodes())

    def __to_dot_format_node_name__(self, node_id):
        s = node_id
        score = self.score(node_id)
        if score is not None:
            s += " ({})".format(score)
        return s

    def to_dot_format(self, output_file=None):
        """
        Format the DAG in graphviz's dot format and return the generated string.
        If $output_file is specified, the result will be written to the
        specified file.

        Args:
            output_file: the output file to write the dot graph to
        """

        out = "digraph {\n"
        for p in self._children:
            for c in self.children(p):
                commits_count = len(list(self.scm.version_range_list(p, c)))
                msg= "    \"{}\" -> \"{}\"[label=\"{} commits, {} tests\"];\n"
                out += msg.format(self.__to_dot_format_node_name__(p),
                                  self.__to_dot_format_node_name__(c),
                                  commits_count, len(self.edge_results(p, c)))
        out += "}"

        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write(out)

        return out

    def bisecting_scores(self, head):
        """
        Return a list of scores for all the nodes of the graph to determine
        which ones would be the best candidates for bisecting between $head and
        all the points reachable from $head.

        Args:
            head: starting point of the search
        """

        # Check if the cache is valid
        if head != self._scores_head:
            # Cache miss, reset the _scores cache
            self._scores = dict()
            self._scores_head = head

            ancestors = dict()
            ancestors_len = dict()
            next_nodes = [head]
            while len(next_nodes) > 0:
                node = next_nodes.pop(0)
                if node in ancestors_len:
                    continue

                # Check if all the parents' ancestors are already available
                asAllParents = True
                for p in self.parents(node):
                    if p not in ancestors:
                        # We do not have the ancestors for the current parent,
                        # push back the node in the next_nodes list so as we
                        # have a chance to compute the parent at one point
                        next_nodes.append(p)
                        asAllParents = False

                        # Explore one branch at a time, to avoid work duplication
                        # and memory explosion
                        break

                if not asAllParents:
                    continue

                cur_ancestors = set()
                for p in self.parents(node):
                    # count how many children of $p are still in need of $p's score
                    children_without_results = 0
                    for c in self.children(p):
                        if c not in ancestors_len:
                            children_without_results += 1
                        if children_without_results >= 2:
                            break

                    # If only one child needs the result, it must be $node and
                    # we can safely destroy the set (pop) to free some space
                    if children_without_results == 1:
                        cur_ancestors |= ancestors.pop(p)
                    else:
                        cur_ancestors |= ancestors[p]

                cur_ancestors.add(node)
                ancestors[node] = cur_ancestors
                ancestors_len[node] = len(cur_ancestors)

                for child in self.children(node):
                    next_nodes.insert(0, child)

            # Create a list of tuples (commit, score) with the updated the score
            # so as to find the middle commit:
            # score = min(score, len(scores) - score)
            ret = []
            for c in ancestors_len:
                score = min(ancestors_len[c], len(ancestors_len) - ancestors_len[c])
                ret.append((c, score))
                self._scores[c] = score
            self._scores_sorted = sorted(ret, key=lambda e: e[1], reverse=True)

        return self._scores_sorted

    def add_DAG(self, g, head):
        """
        Add $head from the DAG $g and all its parents

        Args:
            head: head of the part of the graph to add to self
        """
        next_nodes = [head]

        while len(next_nodes) > 0:
            node = next_nodes.pop()
            for p in g._parents.get(node, []):
                if p not in self._children:
                    self.add_edge(p, node)
                    next_nodes.append(p)

    def find_closest_nodes_with_results(self, start_id):
        """
        Go through all the different paths going from start_id and stop each
        path when we have found all the results found on start_id or if this is
        the end of the current branch.
        To avoid a combinatorial explosion, we need to merge pathes together. To
        do so, we create yet another graph which is indexed by both a node_id and
        the set of results.

        To further reduce the amount of work, we merge (NodeA, SetA) and
        (NodeA, SetB) by only creating (NodeA, SetB - SetA)
        When we are done, we return the list of closest results and a DAG
        containing the list of commits between the two closest results

        Args:
            start_id: the node id to start looking from
        """
        start_results = self.results(start_id)
        if len(start_results) == 0:
            return []

        ResultsNode = namedtuple('ResultsNode', ['node_id', 'parents', 'results', 'results_count'])
        start_node = ResultsNode(start_id, None, start_results, len(start_results))

        result_nodes = dict()
        result_nodes[start_node.node_id] = [start_node]
        next_nodes = [start_node]

        found = list()
        while True:
            try:
                n = next_nodes.pop(0)
            except IndexError:
                break

            if len(n.results) == 0:
                continue

            for c in self._children.get(n.node_id, []):
                force_add = False

                # prepare for adding the new node in the result nodes
                if c not in result_nodes:
                    result_nodes[c] = []

                # Compute the set of results we need
                if c in self._results:
                    needed_results = n.results - self._results[c]
                    force_add = True
                    needed_results_local = True
                else:
                    needed_results = n.results
                    needed_results_local = False

                # Try to coalesce needed results in the existing sets of results
                # Add n as a parent if there is any intersection and remove this
                # intersection set from needed_results
                len_needed_results = len(needed_results)
                for sub_node in result_nodes[c]:
                    if needed_results_local:
                        needed_results -= sub_node.results
                    else:
                        needed_results = n.results - sub_node.results
                        needed_results_local = True

                    cur_len = len(needed_results)
                    if cur_len != len_needed_results:
                        sub_node.parents.append(n.node_id)
                        len_needed_results = cur_len

                # If there are any results left, create a new node with the
                # remaining set of results and add it to the list of next_nodes
                # at the right location (ordered from the biggest results_count
                # to the lowest one
                if len_needed_results > 0 or force_add:
                    child = ResultsNode(c, [n.node_id], needed_results, len_needed_results)
                    result_nodes[c].append(child)

                    # Look for the first element with a lower or equal results_count than
                    # the current one
                    i = 0
                    for nn in iter(next_nodes):
                          if nn.results_count <= len_needed_results:
                              break
                          i += 1
                    next_nodes.insert(i, child)

                    # Now, check what we actually found
                    found_results = (n.results - needed_results) & self.results(c)
                    if found_results != set():
                        found.append((c, found_results))

        return found

class GitRepo:
    def __init__(self, repo_path):
        """
        Create a git Repository object

        Args:
            repo_path: A valid path to the git repo
        """

        self.repo_path = repo_path
        self.repo = pygit2.Repository(repo_path)
        self._cached_merged_bases = dict()

    def full_version_name(self, version):
        """
        Convert from a short-hand to a full version.

        WARNING: This may fail, so try not to rely on this feature.

        Args:
            version: a valid version name to be converted to a full version
        """

        try:
            rev = self.repo.revparse_single(version)
            if type(rev) == pygit2.Tag:
                return str(rev.target.hex)
            else:
                return str(rev.oid)
        except:
            return version

    def short_version_name(self, version):
        """
        Convert from a full version name to a more readable one.

        Args:
            version: a valid version name to be converted to a short version
        """

        try:
            full_name = self.full_version_name(version)

            digits = 7
            while str(self.repo.revparse_single(full_name[0:digits]).oid) != full_name:
                digits += 1

            return full_name[0:digits]
        except:
            return version

    def list_versions(self, head="HEAD", restrict_to_commits=[]):
        """
        List all the versions accessible from $head, limited to versions found
        in $restrict_to_commits and output them from $head down.

        Args:
            head: a valid version name that will be used as the top of the tree
            restrict_to_commits: a list of acceptable versions that will be used to filter the output
        """

        head = self.repo.revparse_single(head).oid
        commits = None

        # Resolve all the shorthand IDs to the full one
        if len(restrict_to_commits) > 0:
            commits = set()
            for c in restrict_to_commits:
                commits.add(self.full_version_name(c))

        walker = self.repo.walk(head, pygit2.GIT_SORT_TOPOLOGICAL | pygit2.GIT_SORT_TIME)
        for commit in walker:
            if commits is not None:
                if str(commit.oid) in commits:
                    yield str(commit.oid)
                    commits.remove(str(commit.oid))
                    if len(commits) == 0:
                        return
            else:
                yield str(commit.oid)

    def walk(self, heads, ignores):
        """
        Return a DAG containing all the versions accessible from all versions
        found in $heads but not from all the versions found in $ignores.

        The difference with subDAG is that it will not include the nodes found
        in $ignore.

        Args:
            heads: list of versions to start from
            ignores: list of versions which should be ignored, along with
                     heir ancestors
        """

        g = ResultsDAG(self)

        # Create a walker that will visit all the commits from $h but ignore
        # all the other versions
        first = heads.pop()
        walker = self.repo.walk(first, pygit2.GIT_SORT_TOPOLOGICAL | pygit2.GIT_SORT_REVERSE)
        for head in heads:
            walker.push(head)
        heads.append(first)

        for ignore in ignores:
            walker.hide(ignore)

        # Now walk the commits and add them to the graph
        visited = set()
        for commit in walker:
            visited.add(commit.oid)
            for parent in commit.parents:
                if parent.oid in visited:
                    g.add_edge(str(parent.oid), str(commit.oid))

        return g

    def merge_base(self, versions):
        """
        Returns the merge base of multiple versions

        Args:
            versions: the versions you want to have the merge base for
        """

        k = (frozenset(versions))
        cached = self._cached_merged_bases.get(k, None)
        if cached == None:
            # Find the merge base of all the versions
            s = set(versions)
            while len(s) > 1:
                v1 = self.repo.revparse_single(s.pop()).oid
                v2 = self.repo.revparse_single(s.pop()).oid

                v3 = self.repo.merge_base(v1, v2)
                if v3 is not None:
                    s.add(str(v3))

            cached = s.pop()
            self._cached_merged_bases[k] = cached

        return cached

    def subDAG(self, versions):
        """
        Returns a DAG which contains the minimal subset of commits which compose
        all the specified versions.

        Args:
            versions: the versions you want to have the minimal history for
        """

        g = ResultsDAG(self)
        if len(versions) == 0:
            return g

        # Create a walker that will visit all the commits from $h but ignore
        # all the other versions
        first = versions.pop()
        walker = self.repo.walk(first)
        for v in versions:
            walker.push(v)
        versions.append(first)

        merge_base = self.merge_base(versions)
        walker.hide(merge_base)

        # Now walk the commits and add them to the graph
        for commit in walker:
            for parent in commit.parents:
                g.add_edge(str(parent.oid), str(commit.oid))

        return g

    def version_range_list(self, before, after, ignore = set()):
        """
        Returns the list of commits between $before and $after, not including $before.

        Args:
            before: a valid version name
            after: a valid version name
            ignore: set of version to ignore, preventing all paths to go through it
        """

        head = self.repo.revparse_single(after).oid
        end = self.repo.revparse_single(before).oid

        walker = self.repo.walk(head, pygit2.GIT_SORT_TOPOLOGICAL | pygit2.GIT_SORT_REVERSE)
        walker.hide(end)
        for i in ignore:
            if self.repo.revparse_single(i).oid != head:
                walker.hide(i)

        parents = set([end])
        for commit in walker:
            if str(commit.oid) in ignore:
                continue

            found = False
            for p in commit.parents:
                if p.oid in parents:
                    found = True
                    parents.add(commit.oid)
                    break
            if found:
                yield str(commit.oid)

    def version_description(self, version):
        """
        Return a one-line description of the version.

        Args:
            version: a valid version name
        """

        return self.repo.revparse_single(version).message.splitlines()[0]

    def version_parents(self, version):
        """
        Return the list of parents for the version $version.

        Args:
            version: a valid version name
        """

        ret = []
        for p in self.repo.revparse_single(version).parents:
            ret.append(str(p.oid))
        return ret

class NoRepo:
    def __init__(self, repo_path):
        """
        Create a git Repository object

        Args:
            repo_path: A valid path to the git repo
        """

        self.repo_path = repo_path
        self._desc = dict()
        self._version_graph = ResultsDAG(self)
        self._first_version = None
        self._last_version = None

        try:
            with open(repo_path + "/commit_list", 'rt') as f:
                prev = None
                for line in f:
                    line = line.strip()
                    fields = line.split(' ')

                    version = fields[0]
                    name = " ".join(fields[1:])

                    self._desc[version] = name

                    if self._first_version is None:
                        self._first_version = version
                    if prev is not None:
                        self._version_graph.add_edge(prev, version)
                    self._last_version = version
                    prev = version
        except IOError:
            pass

    def full_version_name(self, version):
        """
        Convert from a short-hand to a full version.

        WARNING: This may fail, so try not to rely on this feature.

        Args:
            version: a valid version name to be converted to a full version
        """

        return version

    def short_version_name(self, version):
        """
        Convert from a full version name to a more readable one.

        Args:
            version: a valid version name to be converted to a short version
        """

        return version

    def walk(self, heads, ignores):
        """
        Return a DAG containing all the versions accessible from all versions
        found in $heads but not from all the versions found in $ignores.

        The difference with subDAG is that it will not include the nodes found
        in $ignore.

        Args:
            heads: list of versions to start from
            ignores: list of versions which should be ignored, along with
                     heir ancestors
        """

        # Pretend that every version is next to each others, as we cannot bisect
        # anything anyway. To do so, just return an empty ResultsDAG.
        return ResultsDAG(self)

    def merge_base(self, versions):
        """
        Returns the merge base of multiple versions

        Args:
            versions: the versions you want to have the merge base for
        """

        # We have a linear history, we just need to get one version and go down
        # its parents and always store the earliest version in $version. Best
        # case scenario, we were already the merge_base and we will not find any
        version = next(iter(versions))
        merge_base = version
        while len(self._version_graph.parents(version)) > 0:
            version = self._version_graph.parents(version)[0]
            if version in versions:
                merge_base = version

        return merge_base


    def subDAG(self, versions):
        """
        Returns a DAG which contains the minimal subset of commits which compose
        all the specified versions.

        Args:
            versions: the versions you want to have the minimal history for
        """

        # let's be lazy and just return the graph we have, it is small anyway
        return copy.deepcopy(self._version_graph)

    def list_versions(self, head="HEAD", restrict_to_commits=[]):
        """
        List all the versions accessible from $head, limited to versions found
        in $restrict_to_commits and output them from $head down.

        Args:
            head: a valid version name that will be used as the top of the tree
            restrict_to_commits: a list of acceptable versions that will be used to filter the output
        """

        if head == "HEAD":
            head = self._last_version

        cur = head
        while cur is not None:
            if len(restrict_to_commits) == 0 or cur in restrict_to_commits:
                yield cur

            # Go to the next version (there can be only one or none)
            parents = self._version_graph.parents(cur)
            if len(parents) == 1:
                cur = parents[0]
            else:
                cur = None

    def version_range_list(self, before, after, ignore = set()):
        """
        Return the list of commits between $before and $after, not including $before.

        Args:
            before: a valid version name
            after: a valid version name
            ignore: set of version to ignore, preventing all paths to go through it
        """

        if after == "HEAD":
            after = self._last_version

        # First check that both before and after are in _version_graph
        if (before not in self._version_graph.nodes() or
            after not in self._version_graph.nodes()):
            return []

        res = []
        cur = before
        while True:
            if cur != before and cur not in ignore:
                res.append(cur)

            children = self._version_graph.children(cur)
            if len(children) == 1:
                cur = children[0]
            else:
                # We are at the end of the chain and we did not find after, abort...
                return res

        return res

    def version_description(self, version):
        """
        Return a one-line description of the version.

        Args:
            version: a valid version name
        """

        return self._desc.get(version, "")

    def version_parents(self, version):
        """
        Return the list of parents for the version $version.

        Args:
            version: a valid version name
        """

        return self._version_graph.parents(version)
