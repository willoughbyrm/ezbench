    <%! import os %>
    <%! import html %>
    <%! import base64 %>
    <%! from ezbench.report import Event, EventRenderingChange %>

    <%
         key = 0
         week_prev = -1
    %>

    <div class="css-treeview">
      <ul>

    % for c in events:
      <%
        key = key + 1
        id = "events_" + str(key)

        subentries = ""
        for t in events[c]:
          if len(subentries) > 0:
            subentries += ", "
          subentries += "{} {}(s)".format(len(events[c][t]), t)

        commit_date = c.commit_date()
        label = "{} ({}) - {}".format(str(c), subentries, commit_date)
        label = html.escape(label)
        type_checked = len(events[c]) == 1

        isocal = commit_date.isocalendar()
        week = "{}-{}".format(isocal[0], isocal[1])
      %>

      % if week != week_prev:
      <%
        week_prev = week
      %>
        <h4>Work week ${week}</h4>
      % endif

        <li>
          <input type="checkbox" id="${id}"/>
          <label class="tree_node" for="${id}">
            + ${label}
          </label>
          <ul>
      % for t in events[c]:
        <%
          key = key + 1
          id = "events_" + str(key)
        %>
            <li>
              <input type="checkbox" id="${id}" checked="${type_checked}"/>
              <label class="tree_node" for="${id}">
                + ${t} (${len(events[c][t])} test(s))
              </label>
              <ul>
        % for test in events[c][t]:
          % if not isinstance(test, Event):
            <%
              key = key + 1
              id = "events_" + str(key)
            %>
                <li>
                  <input type="checkbox" id="${id}" checked="${type_checked}"/>
                  <label class="tree_node" for="${id}">
                    + ${test} (${len(events[c][t][test])} reports(s))
                  </label>
                  <ul>
            % for r in events[c][t][test]:
              <%
                key = key + 1
                id = "events_" + str(key)
              %>
                    <li>
                      <input type="checkbox" id="${id}" checked="${type_checked}"/>
                      <label class="tree_node" for="${id}">
                        + ${r} (${len(events[c][t][test][r])} result(s))
                      </label>
                      <ul>
              % for e in events[c][t][test][r]:
                % if not isinstance(e, EventRenderingChange):
                        <li>${html.escape(e.short_desc)}</li>
                % else:
                  <%
                    # Reconstruct image path
                    new = e.result.average_image_file
                    old = new.replace(e.commit_range.new.sha1, e.commit_range.old.sha1)
                    diff = '{}_compare_{}'.format(new, os.path.basename(old))

                    old_image = ''
                    diff_image = ''
                    new_image = ''

                    if embed:
                      old_image = 'data:image/png;base64,' + base64.b64encode(open(old, 'rb').read()).decode()
                      diff_image = 'data:image/png;base64,' + base64.b64encode(open(diff, 'rb').read()).decode()
                      new_image = 'data:image/png;base64,' + base64.b64encode(open(new, 'rb').read()).decode()
                    else:
                      old_image = os.path.join(os.path.basename(old))
                      diff_image = os.path.join(os.path.basename(diff))
                      new_image = os.path.join(os.path.basename(new))
                  %>
                        <li>${html.escape(e.short_desc)}</li>
                        <img src="${old_image}" style="max-width:20%;">
                        <img src="${diff_image}" style="max-width:20%;">
                        <img src="${new_image}" style="max-width:20%;">
                % endif
              % endfor
                      </ul>
                    </li>
            % endfor
                  </ul>
                </li>
          % endif
        % endfor
              </ul>
            </li>
      % endfor
          </ul>
        </li>
    % endfor
      </ul>
    </div>
