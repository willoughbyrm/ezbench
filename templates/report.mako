<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
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
  </head>

  <body>
    <h2>Events</h2>
    <%include file="events.mako"/>
  </body>
</html>
