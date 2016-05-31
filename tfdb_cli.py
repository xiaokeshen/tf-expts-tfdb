"""TensorFlow debugger (tfdb) command-line interface (CLI)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np  # pylint: disable=unused-import
from tensorflow.python.client import debugger


class CommandLineDebugger(object):
  """Command-line interface (CLI) debugger for tfdb."""

  def __init__(self, debug_sess, node, num_times=1, feed=None,
               auto_step=False):
    """Create an instance of tfdb CLI.

    This constructor takes the node to execute and number of times to execute
    as arguments and creates a tfdb CLI object. Unless auto_step is set to True,
    it will break right after the first node (_SOURCE) once the instance is
    created.

    Args:
      debug_sess: The debug Session object that this tfdb CLI will use
      node: The node to execute (evaluate)
      num_times: Number of times to execute the node for (Default: 1)
      feed: Optional feed_dict for the execution
      auto_step: Optional boolean indicating whether the debugger should step
        through all steps automatically, without breaking.
    """
    print("In CommandLineDebugger __init__")
    self._debug_round = debugger.DebugRound(debug_sess,
                                            node,
                                            num_times=num_times,
                                            feed=feed)

    self._node_order = self._debug_round.query_node_order()
    print("_node_order = %s" % repr(self._node_order))

    self._print_node_order()

    self._prompt = "tfdb> "
    self._initialize_help()

    # Handles to node breakpoints: dict
    #   keys: breakpoint names (maybe without prefix)
    #   values: lists of breakpoint handles
    self._node_bps = {}

    if auto_step:
      self.continue_to_end()
    else:
      print("Calling cli_loop()")
      self.cli_loop()

  def _initialize_help(self):
    """Initialize the help information for the CLI."""

    c_hlp = {}  # Coarse help
    f_hlp = {}  # Fine help

    # Coarse help
    c_hlp["break"] = "Insert a breakpoint"
    c_hlp["breakpoints"] = "Inspect all currently-set breakpoints"
    c_hlp["clear"] = "Clear all breakpoints"
    c_hlp["cont"] = "Continue to the end or to a specific node"
    c_hlp["delete"] = "Delete a breakpoint"
    c_hlp["exit"] = "Exit the tfdb CLI"
    c_hlp["help"] = "Print this help messag"
    c_hlp["inject"] = "Inject Tensor value to a node"
    c_hlp["print"] = "Print Tensor values on node"
    c_hlp["step"] = "Step to following node(s)"
    c_hlp["where"] = "Query the position the debugger is at"

    # Fine help
    f_hlp["break"] = """Insert breakpoint

    Syntax: break <node>

    Insert a breakpoint at the specified node.
    In the case of a multi-repetition (num_times > 1) debug round, node can
    either be a node name prefixed with the 0-based repetition number, e.g.,

    break 0_node_a

    in which case the debugger will break at node_a only in the first
    repetition. Or the non-prefixed node name can be used, e.g.,

    break node_a

    in which case the debugger will break at node_a in every repetition of the
    run."""

    f_hlp["breakpoints"] = """List all breakpoints

    Syntax: breakpoints

    Print a list of all breakpoints that are set currently in the debugger."""

    f_hlp["clear"] = """Clear all currently-set breakpoints

    Syntax: clear"""

    f_hlp["cont"] = """Continue

    Syntax: cont [node]

    If node is unspecified, will continue to the end or when a breakpoint is
    first encountered.
    If node is specified and a valid node in the executed subgraph, will
    continue to the node or when a breakpoint is first encountered.

    In the case of a multi-repetition run, the node name can either be prefixed
    by the 0-based repetition number (e.g., 0_node_a) or unprefxied, in which
    case the debugger will try to continue to the node in the current
    repetition"""

    f_hlp["delete"] = """Delete breakpoint

    Syntax: delete [node]

    Delete a breakpoint by node name.
    In the case of multi-repetition runs, the node name should match the one
    used when setting the breakpoint, e.g.,

    break 2_node_a
    delete 2_node_a

    break node_a
    delete node_a"""

    f_hlp["exit"] = """Exit the current tfdb CLI

    Syntax: exit"""

    f_hlp["help"] = """Print help information

    Syntax: help [topic]

    If topic is unspecified, print global (coarse) help for all commands of the
    debugger.
    If topic is specified and a valid command or topic  of the debug, print the
    fine-grained help info for the topic."""

    f_hlp["inject"] = """Inject Tensor value to current node

    Syntax: inject <value>

    Inject a new Tensor value to the current node.
    "value" is an numpy array, e.g.:

    inject 3
    inject [-0.1, 0.2, 0.5]
    inject np.ones([40, 10])"""

    f_hlp["print"] = """Print Tensor value on a node

    Syntax: print [noded]

    If node is unspecified, print the Tensor value on the current node.
    If node is specified, print the Tensor value on the specified node (through
    node name). This assumes that the node does not exist in the executed
    subgraph or if it and it has finished executing at least once in prior
    repetitions."""

    f_hlp["step"] = """Step

    Syntax: step [num_times=1]

    If num_times == 1 (default), step to the next node.
    If num_times > 1, step to the n-th node from the current."""

    f_hlp["where"] = """Query debugger state

    where [--full]

    Print the current position of the debugger in the executed subgraph.
    If there are too many nodes in the graph, nodes that are far from the
    current node will be omitted.
    The "full" disables this brief mode and prints all the nodes."""

    self._c_hlp = c_hlp
    self._f_hlp = f_hlp

  def _print_node_order(self):
    print("Subgraph to be executed consists of %d nodes: " %
          len(self._node_order))
    self._print_where(self._debug_round.where())

  def _print_where(self, idx, brief=False):
    """Print the debug round's current location.

    Args:
      idx: Current node index.
      brief: Whether the print out should use a brief format (Default: False).
    """
    brief_lim = 2  # Number of extra nodes to see on each side

    omitted_head = False
    omitted_tail = False
    if not brief:
      nodes_to_print = self._node_order
      begin_idx = 0
    else:
      begin_idx = idx - brief_lim
      if begin_idx < 0:
        begin_idx = 0
      omitted_head = begin_idx > 0

      end_idx = idx + brief_lim
      if end_idx >= len(self._node_order):
        end_idx = len(self._node_order) - 1
      omitted_tail = end_idx < len(self._node_order) - 1

      nodes_to_print = self._node_order[begin_idx : end_idx + 1]

    if omitted_head:
      print("  ...")

    counter = begin_idx
    for node_name in nodes_to_print:
      if counter == idx:
        pointer_str = "-> "
      else:
        pointer_str = "   "
      print("  %s(%d) %s" % (pointer_str, counter, node_name))
      counter += 1

    if omitted_tail:
      print("  ...")

  def _print_error(self, err_msg):
    print(err_msg)
    print("")

  def _get_curr_node(self):
    where_output = self._debug_round.where()
    curr_node = self._node_order[where_output]
    return curr_node, where_output

  def _get_curr_node_val(self):
    curr_node, _ = self._get_curr_node()
    val = self._debug_round.inspect_value(curr_node)
    return val

  def _get_node_val(self, node_name):
    val = None
    try:
      val = self._debug_round.inspect_value(node_name)
    except ValueError as e:
      raise e

    return val

  def _print_node_val(self, node_name, print_none=True):
    val = self._get_node_val(node_name)

    if print_none or val is not None:
      if val is not None:
        shape_str = "(shape = %s)" % repr(val.shape)
      else:
        shape_str = ""

    print("%s = %s %s" %
          (node_name, repr(val), shape_str))

  def _print_curr_node_val(self, print_none=True, time_elapsed=None):
    curr_node, where_output = self._get_curr_node()
    val = self._get_curr_node_val()

    timing_str = ""
    if time_elapsed is not None:
      timing_str = "[%.1f ms] " % (time_elapsed * 1000)

    if print_none or val is not None:
      if val is not None:
        shape_str = "(shape = %s)" % repr(val.shape)
      else:
        shape_str = ""

      print("%s(%d) %s = %s %s" %
            (timing_str, where_output, curr_node, repr(val), shape_str))

  def print_help(self, topic=None):
    if topic is None:
      # Print coarse-level help info
      for key in sorted(self._c_hlp):
        print("%s - %s" % (key.rjust(16, " "), self._c_hlp[key]))
    else:
      # Print fine-grained help info, for a topic (command)
      if topic in self._f_hlp:
        print(self._f_hlp[topic])
      else:
        print("ERROR: Help topic \"%s\" does not exist" % topic)

  def continue_to_end(self):
    while True:
      self._debug_round.step()

      if self._debug_round.is_complete():
        self._debug_round.step()
        break

  def cli_loop(self):
    while True:
      if self._debug_round.is_complete():
        self._debug_round.step()
        break

      command = raw_input(self._prompt).strip()

      if command == "step":
        try:
          _, step_time_elapsed = self._debug_round.step()
        except ValueError as e:
          self._print_error(e)
          continue

        self._print_curr_node_val(time_elapsed=step_time_elapsed)

      elif command.startswith("step "):
        try:
          num_steps = int(command.replace("step ", "").strip())
        except ValueError as e:
          self._print_error("ERROR: Invalid step command")
          continue

        if num_steps < 1:
          self._print_error("ERROR: Invalid number of steps: %d" %
                            num_steps)
          continue

        try:
          _, step_time_elapsed = self._debug_round.step(num_steps=num_steps)
        except ValueError as e:
          self._print_error(e)
          continue

        self._print_curr_node_val(time_elapsed=step_time_elapsed)

      elif command == "where":
        where_output = self._debug_round.where()
        self._print_where(where_output, brief=True)
      elif command == "where --full":
        where_output = self._debug_round.where()
        self._print_where(where_output, brief=False)
      elif command == "cont":
        try:
          _, time_elapsed = self._debug_round.cont()
          self._print_curr_node_val(time_elapsed=time_elapsed)
        except ValueError as e:
          self._print_error(e)
          continue

      elif command.startswith("cont "):
        node_name = command.replace("cont ", "").strip()

        try:
          _, time_elapsed = self._debug_round.cont(node_name=node_name)
          self._print_curr_node_val(time_elapsed=time_elapsed)
        except ValueError as e:
          self._print_error(e)
          continue

      elif command == "print":
        self._print_curr_node_val()
      elif command.startswith("print "):
        node_name = command.replace("print ", "").strip()

        try:
          self._print_node_val(node_name)
        except ValueError as e:
          self._print_error(e)
          continue

      elif command.startswith("inject "):
        node_name, _ = self._get_curr_node()

        new_val_str = command.replace("inject ", "").strip()

        try:
          # pylint: disable=eval-used
          new_val = eval("np.array(%s)" % new_val_str)
        except Exception as e:  # pylint: disable=broad-except
          self._print_error(e)
          continue

        old_val = self._get_curr_node_val()

        # Check shape compatibility
        if new_val.shape != old_val.shape:
          print("WARNING: Injection shape mismatch!")

        new_val = new_val.astype(old_val.dtype)
        try:
          self._debug_round.inject_value(new_val)
          print("Injected value to node %s: %s" % (node_name, new_val))
        except ValueError as e:
          self._print_error(e)
          continue

      elif command.startswith("break "):
        # Insert breakpoint (after) node
        node_name = command.replace("break ", "").strip()

        try:
          # Try to insert the breakpoint and get the handle
          bp_handle = self._debug_round.break_after(node_name)
          self._node_bps[node_name] = bp_handle
        except ValueError as e:
          self._print_error(e)
          continue

      elif command.startswith("delete "):
        # Delete a breakpoint
        node_name = command.replace("delete ", "").strip()

        if node_name not in self._node_bps:
          self._print_error("ERROR: Breakpoint %s is not set" % node_name)
          continue
        bp_handle = self._node_bps[node_name]

        try:
          self._debug_round.remove_breakpoint(bp_handle)
          self._node_bps.pop(node_name)
        except ValueError as e:
          self._print_error(e)
          continue

      elif command == "breakpoints":
        # List breakpoints
        node_bps, _ = self._debug_round.get_breakpoints()
        for bp_name in node_bps:
          node_list = node_bps[bp_name]
          if len(node_list) == 1:
            print("  %s" % bp_name)
          else:
            print("  %s: %s" % (bp_name, repr(node_list)))

      elif command == "clear":
        # Clear all breakpoints
        for node_name in self._node_bps:
          bp_handle = self._node_bps[node_name]
          try:
            self._debug_round.remove_breakpoint(bp_handle)
          except ValueError as e:
            self._print_error(e)

        self._node_bps.clear()

      elif command == "help":
        self.print_help()

      elif command.startswith("help "):
        topic = command.replace("help ", "")
        self.print_help(topic=topic)

      elif command == "exit":
        self.continue_to_end()

      else:
        self._print_error("Invalid command: %s" % command)
        continue

      print("")
