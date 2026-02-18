# Register custom tool parsers into verl's ToolParser registry.
# This import must happen before verl's ToolAgentLoop.__init__ calls
# ToolParser.get_tool_parser(), which it does -- because verl loads our
# tool classes (verl_rl.tools.*) first, triggering this __init__.py.
import verl_rl.parsers.nemotron_tool_parser  # noqa: F401
