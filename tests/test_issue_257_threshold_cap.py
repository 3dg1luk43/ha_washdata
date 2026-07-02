"""Issue #257: detection-threshold NumberSelector ceiling cap.

The OptionsFlow advanced_settings step had a hard ceiling at 500 W (start) /
100 W (stop). High-power devices (e.g. pumps suggesting ~1097 W) could not apply
suggestions because the selector rejected the value.

The options flow has been replaced by the WashData panel which uses plain number
inputs without such hard ceilings. The bug vector no longer exists in the current
code. Tests removed to match the slim config_flow.py.
"""
