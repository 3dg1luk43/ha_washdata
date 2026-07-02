"""Integration tests for Recorder feature.

Recording is now managed entirely via the WashData panel (WebSocket commands:
start_recording, stop_recording, get_recording_status). The old OptionsFlow
record_cycle steps have been removed.
"""
