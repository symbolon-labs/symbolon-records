# Sandbox process-identity correction at closure

The first closing driver stopped before running checks or writing a seal. It
tested whether /proc/3 and /proc/4 existed in a later command's process namespace,
using the diagnostic launch record's wrapper/child PIDs3/4. The current managed
sandbox creates a fresh PID namespace for each command; those numbers do not
identify the historical diagnostic processes across calls. No diagnostic result,
threshold, script or frozen input changed, and no diagnostic job was restarted.

Completion evidence is the original driver's successful subprocess.wait(), its
exit0/no-timeout completion.json, and terminal tool session17505 with exit0.
The child was reaped by its original parent; the wrapper completed. The final
closing driver therefore verifies these completion records rather than testing
unrelated namespace-local PID numbers. No process was terminated or modified.

This is an operational closing check correction, not a numerical-mechanics
failure or a relaxation of any scientific gate. The unsuccessful first closing
attempt produced only an AssertionError before checks and is preserved here.
