# Verify your inference optimizations.

Agents make several model calls to complete one task. A serving change can
improve cache reuse, time to first token or another inference metric without
making the completed workflow faster.

RunEconomy connects those measurements to complete workflow latency and task
success. It can verify a change, identify a regression or decline to verify an
attractive result when the quality evidence is not strong enough.
