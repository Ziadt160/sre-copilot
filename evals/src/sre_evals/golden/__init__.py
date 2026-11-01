"""Golden-set tooling: pick, synthesize, review, validate, mirror."""

import os

# DeepEval phones home by default; the golden tooling never should.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
