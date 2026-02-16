#!/bin/sh
# Run the internal tests against a UMASH_LONG_INPUTS=1 build.
# This exercises the optimised multi-block dispatch paths with
# the full set of unit tests (not just the test_public subset).
exec env UMASH_LONG_INPUTS=1 \
     "$(dirname "$(readlink -f "$0")")/run-tests.sh" "$@"
