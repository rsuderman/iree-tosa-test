from absl import app
from absl import flags
import json
import logging
import multiprocessing
import os
import tqdm

from test_case import TestCase

FLAGS = flags.FLAGS

flags.DEFINE_string('testsuite_dir', None, "Test suite directory")
flags.DEFINE_string('group', None, "Filter group")
flags.DEFINE_string('operator', None, "Filter operator")
flags.DEFINE_string('test', None, "Filter test")
flags.DEFINE_string('output', "/tmp/details.txt", "Output location")

flags.DEFINE_bool('include_error', False, "Include error tests")


def AddAllSubfolders(directory, include=None, exclude=None):
    if isinstance(directory, list) or isinstance(directory, set):
        ret = set()
        for d in directory:
            ret = ret.union(AddAllSubfolders(d, include, exclude))
        return ret

    lst = os.listdir(directory)
    if include is not None:
        lst = [f for f in lst if include in f]

    if exclude is not None:
        lst = [f for f in lst if exclude not in f]

    ret = set()
    for f in lst:
        f_path = os.path.join(directory, f)
        if os.path.isdir(f_path):
            ret.add(f_path)
    return ret


def run(test):
    testcase = TestCase(test)
    status = testcase.run()
    errmsg = "" if status.success() else f"Error: {status.message()}"
    return (test, {
        "status": status.success(),
        "message": errmsg,
        "details": status.details()
    })


def runjobs(tests):
    if len(tests) < 5:
        results = {}
        for test in tests:
            test, result = run(test)
            results[test] = result
        return results

    tests = sorted(tests)
    results = {}
    with multiprocessing.Pool(16) as pool:
        for pair in tqdm.tqdm(pool.imap_unordered(run, tests),
                              total=len(tests)):
            test, result = pair
            if not result["status"]:
                results[test] = result
        print("Completed")
    return results


def main(argv):
    # Disable for sane reasons
    logger = logging.getLogger()
    logger.disabled = True

    testsuite_dir = FLAGS.testsuite_dir
    if testsuite_dir is None:
        fulldir = os.path.dirname(os.path.realpath(__file__))
        base = os.path.split(fulldir)[0]
        testsuite_dir = os.path.join(base, "third_party", "conformance_tests",
                                     "operators")

    groups = AddAllSubfolders(testsuite_dir, FLAGS.group)
    operators = AddAllSubfolders(groups, FLAGS.operator)

    exclude_check = None if FLAGS.include_error else "ERRORIF"
    tests = AddAllSubfolders(operators,
                             include=FLAGS.test,
                             exclude=exclude_check)

    results = runjobs(tests)

    with open(FLAGS.output, "w") as f:
        f.write(json.dumps(results, indent=2))

    failures = 0
    for a in results:
        failures += int(not results[a]["status"])

    successes = len(tests) - failures
    print(f"Full details: {FLAGS.output}")
    print(f"Passed Tests: {successes}")
    print(f"Failed Tests: {failures}")


if __name__ == '__main__':
    app.run(main)
