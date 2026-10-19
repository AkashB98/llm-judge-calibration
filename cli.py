#!/usr/bin/env python3
"""judge-lab CLI.

  judges              list available judge profiles
  grade --judge NAME  grade the golden corpus (JSON verdicts to stdout)
  calibrate --judge NAME   print the human-readable scorecard
  check --judge NAME  CI gate: exit 0 PASS, 1 WARN, 2 FAIL
  demo                end-to-end asserted demo (incl. live API round-trip)
  serve [--port N]    JSON API on 127.0.0.1 (default 8078)
"""

import argparse
import json
import sys

from judge_lab import judges as judges_mod
from judge_lab.corpus import load_corpus
from judge_lab.report import grade_all, render_text, scorecard

EXIT = {"PASS": 0, "WARN": 1, "FAIL": 2}


def cmd_judges(_args):
    for j in judges_mod.list_judges():
        print("%-18s %s" % (j["name"], j["description"]))


def cmd_grade(args):
    judge = judges_mod.get_judge(args.judge)
    items = load_corpus()
    verdicts = grade_all(judge, items)
    print(json.dumps([{"item_id": i["id"], "golden": i["golden"], **v}
                      for i, v in zip(items, verdicts)],
                     sort_keys=True, indent=2))


def cmd_calibrate(args):
    judge = judges_mod.get_judge(args.judge)
    card = scorecard(judge)
    print(render_text(card))


def cmd_check(args):
    judge = judges_mod.get_judge(args.judge)
    card = scorecard(judge)
    print("judge %s: %s" % (judge.name, card["verdict"]))
    for r in card["reasons"]:
        print("  ! %s" % r)
    return EXIT[card["verdict"]]


def cmd_demo(_args):
    import demo
    demo.main()


def cmd_serve(args):
    from judge_lab.api import serve
    serve(port=args.port)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="judge-lab", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("judges", help="list judge profiles")

    g = sub.add_parser("grade", help="grade the golden corpus as JSON")
    g.add_argument("--judge", default="calibrated")

    c = sub.add_parser("calibrate", help="print the scorecard for a judge")
    c.add_argument("--judge", default="calibrated")

    k = sub.add_parser("check", help="CI gate: exit 0 PASS / 1 WARN / 2 FAIL")
    k.add_argument("--judge", default="calibrated")

    sub.add_parser("demo", help="end-to-end asserted demo")

    s = sub.add_parser("serve", help="JSON API on 127.0.0.1")
    s.add_argument("--port", type=int, default=8078)

    args = ap.parse_args(argv)
    fn = {"judges": cmd_judges, "grade": cmd_grade, "calibrate": cmd_calibrate,
          "check": cmd_check, "demo": cmd_demo, "serve": cmd_serve}[args.cmd]
    try:
        rc = fn(args)
    except ValueError as e:
        print("error: %s" % e, file=sys.stderr)
        return 3
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main())
