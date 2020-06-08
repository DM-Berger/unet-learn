import argparse

from pathlib import Path
from typing import Any, Dict


def res(string: str) -> str:
    return str(Path(string).resolve())


def bool_parse(string: str) -> bool:
    lower = string.lower()
    if lower == "true" or lower == "t":
        return True
    if lower == "false" or lower == "f":
        return False
    return bool(string)


# just a helper
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ComputeCanada arguments")
    parser.add_argument(
        "-l", "--logs", metavar="<logging directory>", type=res, nargs=1, action="store"
    )
    parser.add_argument(
        "-g", "--gpus", metavar="<gpus>", type=int, nargs=1, action="store", default=1
    )
    parser.add_argument(
        "-c", "--checkdir", metavar="<checkpoint directory>", type=res, nargs=1, action="store"
    )
    parser.add_argument(
        "--epochs-min", metavar="<min epochs>", type=int, nargs=1, action="store", default=1
    )
    parser.add_argument(
        "--epochs-max", metavar="<max epochs>", type=int, nargs=1, action="store", default=10
    )
    parser.add_argument(
        "--resume",
        metavar="<checkpoint.cpkt file>",
        type=res,
        nargs=1,
        action="store",
        default=None,
    )
    # 'store_true' means default is false, i.e. "on flag present, action is: store TRUE"
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--devrun", action="store_true")
    parser.add_argument("--local", action="store_true")
    return parser


def get_args() -> Dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args()

    # Handle when argparse randomly decides to wrap args in lists
    def delist(item: Any) -> Any:
        return item[0] if isinstance(item, list) else item

    return {
        "half": delist(args.half),
        "devrun": delist(args.devrun),
        "local": delist(args.local),
        "overfit": delist(args.overfit),
        "gpus": delist(args.gpus),
        "logs": delist(args.logs),
        "epochs_min": delist(args.epochs_min),
        "epochs_max": delist(args.epochs_max),
        "resume": delist(args.resume),
        "checkdir": delist(args.checkdir),
    }
