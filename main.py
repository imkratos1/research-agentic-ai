import argparse
import sys

from agent import config, pipeline


def show(event, state):
    print(f"[{event['time']}] {event['phase']:<14} {event['kind']:<8} {event['text']}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="DeepResearch Agent: research a topic and write a cited PDF report")
    parser.add_argument("topic", nargs="?", help="the research topic, in quotes")
    parser.add_argument("--depth", choices=list(config.DEPTH_SETTINGS), default=config.DEPTH)
    parser.add_argument("--resume", metavar="RUN_FOLDER", help="continue a stopped run, e.g. runs/20261010-201234_my-topic")
    args = parser.parse_args()
    if args.resume:
        state, folder = pipeline.resume(args.resume, show)
    elif args.topic:
        state, folder = pipeline.run(args.topic, args.depth, show)
    else:
        parser.error("give a topic, or --resume with a run folder")
    print(f"\nReport saved to {folder / 'report.pdf'}")


if __name__ == "__main__":
    main()
