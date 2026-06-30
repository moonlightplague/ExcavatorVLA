import argparse

import excavator_dataset_tools as tools


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the excavator dataset dashboard.")
    parser.add_argument("--root", default="excavator_auto_dataset")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    tools.serve_dashboard(args.root, host=args.host, port=args.port, open_browser=args.open)


if __name__ == "__main__":
    main()
