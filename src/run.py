import argparse
import logging
from textwrap import dedent

from decompose import decompose_document_into_claims

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="gpt-4-1106-preview", help="The model to use."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Whether to log debug messages."
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run the main function.

    Args:
        args (argparse.Namespace): The parsed arguments.
    """
    document = dedent(
        """\
        The first thing to do is to understand the problem.
        The second thing to do is to decompose the problem into smaller problems.
        The third thing to do is to solve the smaller problems.
        """
    )
    claims = decompose_document_into_claims(document, args.model)
    print(claims)


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    main(args)
