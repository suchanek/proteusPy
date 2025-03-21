"""
This program takes a binary or octant class string and calls the display_overlay() function
to display the class disulfides.

:author: Eric G. Suchanek
:date: 2025-03-18
"""

import argparse

from proteusPy.DisulfideClassGenerator import DisulfideClassGenerator


def main():
    """
    Main function to parse arguments and call the display function.

    :return: None
    """
    parser = argparse.ArgumentParser(
        description="Display disulfides belonging to a specific class."
    )
    parser.add_argument(
        "class_string",
        help="Binary or octant class string (e.g., '00000' for binary, '22632' for octant)",
    )
    parser.add_argument(
        "--base",
        type=int,
        choices=[2, 8],
        default=8,
        help="Base of the class string (2 for binary, 8 for octant)",
    )
    parser.add_argument(
        "--theme",
        choices=["auto", "light", "dark"],
        default="auto",
        help="Background color theme",
    )
    parser.add_argument("--screenshot", action="store_true", help="Save a screenshot")
    parser.add_argument("--movie", action="store_true", help="Save a movie")
    parser.add_argument("--verbose", action="store_true", help="Display verbose output")
    parser.add_argument(
        "--fname",
        default="ss_overlay.png",
        help="Filename to save for the movie or screenshot",
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="Width of the display window"
    )
    parser.add_argument(
        "--height", type=int, default=1024, help="Height of the display window"
    )

    args = parser.parse_args()

    DisulfideClassGenerator.display_class_disulfides(
        args.class_string,
        light=args.theme,
        screenshot=args.screenshot,
        movie=args.movie,
        verbose=args.verbose,
        fname=args.fname,
        winsize=(args.width, args.height),
    )


if __name__ == "__main__":
    main()
