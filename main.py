#!/usr/bin/env python3
"""Entry point for the Bike Power Simulator application."""

from gui import BikeSimApp


def main() -> None:
    app = BikeSimApp()
    app.mainloop()


if __name__ == "__main__":
    main()
