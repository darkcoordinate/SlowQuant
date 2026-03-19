import re
import time
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
import sys
import math as mt
LOGFILE = "filename.txt"


NSTEPS = int(sys.argv[1])

col = int(mt.sqrt(NSTEPS)) + 1

console = Console()

steps_done = [False] * NSTEPS
pattern = re.compile(r"step\s*:\s*(\d+)")

def read_log():
    """Read log file and update finished steps."""
    global steps_done 
    steps_done = [False] * NSTEPS
    try:
        with open(LOGFILE, "r") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    step = int(m.group(1))
                    if 0 <= step < NSTEPS:
                        steps_done[step] = True
    except FileNotFoundError:
        pass


def make_grid():
    """Create a grid of step boxes."""
    table = Table(show_header=False)

    cols = col
    for _ in range(cols):
        table.add_column(justify="center")

    for i in range(0, NSTEPS, cols):
        row = []
        for j in range(cols):
            idx = i + j
            if idx >= NSTEPS:
                row.append("")
                continue

            if steps_done[idx]:
                row.append(f"[black on green]{idx:04d}[/]")
            else:
                row.append(f"[white on red]{idx:04d}[/]")
        table.add_row(*row)

    return Panel(table, title="Step Progress")


with Live(make_grid(), refresh_per_second=4) as live:
    while True:
        read_log()
        live.update(make_grid())
        time.sleep(0.5)