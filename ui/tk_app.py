from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.shared.training import TrainConfig, default_maze, random_maze, train_from_matrix


CELL = 26
COLORS = {
    0: "#3b4252",
    1: "#d8dee9",
    2: "#a3be8c",
    3: "#bf616a",
    "padding": "#4c566a",
    "path": "#88c0d0",
    "bg": "#2e3440",
    "panel": "#3b4252",
    "ink": "#eceff4",
    "muted": "#d8dee9",
}


class MazeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("REINFORCE Maze Lab")
        self.configure(bg=COLORS["bg"])
        self.resizable(False, False)

        self.maze = default_maze()
        self.brush = tk.IntVar(value=1)
        self.events = queue.Queue()
        self.path = []

        self._build_ui()
        self._draw_board()

    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=COLORS["panel"])
        style.configure("TLabel", background=COLORS["panel"], foreground=COLORS["ink"])
        style.configure("Muted.TLabel", background=COLORS["panel"], foreground=COLORS["muted"])
        style.configure("TButton", padding=8)
        style.configure("TRadiobutton", background=COLORS["panel"], foreground=COLORS["ink"])

        outer = ttk.Frame(self, padding=18)
        outer.grid(row=0, column=0)

        title = ttk.Label(outer, text="REINFORCE Maze Lab", font=("TkDefaultFont", 18, "bold"))
        title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 14))

        tools = ttk.Frame(outer, padding=12)
        tools.grid(row=1, column=0, sticky="n")
        ttk.Label(tools, text="Brushes", style="Muted.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        for idx, (text, value) in enumerate([("Wall", 1), ("Start", 2), ("End", 3), ("Erase", 0)], start=1):
            ttk.Radiobutton(tools, text=text, value=value, variable=self.brush).grid(row=idx, column=0, sticky="w")

        self.epochs = self._entry(tools, "Epochs", "120", 6)
        self.batch = self._entry(tools, "Batch", "64", 8)
        self.limit = self._entry(tools, "Limit", "100", 10)
        self.seed = self._entry(tools, "Seed", "", 12)
        ttk.Button(tools, text="Reset Maze", command=self._reset).grid(row=14, column=0, sticky="ew", pady=(14, 0))
        ttk.Button(tools, text="Random Maze", command=self._random_maze).grid(row=15, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(tools, text="Train", command=self._train).grid(row=16, column=0, sticky="ew", pady=(8, 0))

        self.canvas = tk.Canvas(
            outer,
            width=len(self.maze[0]) * CELL,
            height=len(self.maze) * CELL,
            bg=COLORS["bg"],
            highlightthickness=0,
        )
        self.canvas.grid(row=1, column=1, padx=16)
        self.canvas.bind("<Button-1>", self._paint_event)
        self.canvas.bind("<B1-Motion>", self._paint_event)

        stats = ttk.Frame(outer, padding=12)
        stats.grid(row=1, column=2, sticky="n")
        self.progress = ttk.Progressbar(stats, length=180, maximum=100)
        self.progress.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        self.stat_labels = {}
        for row, key in enumerate(["Status", "Epochs", "Success", "Avg Return", "Best Return", "Path", "Seconds"], start=1):
            ttk.Label(stats, text=key, style="Muted.TLabel").grid(row=row, column=0, sticky="w", pady=4)
            value = ttk.Label(stats, text="-")
            value.grid(row=row, column=1, sticky="e", pady=4)
            self.stat_labels[key] = value
        self.stat_labels["Status"].configure(text="Idle")

    def _entry(self, parent, label, value, row):
        ttk.Label(parent, text=label, style="Muted.TLabel").grid(row=row, column=0, sticky="w", pady=(12, 2))
        entry = ttk.Entry(parent, width=10)
        entry.insert(0, value)
        entry.grid(row=row + 1, column=0, sticky="ew")
        return entry

    def _draw_board(self):
        self.canvas.configure(width=len(self.maze[0]) * CELL, height=len(self.maze) * CELL)
        self.canvas.delete("all")
        path = {tuple(item) for item in self.path}
        for r, row in enumerate(self.maze):
            for c, value in enumerate(row):
                x0 = c * CELL
                y0 = r * CELL
                x1 = x0 + CELL
                y1 = y0 + CELL
                fill = COLORS[value]
                outline = COLORS["padding"] if self._is_padding(r, c) else COLORS["bg"]
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline=outline)
                if (r, c) in path and value == 0:
                    self.canvas.create_rectangle(x0 + 5, y0 + 5, x1 - 5, y1 - 5, outline=COLORS["path"], width=2)

    def _paint_event(self, event):
        row = event.y // CELL
        col = event.x // CELL
        if not (0 <= row < len(self.maze) and 0 <= col < len(self.maze[0])):
            return
        brush = self.brush.get()
        if brush in (2, 3) and not self._is_padding(row, col):
            self.stat_labels["Status"].configure(text="Start/end must be border cells")
            return
        if self._is_padding(row, col) and brush == 0:
            self.stat_labels["Status"].configure(text="Padding stays walled")
            return
        if self.maze[row][col] in (2, 3) and brush == 1:
            return
        if brush in (2, 3):
            for r, line in enumerate(self.maze):
                for c, value in enumerate(line):
                    if value == brush:
                        self.maze[r][c] = 1 if self._is_padding(r, c) else 0
        self.maze[row][col] = brush
        self.path = []
        self._draw_board()

    def _train(self):
        self.stat_labels["Status"].configure(text="Training")
        self.progress.configure(value=0)
        config = TrainConfig(
            epochs=int(self.epochs.get()),
            sampling_quantity=int(self.batch.get()),
            episode_limit=int(self.limit.get()),
            seed=int(self.seed.get()) if self.seed.get().strip() else None,
        )
        thread = threading.Thread(target=self._train_worker, args=(config,), daemon=True)
        thread.start()
        self.after(100, self._drain_events)

    def _train_worker(self, config):
        try:
            result = train_from_matrix(self.maze, config, self.events.put)
            self.events.put({"done": result})
        except Exception as exc:
            self.events.put({"error": str(exc)})

    def _drain_events(self):
        while not self.events.empty():
            event = self.events.get()
            if "done" in event:
                self._apply_result(event["done"])
                return
            if "error" in event:
                self.stat_labels["Status"].configure(text=event["error"])
                return
            total = max(1, event["total_epochs"])
            self.progress.configure(value=((event["epoch"] + 1) / total) * 100)
            self.stat_labels["Epochs"].configure(text=str(event["epoch"] + 1))
            self.stat_labels["Success"].configure(text=f"{event['success_rate']:.0%}")
            self.stat_labels["Avg Return"].configure(text=f"{event['avg_return']:.3f}")
            self.stat_labels["Best Return"].configure(text=f"{event['best_return']:.3f}")
            path_prefix = "" if event["path_success"] else ">"
            self.stat_labels["Path"].configure(text=f"{path_prefix}{event['path_steps']}/{event['shortest_path']}")
            self.stat_labels["Seconds"].configure(text=f"{event['seconds']:.2f}")
        self.after(100, self._drain_events)

    def _apply_result(self, result):
        self.path = result["greedy"]["path"]
        self.progress.configure(value=100)
        self.stat_labels["Status"].configure(text="Solved" if result["greedy"]["success"] else "Stopped")
        self.stat_labels["Epochs"].configure(text=str(result["epochs_run"]))
        self.stat_labels["Success"].configure(text=f"{result['final_eval']['success_rate']:.0%}")
        self.stat_labels["Avg Return"].configure(text=f"{result['final_eval']['avg_return']:.3f}")
        self.stat_labels["Best Return"].configure(text=f"{result['final_eval']['best_return']:.3f}")
        self.stat_labels["Path"].configure(text=f"{result['greedy']['steps']}/{result['shortest_path']}")
        self.stat_labels["Seconds"].configure(text=f"{result['seconds']:.2f}")
        self._draw_board()

    def _reset(self):
        self.maze = default_maze()
        self.path = []
        self._draw_board()

    def _random_maze(self):
        self.maze = random_maze()
        self.path = []
        self._draw_board()

    def _is_padding(self, row, col):
        return row == 0 or col == 0 or row == len(self.maze) - 1 or col == len(self.maze[0]) - 1


if __name__ == "__main__":
    MazeApp().mainloop()
