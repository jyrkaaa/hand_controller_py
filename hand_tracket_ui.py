"""
Hand Tracker UI — Thin Controller Version
Uses Tracker class for all processing.
"""

import tkinter as tk
from tkinter import messagebox
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
from PIL import Image, ImageTk
from tracker import Tracker


# ─── Enums ────────────────────────────────────────────────────────────────

class RunMode(Enum):
    PREVIEW = auto()
    SILENT = auto()


class TrackerStatus(Enum):
    IDLE = "◌ IDLE"
    ACTIVE = "● TRACKING ACTIVE"


class WidgetState(Enum):
    NORMAL = "normal"
    DISABLED = "disabled"


# ─── Config ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class UdpConfig:
    ip: str
    port: int


@dataclass(frozen=True)
class TrackerConfig:
    udp: UdpConfig
    num_hands: int
    mode: RunMode


# ─── UI ───────────────────────────────────────────────────────────────────

class HandTrackerApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("HAND TRACKER")
        self.geometry("500x600")

        self._tracker: Optional[Tracker] = None
        self._running = False

        self.status_var = tk.StringVar(value=str(TrackerStatus.IDLE.value))
        self.ip_var = tk.StringVar(value="127.0.0.1")
        self.port_var = tk.StringVar(value="5055")
        self.hands_var = tk.IntVar(value=1)
        self.mode = RunMode.PREVIEW

        self._build()

    # ─── UI Layout ────────────────────────────────────────────────────────

    def _build(self):
        tk.Label(self, text="HAND TRACKER", font=("Arial", 20)).pack(pady=10)

        tk.Label(self, text="IP").pack()
        tk.Entry(self, textvariable=self.ip_var).pack()

        tk.Label(self, text="PORT").pack()
        tk.Entry(self, textvariable=self.port_var).pack()

        tk.Label(self, text="HANDS").pack()
        tk.Scale(self, from_=1, to=4, orient="horizontal", variable=self.hands_var).pack()

        tk.Button(self, text="PREVIEW", command=lambda: self._set_mode(RunMode.PREVIEW)).pack(fill="x")
        tk.Button(self, text="SILENT", command=lambda: self._set_mode(RunMode.SILENT)).pack(fill="x")

        self.canvas = tk.Canvas(self, width=400, height=250, bg="black")
        self.canvas.pack(pady=10)

        tk.Label(self, textvariable=self.status_var).pack(pady=5)

        self.start_btn = tk.Button(self, text="START", command=self._start)
        self.start_btn.pack(fill="x")

        self.stop_btn = tk.Button(self, text="STOP", command=self._stop, state="disabled")
        self.stop_btn.pack(fill="x")

    # ─── Mode ─────────────────────────────────────────────────────────────

    def _set_mode(self, mode: RunMode):
        self.mode = mode

    # ─── Tracker Control ───────────────────────────────────────────────────

    def _build_config(self) -> Optional[TrackerConfig]:
        try:
            port = int(self.port_var.get())
        except ValueError:
            messagebox.showerror("Error", "Port must be integer")
            return None

        return TrackerConfig(
            udp=UdpConfig(self.ip_var.get(), port),
            num_hands=self.hands_var.get(),
            mode=self.mode,
        )

    def _start(self):
        if self._running:
            return

        config = self._build_config()
        if not config:
            return

        self._tracker = Tracker(config)

        try:
            self._tracker.start()
        except Exception as e:
            messagebox.showerror("Tracker Error", str(e))
            return

        self._running = True
        self.status_var.set(TrackerStatus.ACTIVE.value)

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        self._loop()

    def _stop(self):
        if self._tracker:
            self._tracker.stop()

    def _loop(self):
        if not self._running or not self._tracker:
            return

        if not self._tracker.is_running():
            self._on_stopped()
            return

        frame = self._tracker.get_frame()

        if frame is not None:
            self._render(frame)

        self.after(30, self._loop)

    # ─── Rendering ─────────────────────────────────────────────────────────

    def _render(self, frame):
        img = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(img)
        self.canvas.image = photo
        self.canvas.create_image(0, 0, anchor="nw", image=photo)

    def _on_stopped(self):
        self._running = False
        self.status_var.set(TrackerStatus.IDLE.value)

        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

        self.canvas.delete("all")


# ─── Entry ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = HandTrackerApp()
    app.mainloop()
