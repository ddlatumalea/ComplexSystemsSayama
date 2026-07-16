"""Simulates the discrete-time model:

    x_t = a * x_{t-1} + b

where `a` is the ratio between consecutive states and `b` is a constant
intercept added at every step.

Follows the standard simulation structure used throughout the course:
1) Initialize - set up the initial state
2) Observe   - record/report the current state
3) Update    - advance the state by one time step

There are several behaviours:
- We can create an exponential decay or growth model, depending when b crosses a. 
- We can find a zigzag pattern once a is negative that can be inverted when b crosses a.
"""

from dataclasses import dataclass, field


@dataclass
class Model:
    """Holds parameters and state for the x_t = a * x_{t-1} + b simulation."""

    a: float = 1.1
    b: float = 0.0
    x0: float = 1.0

    x: float = field(init=False)
    history: list[float] = field(init=False)

    def initialize(self) -> None:
        self.x = self.x0
        self.history = [self.x]

    def observe(self) -> None:
        self.history.append(self.x)

    def update(self) -> None:
        self.x = self.a * self.x + self.b

    def run(self, steps: int) -> list[float]:
        self.initialize()
        for _ in range(steps):
            self.update()
            self.observe()
        return self.history


class SimulationWindow:
    """Plots the model's trajectory with controls for `a`, `b`, and step count."""

    A_MIN, A_MAX, A_STEP = -3.0, 3.0, 0.01
    B_MIN, B_MAX, B_STEP = -5.0, 5.0, 0.01

    def __init__(self, model: Model, steps: int = 30) -> None:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets

        self.model = model
        self.steps = steps

        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle("x_t = a * x_(t-1) + b")
        layout = QtWidgets.QVBoxLayout(self.window)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("bottom", "t")
        self.plot_widget.setLabel("left", "x")
        self.curve = self.plot_widget.plot(symbol="o")
        layout.addWidget(self.plot_widget)

        self.a_label = QtWidgets.QLabel()
        layout.addWidget(self.a_label)

        self.a_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.a_slider.setMinimum(round(self.A_MIN / self.A_STEP))
        self.a_slider.setMaximum(round(self.A_MAX / self.A_STEP))
        self.a_slider.setValue(round(self.model.a / self.A_STEP))
        self.a_slider.valueChanged.connect(self._on_a_changed)
        layout.addWidget(self.a_slider)

        self.b_label = QtWidgets.QLabel()
        layout.addWidget(self.b_label)

        self.b_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.b_slider.setMinimum(round(self.B_MIN / self.B_STEP))
        self.b_slider.setMaximum(round(self.B_MAX / self.B_STEP))
        self.b_slider.setValue(round(self.model.b / self.B_STEP))
        self.b_slider.valueChanged.connect(self._on_b_changed)
        layout.addWidget(self.b_slider)

        controls = QtWidgets.QHBoxLayout()

        controls.addWidget(QtWidgets.QLabel("steps"))
        self.steps_spin = QtWidgets.QSpinBox()
        self.steps_spin.setRange(1, 10_000)
        self.steps_spin.setValue(self.steps)
        self.steps_spin.valueChanged.connect(self._on_steps_changed)
        controls.addWidget(self.steps_spin)

        layout.addLayout(controls)

        self._update_plot()

    def _on_a_changed(self, value: int) -> None:
        self.model.a = value * self.A_STEP
        self._update_plot()

    def _on_b_changed(self, value: int) -> None:
        self.model.b = value * self.B_STEP
        self._update_plot()

    def _on_steps_changed(self, value: int) -> None:
        self.steps = value
        self._update_plot()

    def _update_plot(self) -> None:
        history = self.model.run(self.steps)
        self.curve.setData(history)
        self.plot_widget.setTitle(f"x_t = {self.model.a:.2f} * x_(t-1) + {self.model.b:.2f}")
        self.a_label.setText(f"a = {self.model.a:.2f}")
        self.b_label.setText(f"b = {self.model.b:.2f}")

    def show(self) -> None:
        self.window.show()


def main() -> None:
    model = Model(a=1.1, x0=1.0)
    history = model.run(steps=30)
    print(history)

    try:
        from pyqtgraph.Qt import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        sim_window = SimulationWindow(model, steps=30)
        sim_window.show()
        app.exec()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
