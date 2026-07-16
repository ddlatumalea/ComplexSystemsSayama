"""Simulate a discrete time model using two variables:

    x_t = 0.5 * x_{t-1} + y_{t-1}
    y_t = -0.5 * x_{t-1} + y_{t-1}
    
This can be done using linear algebra such that we have:

    x           0.5     1       x
    y       =   -0.5    1       y
        t                           t-1
"""

from dataclasses import dataclass, field
import numpy as np

@dataclass
class Model:
    """Holds parameters and state for simulation"""
    X0: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0]))
    c1: float = 0.5
    c2: float = 1.0

    X: np.ndarray = field(init=False)
    history: list[np.ndarray] = field(init=False)

    @property
    def coef(self) -> np.ndarray:
        return np.array([[self.c1, self.c2], [-self.c1, self.c2]])

    def initialize(self) -> None:
        self.X = self.X0
        self.history = [self.X]

    def observe(self) -> None:
        self.history.append(self.X)

    def update(self) -> None:
        self.X = self.coef @ self.X

    def run(self, steps: int) -> list[float]:
        self.initialize()
        for _ in range(steps):
            self.update()
            self.observe()

        return self.history_array()

    def history_array(self) -> np.ndarray:
        """Stack the recorded states into a (steps + 1, n_vars) array."""
        return np.stack(self.history, axis=0)
    
class SimulationWindow:
    """Plots trajectories and phase space with sliders to adjust `c1` and `c2`."""

    C1_MIN, C1_MAX, C1_STEP = -2.0, 2.0, 0.01
    C2_MIN, C2_MAX, C2_STEP = -2.0, 2.0, 0.01

    def __init__(self, model: Model, steps: int = 30) -> None:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets

        self.model = model
        self.steps = steps

        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle("Oscillation")
        layout = QtWidgets.QVBoxLayout(self.window)

        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.resize(900, 450)
        layout.addWidget(self.graphics)

        self.trajectories = self.graphics.addPlot(title="Trajectories of x and y")
        self.trajectories.addLegend()
        self.trajectories.setLabel("bottom", "t")
        self.trajectories.setLabel("left", "value")
        self.x_curve = self.trajectories.plot(pen="r", name="x")
        self.y_curve = self.trajectories.plot(pen="b", name="y")

        self.phase_space = self.graphics.addPlot(title="Phase space")
        self.phase_space.setLabel("bottom", "x")
        self.phase_space.setLabel("left", "y")
        self.phase_curve = self.phase_space.plot(pen="g")

        self.c1_label = QtWidgets.QLabel()
        layout.addWidget(self.c1_label)

        self.c1_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.c1_slider.setMinimum(round(self.C1_MIN / self.C1_STEP))
        self.c1_slider.setMaximum(round(self.C1_MAX / self.C1_STEP))
        self.c1_slider.setValue(round(self.model.c1 / self.C1_STEP))
        self.c1_slider.valueChanged.connect(self._on_c1_changed)
        layout.addWidget(self.c1_slider)

        self.c2_label = QtWidgets.QLabel()
        layout.addWidget(self.c2_label)

        self.c2_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.c2_slider.setMinimum(round(self.C2_MIN / self.C2_STEP))
        self.c2_slider.setMaximum(round(self.C2_MAX / self.C2_STEP))
        self.c2_slider.setValue(round(self.model.c2 / self.C2_STEP))
        self.c2_slider.valueChanged.connect(self._on_c2_changed)
        layout.addWidget(self.c2_slider)

        self._update_plot()

    def _on_c1_changed(self, value: int) -> None:
        self.model.c1 = value * self.C1_STEP
        self._update_plot()

    def _on_c2_changed(self, value: int) -> None:
        self.model.c2 = value * self.C2_STEP
        self._update_plot()

    def _update_plot(self) -> None:
        history = self.model.run(self.steps)
        x = history[:, 0]
        y = history[:, 1]

        self.x_curve.setData(x)
        self.y_curve.setData(y)
        self.phase_curve.setData(x, y)

        self.c1_label.setText(f"c1 = {self.model.c1:.2f}")
        self.c2_label.setText(f"c2 = {self.model.c2:.2f}")

    def show(self) -> None:
        self.window.show()


def main() -> None:
    model = Model()

    from pyqtgraph.Qt import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    sim_window = SimulationWindow(model, steps=30)
    sim_window.show()
    app.exec()

if __name__ == "__main__":
    main()