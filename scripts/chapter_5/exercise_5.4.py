from dataclasses import dataclass, field
import numpy as np

@dataclass
class Model:
    x0: float = 1
    y0: float = 1
    
    history: list[float] = field(init=False)
    
    def initialize(self) -> None:
        self.x = self.x0
        self.y = self.y0
        self.history = [(self.x, self.y)]
        
    def observe(self) -> None:
        self.history.append((self.x, self.y))
        
    def update(self) -> None:
        x_prev, y_prev = self.x, self.y
        self.x = x_prev + 0.1 * (x_prev - x_prev * y_prev)
        self.y = y_prev + 0.1 * (y_prev - x_prev * y_prev)

    def run(self, steps: int) -> list[float]:
        self.initialize()
        for _ in range(steps):
            self.update()
            self.observe()

        return self.history


class SimulationWindow:
    """Plots the phase space (x vs y) with sliders for `x0` and `y0`."""

    X0_MIN, X0_MAX, X0_STEP = -10.0, 10.0, 0.01
    Y0_MIN, Y0_MAX, Y0_STEP = -10.0, 10.0, 0.01

    def __init__(self, model: Model, steps: int = 30) -> None:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets

        self.model = model
        self.steps = steps

        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle("Phase space")
        layout = QtWidgets.QVBoxLayout(self.window)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("bottom", "x")
        self.plot_widget.setLabel("left", "y")
        self.curve = self.plot_widget.plot(pen="g", symbol="o")
        layout.addWidget(self.plot_widget)

        self.x0_label = QtWidgets.QLabel()
        layout.addWidget(self.x0_label)
        self.x0_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.x0_slider.setMinimum(round(self.X0_MIN / self.X0_STEP))
        self.x0_slider.setMaximum(round(self.X0_MAX / self.X0_STEP))
        self.x0_slider.setValue(round(self.model.x0 / self.X0_STEP))
        self.x0_slider.valueChanged.connect(self._on_x0_changed)
        layout.addWidget(self.x0_slider)

        self.y0_label = QtWidgets.QLabel()
        layout.addWidget(self.y0_label)
        self.y0_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.y0_slider.setMinimum(round(self.Y0_MIN / self.Y0_STEP))
        self.y0_slider.setMaximum(round(self.Y0_MAX / self.Y0_STEP))
        self.y0_slider.setValue(round(self.model.y0 / self.Y0_STEP))
        self.y0_slider.valueChanged.connect(self._on_y0_changed)
        layout.addWidget(self.y0_slider)

        self._update_plot()

    def _on_x0_changed(self, value: int) -> None:
        self.model.x0 = value * self.X0_STEP
        self._update_plot()

    def _on_y0_changed(self, value: int) -> None:
        self.model.y0 = value * self.Y0_STEP
        self._update_plot()

    def _update_plot(self) -> None:
        history = np.array(self.model.run(self.steps))
        self.curve.setData(history[:, 0], history[:, 1])

        self.plot_widget.setTitle(f"x0 = {self.model.x0:.2f}, y0 = {self.model.y0:.2f}")
        self.x0_label.setText(f"x0 = {self.model.x0:.2f}")
        self.y0_label.setText(f"y0 = {self.model.y0:.2f}")

    def show(self) -> None:
        self.window.show()


def plot_phase_portrait(steps: int = 30) -> None:
    """Run the model from a grid of (x0, y0) initial conditions in [0, 10)
    and overlay every resulting trajectory on one phase-space plot."""
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    plot = pg.plot(title="Phase portrait")
    plot.setLabel("bottom", "x")
    plot.setLabel("left", "y")

    for x0 in np.arange(0, 2, 0.1):
        for y0 in np.arange(0, 2, 0.1):
            model = Model(x0=x0, y0=y0)
            history = np.array(model.run(steps))
            plot.plot(history[:, 0], history[:, 1], pen=pg.mkPen(width=1))

    app.exec()


def main() -> None:
    model = Model()

    from pyqtgraph.Qt import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    sim_window = SimulationWindow(model, steps=30)
    sim_window.show()
    app.exec()

if __name__ == '__main__':
    # main()
    plot_phase_portrait()