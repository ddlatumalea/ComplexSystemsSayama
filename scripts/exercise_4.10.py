from dataclasses import dataclass, field

@dataclass
class Model:
    a: float = 0.0005
    Z: float = 20
    K: float = 100
    
    x0: float = 20
    
    x: float = field(init=False)
    history: list[float] = field(init=False)
    
    def initialize(self) -> None:
        self.x = self.x0
        self.history = [self.x]
        
    def observe(self) -> None:
        self.history.append(self.x)
        
    def update(self) -> None:
        self.x = self.x + (-self.a * (self.x - self.Z) * (self.x - self.K)) * self.x
        
    def run(self, steps: int) -> list[float]:
        self.initialize()
        for _ in range(steps):
            self.update()
            self.observe()
            
        return self.history
    
class SimulationWindow:
    """Plots the model's trajectory with sliders for `a`, `Z`, `K`, and `x0`."""

    A_MIN, A_MAX, A_STEP = 0.0, 0.01, 0.0001
    Z_MIN, Z_MAX, Z_STEP = 0.0, 200.0, 1.0
    K_MIN, K_MAX, K_STEP = 1.0, 500.0, 1.0
    X0_MIN, X0_MAX, X0_STEP = 0.0, 500.0, 1.0

    def __init__(self, model: Model, steps: int = 30) -> None:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets

        self.model = model
        self.steps = steps

        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle("Growth model with Allee effect")
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

        self.z_label = QtWidgets.QLabel()
        layout.addWidget(self.z_label)
        self.z_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.z_slider.setMinimum(round(self.Z_MIN / self.Z_STEP))
        self.z_slider.setMaximum(round(self.Z_MAX / self.Z_STEP))
        self.z_slider.setValue(round(self.model.Z / self.Z_STEP))
        self.z_slider.valueChanged.connect(self._on_z_changed)
        layout.addWidget(self.z_slider)

        self.k_label = QtWidgets.QLabel()
        layout.addWidget(self.k_label)
        self.k_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.k_slider.setMinimum(round(self.K_MIN / self.K_STEP))
        self.k_slider.setMaximum(round(self.K_MAX / self.K_STEP))
        self.k_slider.setValue(round(self.model.K / self.K_STEP))
        self.k_slider.valueChanged.connect(self._on_k_changed)
        layout.addWidget(self.k_slider)

        self.x0_label = QtWidgets.QLabel()
        layout.addWidget(self.x0_label)
        self.x0_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.x0_slider.setMinimum(round(self.X0_MIN / self.X0_STEP))
        self.x0_slider.setMaximum(round(self.X0_MAX / self.X0_STEP))
        self.x0_slider.setValue(round(self.model.x0 / self.X0_STEP))
        self.x0_slider.valueChanged.connect(self._on_x0_changed)
        layout.addWidget(self.x0_slider)

        self._update_plot()

    def _on_a_changed(self, value: int) -> None:
        self.model.a = value * self.A_STEP
        self._update_plot()

    def _on_z_changed(self, value: int) -> None:
        self.model.Z = value * self.Z_STEP
        self._update_plot()

    def _on_k_changed(self, value: int) -> None:
        self.model.K = value * self.K_STEP
        self._update_plot()

    def _on_x0_changed(self, value: int) -> None:
        self.model.x0 = value * self.X0_STEP
        self._update_plot()

    def _update_plot(self) -> None:
        history = self.model.run(self.steps)
        self.curve.setData(history)
        self.plot_widget.setTitle(
            f"a = {self.model.a:.4f}, Z = {self.model.Z:.0f}, "
            f"K = {self.model.K:.0f}, x0 = {self.model.x0:.0f}"
        )
        self.a_label.setText(f"a = {self.model.a:.4f}")
        self.z_label.setText(f"Z = {self.model.Z:.0f}")
        self.k_label.setText(f"K = {self.model.K:.0f}")
        self.x0_label.setText(f"x0 = {self.model.x0:.0f}")

    def show(self) -> None:
        self.window.show()


def main() -> None:
    model = Model(a=0.0005, Z=21, K=100, x0=20)

    from pyqtgraph.Qt import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    sim_window = SimulationWindow(model, steps=30)
    sim_window.show()
    app.exec()

if __name__ == '__main__':
    main()