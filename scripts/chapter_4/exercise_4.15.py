from dataclasses import dataclass, field
import numpy as np

@dataclass
class Model:
    r: float = 0.1 # scaling
    
    c0: float = 0.5
    l0: float = 0.2
    
    history: list[float] = field(init=False)
    
    def initialize(self) -> None:
        self.c = self.c0
        self.l = self.l0
        self.n = 1 - self.c - self.l 
        
        self.history = [(self.c, self.l, self.n)]
        
    def observe(self) -> None:
        self.history.append((self.c, self.l, self.n))
        
    def update(self) -> None:
        c_prev, l_prev, n_prev = self.c, self.l, self.n
        
        self.c = c_prev + self.r * max(c_prev - l_prev, 0) * l_prev \
                        + self.r * max(c_prev - n_prev, 0) * n_prev \
                        - self.r * max(l_prev - c_prev, 0) * c_prev \
                        - self.r * max(n_prev - c_prev, 0) * c_prev
    
        self.l = l_prev + self.r * max(l_prev - c_prev, 0) * c_prev \
                        + self.r * max(l_prev - n_prev, 0) * n_prev \
                        - self.r * max(c_prev - l_prev, 0) * l_prev \
                        - self.r * max(n_prev - l_prev, 0) * l_prev
                        
        self.n = 1 - self.c - self.l
        
    def run(self, steps: int) -> list[float]:
        self.initialize()
        for _ in range(steps):
            self.update()
            self.observe()
            
        return self.history
                     
class SimulationWindow:
    """Plots c, l, n through time with sliders for `c0` and `l0`.

    c0 and l0 are coupled: both must stay in [0, 1] and c0 + l0 <= 1
    (since n0 = 1 - c0 - l0 must itself stay in [0, 1]). Moving one
    slider past the remaining budget clamps the other down to fit.
    """

    C_MIN, C_MAX, C_STEP = 0.0, 1.0, 0.01
    L_MIN, L_MAX, L_STEP = 0.0, 1.0, 0.01
    R_MIN, R_MAX, R_STEP = 0.0, 1.0, 0.01

    def __init__(self, model: Model, steps: int = 30) -> None:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets

        self.model = model
        self.steps = steps

        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle("c, l, n through time")
        layout = QtWidgets.QVBoxLayout(self.window)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.addLegend()
        self.plot_widget.setLabel("bottom", "t")
        self.plot_widget.setLabel("left", "fraction")
        self.plot_widget.setYRange(0, 1, padding=0)
        self.plot_widget.enableAutoRange("y", False)
        self.c_curve = self.plot_widget.plot(pen="r", name="c")
        self.l_curve = self.plot_widget.plot(pen="g", name="l")
        self.n_curve = self.plot_widget.plot(pen="b", name="n")
        layout.addWidget(self.plot_widget)

        self.c_label = QtWidgets.QLabel()
        layout.addWidget(self.c_label)
        self.c_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.c_slider.setMinimum(round(self.C_MIN / self.C_STEP))
        self.c_slider.setMaximum(round(self.C_MAX / self.C_STEP))
        self.c_slider.setValue(round(self.model.c0 / self.C_STEP))
        self.c_slider.valueChanged.connect(self._on_c_changed)
        layout.addWidget(self.c_slider)

        self.l_label = QtWidgets.QLabel()
        layout.addWidget(self.l_label)
        self.l_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.l_slider.setMinimum(round(self.L_MIN / self.L_STEP))
        self.l_slider.setMaximum(round(self.L_MAX / self.L_STEP))
        self.l_slider.setValue(round(self.model.l0 / self.L_STEP))
        self.l_slider.valueChanged.connect(self._on_l_changed)
        layout.addWidget(self.l_slider)

        self.r_label = QtWidgets.QLabel()
        layout.addWidget(self.r_label)
        self.r_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.r_slider.setMinimum(round(self.R_MIN / self.R_STEP))
        self.r_slider.setMaximum(round(self.R_MAX / self.R_STEP))
        self.r_slider.setValue(round(self.model.r / self.R_STEP))
        self.r_slider.valueChanged.connect(self._on_r_changed)
        layout.addWidget(self.r_slider)

        self._update_plot()

    def _on_c_changed(self, value: int) -> None:
        self.model.c0 = value * self.C_STEP
        if self.model.c0 + self.model.l0 > 1.0:
            self.model.l0 = 1.0 - self.model.c0
            self.l_slider.blockSignals(True)
            self.l_slider.setValue(round(self.model.l0 / self.L_STEP))
            self.l_slider.blockSignals(False)
        self._update_plot()

    def _on_l_changed(self, value: int) -> None:
        self.model.l0 = value * self.L_STEP
        if self.model.c0 + self.model.l0 > 1.0:
            self.model.c0 = 1.0 - self.model.l0
            self.c_slider.blockSignals(True)
            self.c_slider.setValue(round(self.model.c0 / self.C_STEP))
            self.c_slider.blockSignals(False)
        self._update_plot()

    def _on_r_changed(self, value: int) -> None:
        self.model.r = value * self.R_STEP
        self._update_plot()

    def _update_plot(self) -> None:
        history = np.array(self.model.run(self.steps))
        self.c_curve.setData(history[:, 0])
        self.l_curve.setData(history[:, 1])
        self.n_curve.setData(history[:, 2])

        self.c_label.setText(f"c0 = {self.model.c0:.2f}")
        self.l_label.setText(f"l0 = {self.model.l0:.2f}")
        self.r_label.setText(f"r = {self.model.r:.2f}")

    def show(self) -> None:
        self.window.show()


def main() -> None:
    model = Model()

    from pyqtgraph.Qt import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    sim_window = SimulationWindow(model, steps=100)
    sim_window.show()
    app.exec()

if __name__ == '__main__':
    main()