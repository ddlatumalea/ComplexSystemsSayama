"""This is an implementation of exercise 4.11 where we have:
A prey population:

$$
x_t = x_{t-1} + rx_{t-1}(1 - (x_{t-1} / K)) - (1 - (1 / (by_{t-1} + 1)x_{t-1}))
$$

Where:
- x_t is the total prey population at time t
- x_{t-1} is the total prey population at time t-1
- rx_{t-1}(1 - (x_{t-1} / K)) consists of two aspects:
    rx_{t-1} refers to the intrinsic growth rate of the population. 
    (1 - (x_{t-1} / K)) refers to the population size that is sustainable. If x_{t-1} << K, then this term is approximately 1. If x_{t-1} -> K, it shrinks to 0. If x_{t-1} > K, it turns negative and actively shrinks the population.
    Thus, the full term refers to the intrinsic growth rate multiplied by what the environment allows. 
- (1 - (1 / (by_{t-1} + 1)x_{t-1})) refers to the predation loss term. b controls how mortality saturates.

A predator population:

$$
    y_t = y_{t-1} - dy_{t-1} + cx_{t-1}y_{t-1}
$$
 
Where:
- y_t is the total predator population at time t
- y_{t-1} is the total predator population at time t-1
- dy_{t-1} refers to exponential decay.
- cx_{t-1}y_{t-1} refers to an interaction between prey and predator where c is the conversion from interaction to reproduction.

There are parameters we have to initialize:
Prey
- b: how quickly the mortality saturates. The mortality function reaches half of its maximum value when y = 1 / b
- r: inrinsic growth rate.

Predator
- d: the rate of exponential decay. We can rewrite $y_{t-1} - dy_{t-1}$ as y_{t-1}(1 - d). This gives us a bound for d. $d$ can not be negative. Therefore, d = [0, 1].
    if $d$ = 0, there is no decay; if $d$ = 1, all animals die within one timestep t.
- c: the conversion rate of encounter to reproduction. $c$ can be $[0, \inf]$. However, c > 0, is more realistic otherwise the encounter term will result in 0.
"""

from dataclasses import dataclass, field
import numpy as np

@dataclass
class Model:
    r: float = 0.3 # Intrinsic growth rate
    b: float = 0.2 # Mortality saturation
    K: float = 500 # Maximum population capacity
    c: float = 0.006 # conversion efficiency for encounter -> reproduction
    d: float = 0.1 # Decay term
    
    x0: float = 10
    y0: float  = 2
    
    history: list[float] = field(init=False)
    
    def initialize(self) -> None:
        self.x = self.x0
        self.y = self.y0
        self.history = [(self.x, self.y)]
        
    def observe(self) -> None:
        self.history.append((self.x, self.y))
        
    def update(self) -> None:
        x_prev, y_prev = self.x, self.y
        predation_loss = x_prev * (1 - 1 / (self.b * y_prev + 1))

        self.x = x_prev + self.r * x_prev * (1 - x_prev / self.K) - predation_loss
        self.y = y_prev - self.d * y_prev + self.c * x_prev * y_prev
        
    def run(self, steps: int) -> list[float]:
        self.initialize()
        for _ in range(steps):
            self.update()
            self.observe()
            
        return self.history
    
class SimulationWindow:
    """Plots prey/predator trajectories with sliders for `r`, `b`, `K`, `c`, and `d`."""

    R_MIN, R_MAX, R_STEP = 0.0, 5.0, 0.01
    B_MIN, B_MAX, B_STEP = 0.01, 2.0, 0.01
    K_MIN, K_MAX, K_STEP = 1.0, 2000.0, 1.0
    C_MIN, C_MAX, C_STEP = 0.0, 0.5, 0.0001
    D_MIN, D_MAX, D_STEP = 0.0, 1.0, 0.01

    def __init__(self, model: Model, steps: int = 100) -> None:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets

        self.model = model
        self.steps = steps

        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle("Predator-prey model")
        layout = QtWidgets.QVBoxLayout(self.window)

        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.resize(900, 450)
        layout.addWidget(self.graphics)

        self.trajectory_plot = self.graphics.addPlot(title="Trajectories")
        self.trajectory_plot.addLegend()
        self.trajectory_plot.setLabel("bottom", "t")
        self.trajectory_plot.setLabel("left", "population")
        self.x_curve = self.trajectory_plot.plot(pen="r", name="prey (x)")
        self.y_curve = self.trajectory_plot.plot(pen="b", name="predator (y)")

        self.phase_space_plot = self.graphics.addPlot(title="Phase space")
        self.phase_space_plot.setLabel("bottom", "prey (x)")
        self.phase_space_plot.setLabel("left", "predator (y)")
        self.phase_curve = self.phase_space_plot.plot(pen="g")

        self.r_label = QtWidgets.QLabel()
        layout.addWidget(self.r_label)
        self.r_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.r_slider.setMinimum(round(self.R_MIN / self.R_STEP))
        self.r_slider.setMaximum(round(self.R_MAX / self.R_STEP))
        self.r_slider.setValue(round(self.model.r / self.R_STEP))
        self.r_slider.valueChanged.connect(self._on_r_changed)
        layout.addWidget(self.r_slider)

        self.b_label = QtWidgets.QLabel()
        layout.addWidget(self.b_label)
        self.b_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.b_slider.setMinimum(round(self.B_MIN / self.B_STEP))
        self.b_slider.setMaximum(round(self.B_MAX / self.B_STEP))
        self.b_slider.setValue(round(self.model.b / self.B_STEP))
        self.b_slider.valueChanged.connect(self._on_b_changed)
        layout.addWidget(self.b_slider)

        self.k_label = QtWidgets.QLabel()
        layout.addWidget(self.k_label)
        self.k_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.k_slider.setMinimum(round(self.K_MIN / self.K_STEP))
        self.k_slider.setMaximum(round(self.K_MAX / self.K_STEP))
        self.k_slider.setValue(round(self.model.K / self.K_STEP))
        self.k_slider.valueChanged.connect(self._on_k_changed)
        layout.addWidget(self.k_slider)

        self.c_label = QtWidgets.QLabel()
        layout.addWidget(self.c_label)
        self.c_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.c_slider.setMinimum(round(self.C_MIN / self.C_STEP))
        self.c_slider.setMaximum(round(self.C_MAX / self.C_STEP))
        self.c_slider.setValue(round(self.model.c / self.C_STEP))
        self.c_slider.valueChanged.connect(self._on_c_changed)
        layout.addWidget(self.c_slider)

        self.d_label = QtWidgets.QLabel()
        layout.addWidget(self.d_label)
        self.d_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.d_slider.setMinimum(round(self.D_MIN / self.D_STEP))
        self.d_slider.setMaximum(round(self.D_MAX / self.D_STEP))
        self.d_slider.setValue(round(self.model.d / self.D_STEP))
        self.d_slider.valueChanged.connect(self._on_d_changed)
        layout.addWidget(self.d_slider)

        self._update_plot()

    def _on_r_changed(self, value: int) -> None:
        self.model.r = value * self.R_STEP
        self._update_plot()

    def _on_b_changed(self, value: int) -> None:
        self.model.b = value * self.B_STEP
        self._update_plot()

    def _on_k_changed(self, value: int) -> None:
        self.model.K = value * self.K_STEP
        self._update_plot()

    def _on_c_changed(self, value: int) -> None:
        self.model.c = value * self.C_STEP
        self._update_plot()

    def _on_d_changed(self, value: int) -> None:
        self.model.d = value * self.D_STEP
        self._update_plot()

    def _update_plot(self) -> None:
        history = np.array(self.model.run(self.steps))
        self.x_curve.setData(history[:, 0])
        self.y_curve.setData(history[:, 1])
        self.phase_curve.setData(history[:, 0], history[:, 1])

        self.r_label.setText(f"r = {self.model.r:.2f} (intrinsic growth rate)")
        self.b_label.setText(f"b = {self.model.b:.2f} (mortality saturation)")
        self.k_label.setText(f"K = {self.model.K:.0f} (maximum capacity)")
        self.c_label.setText(f"c = {self.model.c:.4f} (conversion encounter to reproduction)")
        self.d_label.setText(f"d = {self.model.d:.2f} (exponential decay rate)")

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