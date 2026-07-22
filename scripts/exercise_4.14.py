"""Here we develop a simulation where two predators compete for the same resource.

In this case we construct the following narrative: both lions and cheetahs prey on a population of Gazelle.

We have the following components:

A combined predation-effort:

$$
    E_{t-1} = b_L L_{t-1} + b_{C} C_{t-1}
$$

Gazelle population change:

$$
    x_t = x_{t-1} + r_x x_{t-1} (1 - x_{t-1} / K) - (E_{t-1} / (E_{t-1} + 1)) x_{t-1}
$$

- Logistic growth toward carrying capacity K.
- Minus a joint saturating mortality fraction driven by total lion + cheetah pressure E_{t-1}, capped at 100%.

Lion population change:

$$
    L_t = L_{t-1} - d_L L_{t-1} + (c_L b_L x_{t-1}) / (1 + h_L x_{t-1} + a_L E_{t-1}) L_{t-1}
$$
- Natural decay d_L L_{t-1}
- Growth from a per-capita consumption rate that rises with gazelle abundance x_{t-1}, saturates due to 
    lions' own handling time h_L, and is suppressed by interference from total competitor pressure E_{t-1}
    
Cheetah population change:
$$
    C_t = C_{t-1} - d_C C_{t-1} + (c_C b_C x_{t-1}) / (1 + h_C x_{t-1} + a_C E_{t-1}) C_{t-1}
$$

Same structure as the lion equation but with cheetah specific parameters.

Parameter table:
- r_x:          Gazelle intrinsic growth rate
- K:            carrying capacity
- b_L, b_C:     per-capita hunting effectiveness of lions/cheetahs
- d_L, d_C:     natural decay rate of lions/cheetahs
- c_L, c_C:     conversion efficiency (kills -> growth)
- h_L, h_C:     own-species handling time constant
- a_L, a_C:     sensitivity to interference from total competitor pressure
"""

from dataclasses import dataclass, field
import numpy as np

@dataclass
class Model:
    r_x: float = 1.0
    K: float = 500
    b_L: float = 1.0
    b_C: float = 1.0
    d_L: float = 1.0
    d_C: float = 1.0
    c_L: float = 1.0
    c_C: float = 1.0
    h_L: float = 1.0
    h_C: float = 1.0
    a_L: float = 1.0
    a_C: float = 1.0
    
    x0: float = 1.0
    L0: float = 1.0
    C0: float = 1.0
    
    history: list[float] = field(init=False)
    
    def initialize(self) -> None:
        self.x = self.x0
        self.L = self.L0
        self.C = self.C0
        self.history = [(self.x, self.L, self.C)]
        
    def observe(self) -> None:
        self.history.append((self.x, self.L, self.C))
        
    def update(self) -> None:
        x_prev, L_prev, C_prev = self.x, self.L, self.C
        
        predation_effort = self.b_L * L_prev + self.b_C * C_prev
        
        self.x = x_prev + self.r_x * x_prev * (1 - x_prev / self.K) - (predation_effort / (predation_effort + 1)) * x_prev
        self.L = L_prev - self.d_L * L_prev + (self.c_L * self.b_L * x_prev) / (1 + self.h_L * x_prev + self.a_L * predation_effort) * L_prev
        self.C = C_prev - self.d_C * C_prev + (self.c_C * self.b_C * x_prev) / (1 + self.h_C * x_prev + self.a_C * predation_effort) * C_prev
        
    def run(self, steps: int) -> list[float]:
        self.initialize()
        for _ in range(steps):
            self.update()
            self.observe()
            
        return self.history
    

class SimulationWindow:
    """Plots gazelle/lion/cheetah trajectories and phase space, with sliders
    for every parameter. `r_x` gets its own full-width row (it's not
    species-specific); `b`, `d`, `c`, `h`, `a` each get a row split into a
    lion column (left) and a cheetah column (right)."""

    # (min, max, step) per parameter
    RANGES = {
        "r_x": (0.0, 5.0, 0.01),
        "b_L": (0.0, 5.0, 0.01), "b_C": (0.0, 5.0, 0.01),
        "d_L": (0.0, 1.0, 0.01), "d_C": (0.0, 1.0, 0.01),
        "c_L": (0.0, 5.0, 0.01), "c_C": (0.0, 5.0, 0.01),
        "h_L": (0.0, 5.0, 0.01), "h_C": (0.0, 5.0, 0.01),
        "a_L": (0.0, 5.0, 0.01), "a_C": (0.0, 5.0, 0.01),
    }

    # rows of (lion_param, cheetah_param, shared description)
    PAIRED_ROWS = [
        ("b_L", "b_C", "hunting effectiveness"),
        ("d_L", "d_C", "decay rate"),
        ("c_L", "c_C", "conversion efficiency"),
        ("h_L", "h_C", "handling time"),
        ("a_L", "a_C", "interference sensitivity"),
    ]

    def __init__(self, model: Model, steps: int = 100) -> None:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets

        self._pg = pg
        self._QtCore = QtCore
        self._QtWidgets = QtWidgets

        self.model = model
        self.steps = steps
        self.param_labels = {}
        self.param_descriptions: dict[str, str] = {}

        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle("Lions vs. cheetahs vs. gazelles")
        layout = QtWidgets.QVBoxLayout(self.window)

        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.resize(900, 450)
        layout.addWidget(self.graphics)

        self.trajectory_plot = self.graphics.addPlot(title="Population through time")
        self.trajectory_plot.addLegend()
        self.trajectory_plot.setLabel("bottom", "t")
        self.trajectory_plot.setLabel("left", "population")
        self.x_curve = self.trajectory_plot.plot(pen="g", name="gazelle (x)")
        self.l_curve = self.trajectory_plot.plot(pen="y", name="lion (L)")
        self.c_curve = self.trajectory_plot.plot(pen="c", name="cheetah (C)")

        self.phase_space_plot = self.graphics.addPlot(title="Phase space")
        self.phase_space_plot.addLegend()
        self.phase_space_plot.setLabel("bottom", "gazelle (x)")
        self.phase_space_plot.setLabel("left", "predator population")
        self.l_phase_curve = self.phase_space_plot.plot(pen="y", name="lion (L)")
        self.c_phase_curve = self.phase_space_plot.plot(pen="c", name="cheetah (C)")

        layout.addWidget(self._build_param_widget("r_x", "gazelle intrinsic growth rate"))

        grid = QtWidgets.QGridLayout()
        for row, (lion_param, cheetah_param, description) in enumerate(self.PAIRED_ROWS):
            grid.addWidget(self._build_param_widget(lion_param, f"lion {description}"), row, 0)
            grid.addWidget(self._build_param_widget(cheetah_param, f"cheetah {description}"), row, 1)
        layout.addLayout(grid)

        self._update_plot()

    def _build_param_widget(self, param: str, description: str):
        QtCore, QtWidgets = self._QtCore, self._QtWidgets
        min_, max_, step = self.RANGES[param]

        container = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)

        label = QtWidgets.QLabel()
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setMinimum(round(min_ / step))
        slider.setMaximum(round(max_ / step))
        slider.setValue(round(getattr(self.model, param) / step))
        slider.valueChanged.connect(lambda value, p=param, s=step: self._on_param_changed(p, value, s))

        vbox.addWidget(label)
        vbox.addWidget(slider)

        self.param_labels[param] = label
        self.param_descriptions[param] = description
        return container

    def _on_param_changed(self, param: str, value: int, step: float) -> None:
        setattr(self.model, param, value * step)
        self._update_plot()

    def _update_plot(self) -> None:
        history = np.array(self.model.run(self.steps))
        x, L, C = history[:, 0], history[:, 1], history[:, 2]

        self.x_curve.setData(x)
        self.l_curve.setData(L)
        self.c_curve.setData(C)
        self.l_phase_curve.setData(x, L)
        self.c_phase_curve.setData(x, C)

        for param, label in self.param_labels.items():
            value = getattr(self.model, param)
            label.setText(f"{param} = {value:.3f} ({self.param_descriptions[param]})")

    def show(self) -> None:
        self.window.show()


def main() -> None:
    from pyqtgraph.Qt import QtWidgets

    model = Model()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    sim_window = SimulationWindow(model, steps=100)
    sim_window.show()
    app.exec()

if __name__ == '__main__':
    main()

        