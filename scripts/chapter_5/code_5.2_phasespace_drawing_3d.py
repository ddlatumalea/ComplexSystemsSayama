from dataclasses import dataclass, field
import numpy as np

@dataclass
class Model:
    x0: float = 1
    y0: float = 1
    z0: float = 1
    
    history: list[float] = field(init=False)
    
    def initialize(self) -> None:
        self.x = self.x0
        self.y = self.y0
        self.z = self.z0
        self.history = [(self.x, self.y, self.z)]
        
    def observe(self) -> None:
        self.history.append((self.x, self.y, self.z))
        
    def update(self) -> None:
        x_prev, y_prev, z_prev = self.x, self.y, self.z
        
        self.x = 0.5 * x_prev + y_prev
        self.y = -0.5 * x_prev + y_prev
        self.z = -x_prev - y_prev + z_prev
        
    def run(self, steps: int) -> None:
        self.initialize()
        for _ in range(steps):
            self.update()
            self.observe()
            
        return self.history
    
def main(steps: int = 30) -> None:
    """Run the model from a grid of (x0, y0, z0) initial conditions in
    [-2, 2) and overlay every resulting trajectory on one 3D phase-space
    plot."""
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from pyqtgraph.Qt import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    view = gl.GLViewWidget()
    view.setWindowTitle("3D phase space")
    view.setCameraPosition(distance=40)
    view.addItem(gl.GLGridItem())
    view.addItem(gl.GLAxisItem())

    initial_conditions = np.arange(-2, 2, 1)
    for x0 in initial_conditions:
        for y0 in initial_conditions:
            for z0 in initial_conditions:
                model = Model(x0=x0, y0=y0, z0=z0)
                history = np.array(model.run(steps))
                color = pg.glColor(pg.intColor(hash((x0, y0, z0)) % 256, 256))
                line = gl.GLLinePlotItem(pos=history, color=color, width=1, antialias=True)
                view.addItem(line)

    view.show()
    app.exec()

if __name__ == '__main__':
    main()