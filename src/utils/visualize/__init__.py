
from .center import CentersVisualizeTest, CentersVisualizeTrain
from .orientation import OrientationVisualizeTest, OrientationVisualizeTrain

def get_visualizer(name, opts):
    if name == 'CentersVisualizeTest':
        return CentersVisualizeTest(**opts)
    elif name == 'CentersVisualizeTrain':
        return CentersVisualizeTrain(**opts)
    if name == 'OrientationVisualizeTest':
        return OrientationVisualizeTest(**opts)
    elif name == 'OrientationVisualizeTrain':
        return OrientationVisualizeTrain(**opts)
    else:
        raise Exception("Unknown visualizer: '%s'" % name)