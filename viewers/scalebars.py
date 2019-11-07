from qtpy import QtWidgets, QtCore
import math
import pyqtgraph as pg


class WorkingScaleBar(pg.ScaleBar):
    """
    A wrapper for pyqtgraph.ScaleBar that scales properly and allows you to set
    the text value independently from the size.

    See https://github.com/pyqtgraph/pyqtgraph/issues/437 for details on the
    scaling problem with pyqtgraph.ScaleBar. Credit to user sjmvm for the
    solution.

    Attributes:
        size    (float):    The width of the scalebar in view pixels.
        _width   (int):      The width of the scale bar.
        brush
        pen
        offset
    """

    def __init__(self, size, val=-1, width=5, brush=None, pen=None,
                 suffix='px', offset=None):
        """
        Displays a rectangular bar to indicate the relative scale of objects on
        the view.

        Args:
            size (float): The width of the scalebar in view pixels.

        Keyword Args:
            val (float): The value displayed in the scalebar text label.
            width (int): The width of the scale bar rectangle.
            brush
            pen
            suffix (str): The units displayed in the scalebar text label.
            offset (int tuple):
        """
        pg.GraphicsObject.__init__(self)
        pg.GraphicsWidgetAnchor.__init__(self)
        self.setFlag(self.ItemHasNoContents)
        self.setAcceptedMouseButtons(QtCore.Qt.NoButton)

        if brush is None:
            brush = pg.getConfigOption('foreground')
        self.brush = pg.fn.mkBrush(brush)
        self.pen = pg.fn.mkPen(pen)
        self._width = width
        self.size = size
        if offset is None:
            offset = (0, 0)
        self.offset = offset

        self.bar = QtWidgets.QGraphicsRectItem()
        self.bar.setPen(self.pen)
        self.bar.setBrush(self.brush)
        self.bar.setParentItem(self)

        if val == -1:
            val = size

        self.text = pg.TextItem(text=pg.fn.siFormat(val, suffix=suffix),
                                anchor=(0.5, 1))
        self.text.setParentItem(self)

    def parentChanged(self):
        view = self.parentItem()
        if view is None:
            return
        view.sigRangeChanged.connect(self.updateDelay)
        self.updateDelay()

    def updateDelay(self):
        QtCore.QTimer.singleShot(100, self.updateBar)

    def updateBar(self):
        view = self.parentItem()
        if view is None:
            return
        p1 = view.mapFromViewToItem(self, QtCore.QPointF(0, 0))
        p2 = view.mapFromViewToItem(self, QtCore.QPointF(self.size, 0))
        w = (p2-p1).x()
        self.bar.setRect(QtCore.QRectF(-w, 0, w, self._width))
        self.text.setPos(-w/2., 0)


class SEMScaleBar(WorkingScaleBar):
    """
    pyqtgraph.ScaleBar which scales properly and chooses a bar size based on
    a given image size and magnification.

    See https://github.com/pyqtgraph/pyqtgraph/issues/437 for details on the
    scaling problem with pyqtgraph.ScaleBar. Credit to user sjmvm for the
    solution.

    Attributes:
        ref_width (float): Size in meters of the reference used for
                           magnification calculation. Set to Polaroid 545 width
                           of 11.4 cm.
        size    (float):    The width of the scalebar in view pixels.
        _width   (int):      The width of the scale bar.
        brush
        pen
        offset
    """

    # Polaroid 545 width in meters
    ref_width = 114e-3

    def __init__(self, mag=0, num_px=128, width=5, brush=None, pen=None,
                 offset=None):
        """
        Displays a rectangular bar to indicate the relative scale of objects on
        the view with a size based on the magnification and number of pixels.

        Args:
            size (float): The width of the scalebar in view pixels.

        Keyword Args:
            mag (float): Magnification of the SEM image.
            num_px (int): Number of pixels in the image.
            brush
            pen
            offset (int tuple):
        """
        frame_size = self.ref_width/mag
        val = frame_size/5
        ord = math.log10(val)
        val = 10**math.floor(ord) * \
            round(10**(round((ord - math.floor(ord)) * 10) / 10))
        size = val*num_px/frame_size
        if mag < 1:
            mag = 1
            suffix = 'px'
        else:
            suffix = 'm'
        WorkingScaleBar.__init__(self, size, val=val, width=width, pen=pen,
                                 brush=brush, suffix=suffix, offset=offset)


class ConfocalScaleBar(WorkingScaleBar):
    """
    pyqtgraph.ScaleBar which scales properly and chooses a bar size based on
    the number of pixels in the image and the image span in m.

    See https://github.com/pyqtgraph/pyqtgraph/issues/437 for details on the
    scaling problem with pyqtgraph.ScaleBar. Credit to user sjmvm for the
    solution.

    Attributes:
        size    (float):    The width of the scalebar in view pixels.
        _width   (int):      The width of the scale bar.
        brush
        pen
        offset
    """

    def __init__(self, span=-1, num_px=128, width=5, brush=None, pen=None,
                 offset=None):
        """
        Displays a rectangular bar to indicate the relative scale of objects on
        the view with a size based on the image span in meters and number of
        pixels.

        Args:
            size (float): The width of the scalebar in view pixels.

        Keyword Args:
            span (float): The width of the image in meters.
            num_px (int): The width of the image in pixels.
            brush
            pen
            offset (int tuple):
        """

        if span > 0:
            ord = math.log10(span/5)
            val = 10**math.floor(ord) * \
                round(10**(round((ord - math.floor(ord)) * 10) / 10))
            size = val*num_px/span
            suffix = 'm'
        else:
            size = num_px
            val = num_px
            suffix = 'px'

        WorkingScaleBar.__init__(self, size, val=val, width=width, pen=pen,
                                 brush=brush, suffix=suffix, offset=offset)
