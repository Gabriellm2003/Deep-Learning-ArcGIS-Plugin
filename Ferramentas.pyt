import arcpy
import subprocess
import sys
import os
import tempfile
import importlib
import imp

import detectionTools
import segmentationTools
imp.reload(detectionTools)
imp.reload(segmentationTools)

from detectionTools import FineTunningDet as FineTunningDet
from detectionTools import AnalyzeImgsDet as AnalyzeImgsDet
from segmentationTools import FineTunningSeg as FineTunningSeg
from segmentationTools import AnalyzeImgsSeg as AnalyzeImgsSeg


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "tcutb"
        self.alias = ""

        # List of tool classes associated with this toolbox
        self.tools = [FineTunningSeg, AnalyzeImgsSeg, FineTunningDet, AnalyzeImgsDet]

