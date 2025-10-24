from .xdfdata import XdfData
from .antxdfdata import AntXdfData

xfd_classes = {
    'XdfData': XdfData,
    'AntXdfData': AntXdfData,
}

def get_xdf_class(classname):
    return xfd_classes[classname]