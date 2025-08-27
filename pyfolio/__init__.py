from pyfolio_reloaded import *

# Optional: make sure submodules are available like in original pyfolio
import sys
import pyfolio_reloaded as _pf

sys.modules[__name__ + ".tears"] = _pf.tears
sys.modules[__name__ + ".timeseries"] = _pf.timeseries
sys.modules[__name__ + ".plotting"] = _pf.plotting
sys.modules[__name__ + ".pos"] = _pf.pos
