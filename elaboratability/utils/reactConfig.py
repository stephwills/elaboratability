from aizynthfinder.context.config import Configuration
from aizynthfinder.context.policy import QuickKerasFilter


AIZYNTH_CONFIG="/home/swills/Oxford/elaboratability/data/aizynthfinder/config.yml"
FILTER_MODEL="/home/swills/Oxford/elaboratability/data/aizynthfinder/uspto_filter_model.onnx"
FILTER_CUTOFF = 0.05
CHECK_ALL_NON_CLASHING = False  # this will check whether all non-clashing elaborations will react - this is slow!
aizynth_config = Configuration.from_file(AIZYNTH_CONFIG)
filter = QuickKerasFilter(key="filter",
                          config=aizynth_config,
                          model=FILTER_MODEL
                          )