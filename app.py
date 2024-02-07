from dashboard.layout import app
from dashboard.pages.home.run_model import DevRunModel
from dashboard.pages.home.label_callback import DevMLLabelCallbacks
from dashboard.pages.home.collapse_callback import DevMLCollapseCallbacks

server = app.server

if __name__ == "__main__":
    # DEV
    DevRunModel()
    DevMLLabelCallbacks()
    DevMLCollapseCallbacks()
    app.run(debug=True)
