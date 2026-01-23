import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from MAR.SystemRouter.training import main


if __name__ == "__main__":
    main(default_dataset="mmlu")
