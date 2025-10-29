# Entry point — launches GUI

from src.gui import LandsatClassifierGUI

def main():
    app = LandsatClassifierGUI()
    app.run()

if __name__ == "__main__":
    main()
