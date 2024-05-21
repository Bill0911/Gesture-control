from utils.cv import CV
from utils import brightness_controll, tab_switch, scroll


def main():
    cv = CV()

    cv(brightness_controll.process, tab_switch.process, scroll.process)


if __name__ == "__main__":
    main()
