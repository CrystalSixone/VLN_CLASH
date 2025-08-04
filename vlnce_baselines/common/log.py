import logging


class HabitatLogger(logging.Logger):
    def __init__(
        self,
        name,
        level,
        filename=None,
        filemode="a",
        stream=None,
        format_str=None,
        dateformat=None,
        style="%",
    ):
        super().__init__(name, level)
        self._formatter = logging.Formatter(format_str, dateformat, style)

        if filename is not None:
            file_handler = logging.FileHandler(filename, filemode)  # type:ignore
            file_handler.setFormatter(self._formatter)
            super().addHandler(file_handler)

        console_handler = logging.StreamHandler(stream)  # type:ignore
        console_handler.setFormatter(self._formatter)
        super().addHandler(console_handler)

    def add_filehandler(self, log_filename):
        filehandler = logging.FileHandler(log_filename)
        filehandler.setFormatter(self._formatter)
        self.addHandler(filehandler)
